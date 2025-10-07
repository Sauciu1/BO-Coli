from ax.core import parameter
from src.BayesClientManager import BayesClientManager
import pandas as pd
import streamlit as st
import ax

import numpy as np


class InitExperiment:
    def __init__(self, Bayes_manager: BayesClientManager = BayesClientManager):
        self.bayes_manager = Bayes_manager

        self.experiment_parameters = {}



    @property
    def selected_gp(self):
        selected_name = st.session_state.get("selected_gp", None)
        if selected_name is None:
            st.warning("Invalid GP selection")
        return selected_name


    @property
    def selected_acquisition_function(self):
        selected_name = st.session_state.get("selected_acq", None)
        if selected_name is None:
            st.warning("Invalid acquisition function selection")
        return selected_name
    
    
    @property
    def client(self):
        return self.bayes_manager._create_ax_client()

    @st.fragment
    def _choose_gaussian_process(self):
        """Choose the Gaussian Process model for the experiment"""
        gp_options = list(BayesClientManager.gp_options.fget().keys())
        if "selected_gp" not in st.session_state:
            st.session_state.selected_gp = gp_options[0]

        st.session_state.selected_gp = st.selectbox(
            "Select a Gaussian Process model:",
            list(gp_options),
            index=list(gp_options).index(st.session_state.selected_gp),
        )



    @st.fragment
    def _choose_acquisition_function(self):
        """Choose Acquisition Function"""
        if "selected_acq" not in st.session_state:
            st.session_state.selected_acq = "qLogExpectedImprovement"

        acqf_options = list(BayesClientManager.acq_f_options.fget().keys())

        st.session_state.selected_acq = st.selectbox(
            "Select an acquisition function:",
            list(acqf_options),
            index=list(acqf_options).index(st.session_state.selected_acq),
        )



    def _model_choice_ui(self):
        """Wrapper to choose both GP model and acquisition function"""
        st.write("### Model and Acquisition Function Selection")
        columns = st.columns(2)
        with columns[0]:
            self._choose_gaussian_process()
        with columns[1]:
            self._choose_acquisition_function()

    @st.fragment
    def parameter_handler(self):
        """Handles all parameter related UI elements"""
        st.subheader("Parameter Configuration")

        if "parameters" not in st.session_state:
            st.session_state.parameters = []
        if "param_counter" not in st.session_state:
            st.session_state.param_counter = 0

        if "params_to_delete" in st.session_state:
            for param_id in st.session_state.params_to_delete:
                st.session_state.parameters = [
                    p for p in st.session_state.parameters if p.get("id") != param_id
                ]
            del st.session_state.params_to_delete
            st.rerun()

        # Display existing parameters
        for i, param in enumerate(st.session_state.parameters):
            # Ensure each parameter has a unique ID
            if "id" not in param:
                param["id"] = st.session_state.param_counter
                st.session_state.param_counter += 1

            param_id = param["id"]
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1, 1])
                with col1:
                    param["name"] = st.text_input(
                        f"Parameter {i+1} Name",
                        param["name"],
                        key=f"name_{param_id}",
                       # value="",
                    )
                with col2:
                    param["lower_bound"] = st.number_input(
                        "Lower Bound",
                        value=param["lower_bound"],
                        key=f"lower_{param_id}",
                        step=1e-10,
                        format="%.2e",
                       # value=np.nan,
                    )
                with col3:
                    param["upper_bound"] = st.number_input(
                        "Upper Bound",
                        value=param["upper_bound"],
                        key=f"upper_{param_id}",
                        step=1e-10,
                        format="%.2e",
                       # value=np.nan,
                    )
                with col4:
                    param["log_scale"] = st.checkbox(
                        "Log Scale", param["log_scale"], key=f"log_{param_id}"
                    )
                with col5:
                    if st.button(
                        "🗑️", key=f"delete_{param_id}", help="Delete parameter"
                    ):
                        if "params_to_delete" not in st.session_state:
                            st.session_state.params_to_delete = []
                        st.session_state.params_to_delete.append(param_id)
                        st.rerun()

        # Add new parameter button
        if st.button("➕ Add Parameter"):
            new_param = {
                "id": st.session_state.param_counter,
                "name": f"param_{len(st.session_state.parameters)+1}",
                "lower_bound": np.nan,
                "upper_bound": np.nan,
                "log_scale": False,
            }
            st.session_state.parameters.append(new_param)
            st.session_state.param_counter += 1
            st.rerun(scope="fragment")

    @property
    def objective_direction(self):
        return st.session_state.objective_direction

    @st.fragment
    def get_session_parameters(self):
        """Returns a list of parameters as dicts"""
        if "parameters" not in st.session_state:
            return []

        # Convert to Ax RangeParameterConfig format
        ax_parameters = {}


        for param in st.session_state.parameters:
            if param["name"] and param["lower_bound"] < param["upper_bound"]:
                ax_parameters[param["name"]] = {
  
                    'lower_bound': param["lower_bound"],
                    'upper_bound': param["upper_bound"],
                    "log_scale": param["log_scale"]
                }
            else:
                st.warning(f"Parameter '{param['name']}' has invalid bounds")

        return ax_parameters

    @property
    def experiment_name(self):
        return st.session_state.experiment_name

    @st.fragment
    def create_experiment(self):
        """Allows to manually name a parameter/dimension,
        set its range and whether the axis is logarithmic"""
        st.markdown("### Define Experiment Parameters")
        st.write("Configure the parameters for your Bayesian optimization experiment.")

        st.text_input("Experiment Name", "My Experiment", key="experiment_name")

        self._model_choice_ui()

        # Objective direction switch
        if "objective_direction" not in st.session_state:
            st.session_state.objective_direction = "Maximize"

        st.session_state.objective_direction = st.radio(
            "Objective Direction",
            ["Maximize", "Minimize"],
            index=0,
            horizontal=True,
        )

        self.parameter_handler()
        params = self.get_session_parameters()
        param_df = pd.DataFrame(params)

        # Create experiment button
        if st.button("Create Experiment", type="secondary"):
            if len(params) == len(st.session_state.parameters) and len(params) > 0:
                # Create the client immediately
                self.bayes_manager = self.CreateClientManager()
                st.session_state.bayes_manager = self.bayes_manager
                self.experiment_parameters = params
                st.session_state.experiment_configured = True

                
                st.success("Experiment configured!")
                st.rerun()

               # st.rerun()

            elif len(params) == 0:
                st.error("Please add at least one parameter to create an experiment.")
            else:
                st.error("Please ensure all parameters have valid names and bounds.")
        
        # Show parameter summary after experiment is created
        if st.session_state.get("experiment_configured", False) and st.session_state.get("client") is not None:
            st.write("### Parameter Summary")
            st.success(
                f"✅ Experiment configured: **{self.experiment_name}** to **{self.objective_direction.lower()}**"
                + f" the objective with **{len(params)}** parameters!"
            )
            st.dataframe(param_df, use_container_width=True)


    def CreateClientManager(self):
        params  = self.get_session_parameters()
        input_labels = [param for param in params.keys()]
        empty_df = pd.DataFrame(columns=input_labels + ["response"])

        bayes_manager = BayesClientManager(
            data = empty_df,
            feature_labels = input_labels,
            response_label = "response",
            bounds = params
        )

        bayes_manager._experiment_name = self.experiment_name

        bayes_manager.gp = self.selected_gp
        bayes_manager.acquisition_function = self.selected_acquisition_function
        return bayes_manager


if __name__ == "__main__":

    if "manager" not in st.session_state:
        st.set_page_config(
            page_title="Start New Experiment",

            layout="wide",
            initial_sidebar_state="collapsed",
        )

        exp = InitExperiment().create_experiment()

        

        print()
