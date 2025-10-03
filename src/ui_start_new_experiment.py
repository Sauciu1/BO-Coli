from ax.core import parameter
from src.ax_helper import BayesClientManager
import pandas as pd
import streamlit as st
import ax
from ax import Client, RangeParameterConfig
from src.model_generation import HeteroWhiteSGP
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import GammaPrior
from botorch.acquisition import (
    qLogNoisyExpectedImprovement,
    ExpectedImprovement,
    UpperConfidenceBound,
    ProbabilityOfImprovement,
)
import numpy as np


class new_experiment:
    def __init__(self):

        self.experiment_parameters = {}
        st.session_state.client = None


    @property
    def gp(self):
        gp_options = {
            "SingleTaskGP": SingleTaskGP,
            "Heteroscedastic White GP": HeteroWhiteSGP,
        }
        selected_name = st.session_state.get("selected_gp", "SingleTaskGP")
        return gp_options.get(selected_name, SingleTaskGP)

    @property
    def acquisition_function(self):
        acq_options = {
            "Expected Improvement": ExpectedImprovement,
            "Upper Confidence Bound": UpperConfidenceBound,
            "Probability of Improvement": ProbabilityOfImprovement,
            "Noisy Expected Improvement": qLogNoisyExpectedImprovement,
        }
        selected_name = st.session_state.get("selected_acq", "Expected Improvement")
        return acq_options.get(selected_name, ExpectedImprovement)
    
    @property
    def objective_direction(self):
        return st.session_state.get("objective_direction", "Maximize")
    
    @property
    def client(self):
        return st.session_state.client

    @st.fragment
    def _choose_gaussian_process(self):
        """Allows to choose the Gaussian Process model for the experiment"""
       # st.subheader("Choose Gaussian Process Model")

        gp_options = {
            "SingleTaskGP": SingleTaskGP,
            "Heteroscedastic White GP": HeteroWhiteSGP,
        }

        if "selected_gp" not in st.session_state:
            st.session_state.selected_gp = "SingleTaskGP"

        st.session_state.selected_gp = st.selectbox(
            "Select a Gaussian Process model:",
            list(gp_options.keys()),
            index=list(gp_options.keys()).index(st.session_state.selected_gp),
        )



    @st.fragment
    def _choose_acquisition_function(self):
       # st.subheader("Choose Acquisition Function")

        acq_options = {
            "Expected Improvement": ExpectedImprovement,
            "Probability of Improvement": ProbabilityOfImprovement,
            "Upper Confidence Bound": UpperConfidenceBound,
            "q Log Noisy Expected Improvement": qLogNoisyExpectedImprovement,
        }

        if "selected_acq" not in st.session_state:
            st.session_state.selected_acq = "Expected Improvement"

        st.session_state.selected_acq = st.selectbox(
            "Select an acquisition function:",
            list(acq_options.keys()),
            index=list(acq_options.keys()).index(st.session_state.selected_acq),
        )

        st.session_state.acquisition_function = acq_options[st.session_state.selected_acq]


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

        # Initialize parameters in session state if not exists
        if "parameters" not in st.session_state:
            st.session_state.parameters = []
        if "param_counter" not in st.session_state:
            st.session_state.param_counter = 0

        # Check for parameters marked for deletion
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
                    param["lower"] = st.number_input(
                        "Lower Bound",
                        value=param["lower"],
                        key=f"lower_{param_id}",
                        step=1e-10,
                        format="%.2e",
                       # value=np.nan,
                    )
                with col3:
                    param["upper"] = st.number_input(
                        "Upper Bound",
                        value=param["upper"],
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
                "lower": np.nan,
                "upper": np.nan,
                "log_scale": False,
            }
            st.session_state.parameters.append(new_param)
            st.session_state.param_counter += 1
            st.rerun(scope="fragment")

    @property
    def objective_direction(self):
        return st.session_state.objective_direction

    @st.fragment
    def get_parameters(self):
        """Returns a list of parameters as dicts"""
        if "parameters" not in st.session_state:
            return []

        # Convert to Ax RangeParameterConfig format
        ax_parameters = []
        for param in st.session_state.parameters:
            if param["name"] and param["lower"] < param["upper"]:
                ax_param = {
                    "name": param["name"],
                    "lower_bound": param["lower"],
                    "upper_bound": param["upper"],
                    "log_scale": param["log_scale"],
                }
                ax_parameters.append(ax_param)
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
        params = self.get_parameters()
        param_df = pd.DataFrame(params)

        # Create experiment button
        if st.button("Create Experiment", type="secondary"):
            if len(params) == len(st.session_state.parameters) and len(params) > 0:
                # Create the client immediately
                st.session_state.client = self.write_params_to_client()
                self.experiment_parameters = params
                
                # Set a flag to show we've created the experiment
                st.session_state.experiment_configured = True
                st.rerun()

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



    def write_params_to_client(self):
        client = Client()

        range_parameters = [
            RangeParameterConfig(
                name=param['name'],
                parameter_type="float",
                bounds=(param['lower_bound'], param['upper_bound']),
                scaling="log" if param['log_scale'] else "linear",
            )
            for param in self.get_parameters()
        ]

        client.configure_experiment(
            name="batch_bo_test",
            parameters=range_parameters,
        )

        client.configure_optimization(
            objective="-loss" if self.objective_direction == "Maximize" else "loss"
        )
        st.session_state.client = client
        return client


if __name__ == "__main__":
    if "manager" not in st.session_state:
        st.set_page_config(
            page_title="Start New Experiment",
            page_icon="🧪",
            layout="wide",
            initial_sidebar_state="collapsed",
        )
        st.title("🧪 Start New Experiment")
        new_experiment().create_experiment()
