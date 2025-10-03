from ax.core import parameter
from ax_helper import BayesClientManager
import pandas as pd
import streamlit as st
import ax
from ax import Client, RangeParameterConfig

class new_experiment:
    def __init__(self):
        self.client = ax.Client()
        self.experiment_parameters = {}
        self.experiment_name = None


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
                st.session_state.parameters = [p for p in st.session_state.parameters if p.get("id") != param_id]
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
                    param["name"] = st.text_input(f"Parameter {i+1} Name", param["name"], key=f"name_{param_id}")
                with col2:
                    param["lower"] = st.number_input("Lower Bound", value=param["lower"], key=f"lower_{param_id}")
                with col3:
                    param["upper"] = st.number_input("Upper Bound", value=param["upper"], key=f"upper_{param_id}")
                with col4:
                    param["log_scale"] = st.checkbox("Log Scale", param["log_scale"], key=f"log_{param_id}")
                with col5:
                    if st.button("🗑️", key=f"delete_{param_id}", help="Delete parameter"):
                        if "params_to_delete" not in st.session_state:
                            st.session_state.params_to_delete = []
                        st.session_state.params_to_delete.append(param_id)
                        st.rerun()
        
        # Add new parameter button
        if st.button("➕ Add Parameter"):
            new_param = {
                "id": st.session_state.param_counter,
                "name": f"param_{len(st.session_state.parameters)+1}",
                "lower": 0.0,
                "upper": 1.0,
                "log_scale": False
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
        
        return ax_parameters

    @st.fragment
    def create_experiment(self):
        """Allows to manually name a parameter/dimension,
        set its range and whether the axis is logarithmic"""
        st.markdown("### Define Experiment Parameters")
        st.write("Configure the parameters for your Bayesian optimization experiment.")

        st.text_input("Experiment Name", "My Experiment", key="experiment_name")
        self.experiment_name = st.session_state.experiment_name

        # Objective direction switch
        if "objective_direction" not in st.session_state:
            st.session_state.objective_direction = "Maximize"

        st.session_state.objective_direction = st.radio(
            "Objective Direction",
            ["Maximize", "Minimize"],
            index=0,
            horizontal=True,
        )

       # self.objective_direction = st.session_state.objective_direction
        st.caption(f"Current setting: {self.objective_direction} the objective.")
        
        # Call the parameter handler
        self.parameter_handler()
        
        # Display parameter summary
        params = self.get_parameters()

        st.subheader("Parameter Summary")
        param_df = pd.DataFrame(params)
        
        
        # Create experiment button
        if st.button("🚀 Create Experiment", type="primary"):
            if len(params) > 0:
                st.success(f"Creating experiment: **{self.experiment_name}** to **{self.objective_direction.lower()}**"+
                           f" the objective with **{len(params)}** parameters!")
                self.experiment_parameters = params
    
                st.dataframe(param_df, use_container_width=True)
 
            else:
                st.error("Please add at least one parameter to create an experiment.")

            if st.button("Finish Setup"):
                self.write_params_to_client()
                st.success("Parameters sent to client!")
                return "exit"

    def write_params_to_client(self):
            
        range_parameters = [
            RangeParameterConfig(
                name=name,
                parameter_type="float",
                bounds=(lower, upper),
                scaling="log" if log else "linear"
            ) for name, lower, upper, log in self.get_parameters()

        ]
    
        self.client.configure_experiment(
            name="batch_bo_test",
            parameters=range_parameters,
        )

        self.client.configure_optimization(objective="-loss" if self.objective_direction == "Maximize" else "loss")

        return self.client


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