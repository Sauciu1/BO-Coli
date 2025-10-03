from email.headerregistry import Group
import streamlit as st

from src.ui_start_new_experiment import new_experiment
from src.ax_helper import BayesClientManager
from src.ui_group_manager import GroupManager
import pickle
import json
from src.ui_plotter import BayesPlotter


class ExperimentInitialiser():
    def __init__(self):
        pass
    def _init_or_load_exp(self):
        # Check if we're in the middle of creating a new experiment
        if st.session_state.get("initializing_experiment", False):
            self._init_experiment()
            return
            
        # Check if we're in the middle of loading an experiment
        if st.session_state.get("loading_experiment", False):
            self._load_experiment()
            return
            
        # Show the main menu if we're not in either flow
        st.title("Bayesian Optimization Experiment Manager")
        st.write("Initialize a new experiment or load an existing one.")

        if "bayes_manager" not in st.session_state:
            st.session_state.bayes_manager = None

 
        if st.button("Initialize New Experiment"):
            st.session_state.initializing_experiment = True
            st.rerun()

        if st.button("Load Existing Experiment"):
            st.session_state.loading_experiment = True
            st.rerun()


    @st.fragment
    def _load_experiment(self):
        """Loads from file. NEEDS REFACTORING"""
        uploaded_file = st.file_uploader("Upload saved experiment file", type=["pkl", "json"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".pkl"):
                    loaded_obj = pickle.load(uploaded_file)
                    # Check if loaded object is already a BayesClientManager
                    if isinstance(loaded_obj, BayesClientManager):
                        st.session_state.bayes_manager = loaded_obj
                    else:
                        st.error("Invalid pickle file format. Expected BayesClientManager object.")
                        return
                else:
                    # For JSON files, save temporarily and pass path to init_from_json
                    import tempfile
                    import os
                    
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue().decode('utf-8'))
                        tmp_path = tmp_file.name
                    
                    try:
                        # Use init_from_json with the file path
                        st.session_state.bayes_manager = BayesClientManager.init_from_json(tmp_path)
                    finally:
                        # Clean up the temporary file
                        os.unlink(tmp_path)
                
                # Set the experiment as created/completed
                st.session_state.experiment_created = True
                # Clear the loading flag
                st.session_state.loading_experiment = False
                st.success("✅ Experiment file loaded successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Failed to load file: {e}")
        else:
            st.info("Awaiting file upload...")

    def _init_experiment(self):
    # Check if experiment was already successfully created
        if "experiment_created" in st.session_state and st.session_state.experiment_created:
            # Show only SUCCESS button
            st.title("🎉 EXPERIMENT CREATED SUCCESSFULLY!")
            if st.button("SUCCESS", type="primary", use_container_width=True):
                st.balloons()
                st.rerun()
            return
        
        if "new_exp" not in st.session_state:
            st.session_state.new_exp = new_experiment()
        
        exp = st.session_state.new_exp
        exp.create_experiment()
        
        # Check if experiment configuration is complete (client exists)
        if st.session_state.get("client") is not None:
            st.divider()
            st.success("✅ Experiment configuration created!")
            if st.button("Finish Setup and Start Experiment", type="primary", use_container_width=True):
                manager = BayesClientManager(st.session_state.client, exp.gp, exp.acquisition_function)
                st.session_state.bayes_manager = manager
                st.session_state.experiment_created = True
                # Clear the initializing flag
                st.session_state.initializing_experiment = False
                st.rerun()



class main_manager():
    def __init__(self):
        self.loader = ExperimentInitialiser()

    @property
    def bayes_manager(self):
        return st.session_state.bayes_manager
  


    def main_loop(self):
        if not st.session_state.get("experiment_created", False):
            self.loader._init_or_load_exp()
        else:
           self.run_group_manager()


    @property
    def group_manager(self)->GroupManager:
        return st.session_state.get("group_manager", GroupManager.init_from_manager(self.bayes_manager))

        
    def run_group_manager(self):
        # Add a reset button at the top
        if st.button("🔄 New Experiment", help="Start a new experiment"):
            # Clear all experiment-related session state
            for key in ['bayes_manager', 'experiment_created', 'initializing_experiment', 
                       'loading_experiment', 'client', 'new_exp', 'experiment_configured', 
                       'group_manager']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        st.session_state.group_manager = GroupManager.init_from_manager(self.bayes_manager)
        self.group_manager.render_all()
        self.group_manager.show_data_stats()

        plotter = BayesPlotter(self.bayes_manager)
        plotter.main_loop()
    
       

if __name__ == "__main__":
    if "manager" not in st.session_state:
        st.session_state.manager = main_manager()

    st.title("Bayesian Optimization for Biological Systems")
    manager = st.session_state.manager
    manager.main_loop()
    

    



            