import streamlit as st

from ui_start_new_experiment import new_experiment
from ax_helper import BayesClientManager
from ui_group_manager import GroupManager
import pickle
import json
from ui_plotter import BayesPlotter


class main_manager():
    def __init__(self):
        pass

    @property
    def bayes_manager(self):
        if "bayes_manager" in st.session_state:
            return st.session_state.bayes_manager
        return None


    def main_loop(self):
        if not st.session_state.get("experiment_created", False):
            self._init_or_load_exp()
        else:
           self.run_group_manager()


    def _init_or_load_exp(self):
        st.title("Bayesian Optimization Experiment Manager")
        st.write("Initialize a new experiment or load an existing one.")

        if "bayes_manager" not in st.session_state:
            st.session_state.bayes_manager = None

 
        if st.button("Initialize New Experiment"):
            self._init_experiment()

        if st.button("Load Existing Experiment"):
            self._load_experiment()

        if st.button("Finish Setup and Start Experiment", type="primary"):
            if "new_exp" not in st.session_state:
                exp = st.session_state.new_exp
                if all(getattr(exp, attr) is not None for attr in ['client', 'gp', 'acquisition_function']):
                    manager = BayesClientManager(exp.client, exp.gp, exp.acquisition_function)
                    st.session_state.bayes_manager = manager
                    st.session_state.experiment_created = True
            elif self.bayes_manager is not None:
                    st.session_state.experiment_created = True


###########
            else:
                st.error("Please complete the experiment configuration before proceeding.")
            
            if "client" in st.session_state and st.session_state.client is not None:
                st.success("✅ Experiment configuration created!")
                st.rerun()

        else:
            st.info("👆 Please create your experiment configuration first by clicking 'Create Experiment' above.")


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


    @property
    def group_manager(self)->GroupManager:
        return st.session_state.get("group_manager", None)
        
    def run_group_manager(self):
        st.session_state.group_manager = GroupManager.init_from_manager(self.bayes_manager)
        self.group_manager.render_all()
        self.group_manager.show_data_stats()

        plotter = BayesPlotter(self.bayes_manager)
        plotter.plot_group_performance()
        plotter.choose_plot_coordinates()
       

if __name__ == "__main__":
    if "manager" not in st.session_state:
        st.session_state.manager = main_manager()

    st.title("Bayesian Optimization for Biological Systems")
    manager = st.session_state.manager
    manager.main_loop()
    

    



            