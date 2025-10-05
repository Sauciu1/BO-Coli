import streamlit as st
import sys
import pickle
import json

from src.ui.InitExperiment import InitExperiment
from src.BayesClientManager import BayesClientManager
from src.ui.GroupUi import GroupUi
from src.ui.BayesPlotter import UiBayesPlotter





class ExperimentInitialiser:
    def _init_or_load_exp(self):
        if st.session_state.get("initializing_experiment", False):
            return self._init_experiment()
            
        st.title("Bayesian Optimization Experiment Manager")
        st.write("Initialize a new experiment or load an existing one.")

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Initialize New Experiment", use_container_width=True):
                st.session_state.initializing_experiment = True
                st.rerun()
                
        with col2:
            self._load_exp_from_pickle_ui()
            # File upload directly visible
            

    def _load_exp_from_pickle_ui(self):
        uploaded_file = st.file_uploader("Upload saved experiment file", type=["pkl"])
        if uploaded_file is not None:
            try:

                uploaded_file.seek(0)

                manager = BayesClientManager.init_self_from_pickle(uploaded_file)
                
                if not isinstance(manager, BayesClientManager):
                    raise ValueError("The loaded object is not a BayesClientManager instance.")
                
                # Store in session state
                st.session_state.bayes_manager = manager
                st.session_state.experiment_created = True
                st.success("✅ Experiment loaded successfully!")
                st.rerun()

            except (pickle.UnpicklingError, AttributeError, ModuleNotFoundError) as e:
                st.error(f"❌ Pickle file error: {e}")
            except Exception as e:
                st.error(f"❌ Failed to load file: {e}")
        


    def _init_experiment(self):        
        if "new_exp" not in st.session_state:
            st.session_state.new_exp = InitExperiment()
        
        st.session_state.new_exp.create_experiment()
        
        if st.session_state.get("experiment_configured", False) and st.session_state.get("bayes_manager"):
            st.success("✅ Experiment configuration created!")
            if st.button("Finish Setup and Start Experiment", type="primary", use_container_width=True):
                st.session_state.experiment_created = True
                st.session_state.initializing_experiment = False
                st.balloons()  # Show celebration immediately
                st.rerun()



class main_manager:
    def __init__(self):
        self.loader = ExperimentInitialiser()

    

    def main_loop(self):
        if not st.session_state.get("experiment_created", False):
            self.loader._init_or_load_exp()
        else:
            self.run_group_manager()

    @property
    def group_manager(self):
        if "group_manager" not in st.session_state and st.session_state.get("bayes_manager"):
            st.session_state.group_manager = GroupUi(st.session_state.bayes_manager)
        return st.session_state.get("group_manager")

        
    def run_group_manager(self):
        bayes_manager = st.session_state.get("bayes_manager")
        if not bayes_manager:
            st.error("No experiment data loaded.")
            return
        
        self.group_manager.render_all()
       # self.group_manager.show_data_stats()

        st.divider()
        st.subheader("Visualization & Analysis")
        UiBayesPlotter(bayes_manager, self.group_manager).main_loop()

        st.divider()

        try:
            # Function to prepare download data with sync
            def prepare_download_data():
                # Sync all groups to manager when download is requested
                self.group_manager.sync_all_groups_to_manager()
                return pickle.dumps(self.group_manager.bayes_manager)
            
            # Direct download button with on-demand data preparation
            st.download_button(
                label="💾 Download Experiment Data",
                data=prepare_download_data(),
                file_name="experiment_data.pkl",
                mime="application/octet-stream",
                help="Download the complete experiment as a pickle file"
            )
        except Exception as e:
            st.error(f"❌ Error preparing download: {e}")
        
        

    

if __name__ == "__main__":
    
    st.set_page_config(
        page_title="Bayesian Optimization for Biological Systems",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    if "manager" not in st.session_state:
        st.session_state.manager = main_manager()
    
    st.session_state.manager.main_loop()