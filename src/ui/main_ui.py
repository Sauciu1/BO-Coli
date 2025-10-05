from os import write
import streamlit as st
import sys
import pickle
import json
import sys

from src.ui.InitExperiment import InitExperiment
from src.BayesClientManager import BayesClientManager
from src.ui.GroupUi import GroupUi
from src.ui.BayesPlotter import BayesPlotter


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
                    raise ValueError(
                        "The loaded object is not a BayesClientManager instance."
                    )

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

        if st.session_state.get(
            "experiment_configured", False
        ) and st.session_state.get("bayes_manager"):
            st.success("✅ Experiment configuration created!")
            if st.button(
                "Finish Setup and Start Experiment",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.experiment_created = True
                st.session_state.initializing_experiment = False
                st.balloons()  # Show celebration immediately
                st.rerun()


class main_manager:
    def __init__(self):
        self.loader = ExperimentInitialiser()

    def main_loop(self):


            
        st.set_page_config(
            page_title="BoColi Bayesian Optimization for Biological Systems",
            layout="wide",
            initial_sidebar_state="collapsed",
        )
        if not st.session_state.get("experiment_created", False):
            self.loader._init_or_load_exp()
            #self.bayes_manager = st.session_state.get("bayes_manager", None)
         
        else:
            self.run_group_manager()
            BayesPlotter(self.bayes_manager).main_loop()

            st.divider()

        self.write_footnote()

    @property
    def group_manager(self):
        if "group_manager" not in st.session_state and st.session_state.get(
            "bayes_manager"
        ):
            st.session_state.group_manager = GroupUi(st.session_state.bayes_manager)
        return st.session_state.get("group_manager")
    
    @property
    def bayes_manager(self):
        return st.session_state.get("bayes_manager", None)

    def run_group_manager(self):
        self.bayes_manager.sync_self = self.group_manager.sync_all_groups_to_manager


        self.group_manager.render_all()
        # self.group_manager.show_data_stats()

        st.divider()
        st.subheader("Visualization & Analysis")
        
        self.group_manager.sync_all_groups_to_manager()
        

  
    def _download_experiment(self):
        try:
            # Function to prepare download data with sync
            def prepare_download_data():
                # Sync all groups to manager when download is requested
                self.group_manager.sync_all_groups_to_manager()
                return pickle.dumps(self.group_manager.bayes_manager)

            import time

            current_time = time.strftime("%Y%m%d_%H%M")

            # Direct download button with on-demand data preparation
            st.download_button(
                label="💾 Download Experiment Data",
                data=prepare_download_data(),
                file_name=f"bo_coli_{self.bayes_manager.experiment_name}_{current_time}.pkl",
                mime="application/octet-stream",
                help="Download the complete experiment as a pickle file",
            )

            def prepare_csv_data():
                self.group_manager.sync_all_groups_to_manager()
                df = self.group_manager.bayes_manager.data
                return df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="𝄜 Download as .csv (CANNOT BE REUPLOADED)",
                data=prepare_csv_data(),
                file_name=f"bo_coli_{self.bayes_manager.experiment_name}_{current_time}.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"❌ Error preparing download: {e}")


    def write_footnote(self):
        with st.expander("ℹ️ About this App", expanded=False):
            st.markdown(
                """
                ### About This App
                This application enables no-code Bayesian Optimization experiments.
                It allows the user to find the optimal parameters for a black-box response function.

                The HeteroWhiteNoise evaluates heteroscedastic noise from technical repeats and combines it with a small white noise to ensure numerical stability.
                It was specifically designed to handle noise and technical replicates and thus should serve biological data extremely well.

                
                ### Quick Start Guide
                * **Initialize**: Start a new experiment or load an existing one using the sidebar options.
                    * New Experiment: Configure parameters, input initial data, and set up the optimization process.
                * **Manage Groups**: Create and manage experimental groups, each with its own desirable number of replicates and data.
                * **Run Optimization**: Use the built-in Bayesian Optimization tools to suggest new experimental conditions based on your data.
                * **Visualize Results**: Utilize the integrated visualization tools to analyze and interpret your experimental results.
                * **Download Data**: Save your experiment state and results, the file can be reuploaded during the next session.

                **NOTE** - Ensure to save your experiment data before closing the app to avoid data loss. There is no auto-save feature.
                    
                
                ### Life Cycle
                The application was developed by Povilas from Imperial's iGEM 2025, to support the wet lab experimentation.
                It will be maintained and updated, however, is static as part of the iGEM competition up until wiki thaw 9th November 2025.
                Link to active GitHub repository will be provided here after the competition.
                * Main changes will include hosting and data storage solutions. Both of which could not be implemented due to iGEM rules.



                For any questions or support please contact:
                * p.sauciuvienas@gmail.com
                * or https://github.com/Sauciu1
                """
            )


if __name__ == "__main__":
    if '--server.port' not in sys.argv:
        sys.argv.extend(['--server.port', '8989'])


    if "manager" not in st.session_state:
        st.session_state.manager = main_manager()

    st.session_state.manager.main_loop()



