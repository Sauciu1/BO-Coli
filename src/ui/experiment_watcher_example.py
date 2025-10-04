"""
Example of how to use the experiment creation detection functions.
This demonstrates how other components can watch for and react to experiment creation.
"""

import streamlit as st
from src.ui.UI_main import ExperimentInitialiser

def on_experiment_created(bayes_manager):
    """
    Callback function that gets called when an experiment is created.
    
    Args:
        bayes_manager: The BayesClientManager instance that was created
    """
    st.success(f"🎉 Experiment detected! Manager type: {type(bayes_manager).__name__}")
    st.info(f"Feature labels: {bayes_manager.feature_labels}")
    st.info(f"Response label: {bayes_manager.response_label}")
    
def example_usage():
    """Example of how to use the experiment detection functions"""
    
    st.title("Experiment Detection Example")
    
    # Method 1: Check if experiment is ready
    if ExperimentInitialiser.is_experiment_ready():
        st.success("✅ Experiment is ready!")
        manager = ExperimentInitialiser.get_created_experiment()
        st.write(f"Manager instance: {manager}")
    else:
        st.info("⏳ No experiment created yet")
    
    # Method 2: Watch for experiment creation with callback
    st.subheader("Watching for Experiment Creation")
    manager = ExperimentInitialiser.watch_experiment_creation(on_experiment_created)
    
    if manager:
        st.write("### Manager Details")
        st.write(f"- Feature labels: {manager.feature_labels}")
        st.write(f"- Response label: {manager.response_label}")
        st.write(f"- Data shape: {manager.data.shape}")
    
    # Method 3: Manual detection check
    if st.button("Check Experiment Status"):
        if st.session_state.get("experiment_configured", False):
            st.success("Experiment has been configured!")
            if st.session_state.get("bayes_manager"):
                st.success("BayesClientManager is available!")
            else:
                st.warning("Experiment configured but manager not found")
        else:
            st.info("No experiment configured yet")

if __name__ == "__main__":
    example_usage()