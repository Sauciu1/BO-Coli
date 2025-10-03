from ax_helper import BayesClientManager
import pandas as pd
import streamlit as st
import ax


class new_experiment:
    def __init__(self):
        client = ax.Client()
        self.experiment_parameters = {}

    @st.fragment
    def all_parameter_handler(self):
        """Handles all parameter related UI elements"""
        pass

    def get_parameters(self):
        """Returns a list of parameters as dicts"""
        pass

    @st.fragment
    def choose_parameters(self):
        """Allows to manually name a parameter/dimension,
        set its range and set whether the axis is logarithmic"""
        pass

if __name__ == "__main__":
    if "manager" not in st.session_state:
        st.set_page_config(
            page_title="Start New Experiment",
            page_icon="🧪",
            layout="wide",
            initial_sidebar_state="collapsed",
        )
        st.title("🧪 Start New Experiment")
        new_experiment().choose_parameters()