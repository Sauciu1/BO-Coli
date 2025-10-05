import json

from src.GPVisualiser import GPVisualiserPlotly
from src.model_generation import HeteroWhiteSGP
from src.BayesClientManager import BayesClientManager
import pandas as pd
import streamlit as st
from botorch.models import SingleTaskGP


class UiBayesPlotter:
    def __init__(self, bayes_manager: BayesClientManager):
        self.bayes_manager = bayes_manager

    @st.fragment
    def choose_plot_coordinates(self):
        """Allows user to manually set coordinates for plotting the GP"""
        with st.expander("📊 Plot GP at Specific Coordinates", expanded=False):
            columns = st.columns([0.3, 0.3, 0.5])

            def set_look_coords():
                defaults = self.bayes_manager.get_best_coordinates()
                best_group = self.bayes_manager.get_best_group()
                # Store the coordinates to be used as slider defaults
                st.session_state['reset_to_best'] = True
                st.session_state['target_coords'] = defaults
                # Update the group selector to the best performing group
                if best_group is not None:
                    st.session_state['group_selector'] = str(best_group)

            def set_group_coords(group_name):
                """Set coordinates to a specific group's values"""
                group_data = self.bayes_manager.data[
                    self.bayes_manager.data[self.bayes_manager.group_label] == group_name
                ]
                if not group_data.empty:
                    # Get the first row of the group (or you could use mean/median)
                    group_coords = group_data[self.bayes_manager.feature_labels].iloc[0].to_dict()
                    # Store the coordinates to be used as slider defaults
                    st.session_state['reset_to_group'] = True
                    st.session_state['target_coords'] = group_coords

            with columns[0]:
                if st.button("Set to Best Performer", key="set_best_performer"):
                    set_look_coords()
                    st.rerun()
                plot_button = st.button("Plot Gaussian Process", key="plot_the_coords", type="primary")

            with columns[1]:

                if self.bayes_manager.has_response and self.bayes_manager.group_label:
                    available_groups = sorted(self.bayes_manager.data[self.bayes_manager.group_label].unique())
                    
                    selected_group = st.selectbox(
                        "Select Observation Group:",
                        options=[""] + list(available_groups),
                        key="group_selector",
                        help="Choose a group to set coordinates to that group's values"
                    )
                    
                    if selected_group and selected_group != "":
                        if st.button("Set to Group", key="set_to_group"):
                            set_group_coords(selected_group)
                            st.rerun()

            with columns[2]:
                self.look_coords_slider()

            # Execute plotting inside the expander
            if plot_button and self.bayes_manager.has_response:
                if self.plot_coords is None or len(self.plot_coords) != len(self.bayes_manager.feature_labels):
                    st.error(
                        f"Please provide exactly {len(self.bayes_manager.feature_labels)} coordinates."
                    )
                else:
                    self.plot_gaussian_process(gp_model=SingleTaskGP, coords=self.plot_coords)
            elif plot_button:
                st.warning("No data available to plot the GP.")

    def look_coords_slider(self):
        """Create sliders for each parameter to set look coordinates"""
        if self.bayes_manager is None or not self.bayes_manager.feature_labels:
            st.error("Bayesian manager or feature labels not provided.")
            return

        # Check if we need to reset coordinates
        target_coords = None
        if st.session_state.get('reset_to_best', False) or st.session_state.get('reset_to_group', False):
            target_coords = st.session_state.get('target_coords', {})
            # Clear the reset flags
            st.session_state['reset_to_best'] = False
            st.session_state['reset_to_group'] = False

        # Get default values
        defaults = self.bayes_manager.get_best_coordinates()
        current_coords = {}
        
        for param in self.bayes_manager.feature_labels:
            bounds = self.bounds[param]
            
            # Use target coordinates if resetting, otherwise use defaults
            if target_coords and param in target_coords:
                default_value = target_coords[param]
            else:
                default_value = defaults.get(param, bounds["lower_bound"])
            
            slider_value = st.slider(
                label=f"Set {param}",
                min_value=bounds["lower_bound"],
                max_value=bounds["upper_bound"],
                value=default_value,
                step=(bounds["upper_bound"] - bounds["lower_bound"]) / 100.0,
                format="%.3e",
                key=f"slider_{param}"
            )
            
            current_coords[param] = slider_value

        # Update bayes_manager with current slider values
        self.bayes_manager.look_coords = pd.DataFrame.from_dict(
            {k: [v] for k, v in current_coords.items()}
        ).T.rename(columns={0: "value"})

    @property
    def bounds(self) -> dict:
        if self.bayes_manager.bounds:
            return self.bayes_manager.bounds
        else:
            raise ValueError("Bounds are not defined in the BayesClientManager.")

    @property
    def plot_coords(self):
        # Get current slider values directly from session state
        coords = []
        for param in self.bayes_manager.feature_labels:
            slider_key = f"slider_{param}"
            if slider_key in st.session_state:
                coords.append(st.session_state[slider_key])
            else:
                # Fallback to bayes_manager if slider not initialized
                coords.append(self.bayes_manager.look_coords["value"][param])
        return coords


    def plot_gaussian_process(self, gp_model=SingleTaskGP, coords=None):
        """Plot the Gaussian Process using GPVisualiserPlotly"""
        if self.bayes_manager is None or gp_model is None:
            st.error("Bayesian manager or GP model not provided.")
            return
        elif not self.bayes_manager.has_response:
            st.warning("No observations available to plot the GP.")
            return

        plotter = GPVisualiserPlotly(
            self.bayes_manager
        )

        fig, ax = plotter.plot_all(coordinates=coords)
        st.plotly_chart(
            fig, 
            use_container_width=True, 
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
            }
        )

    @st.fragment
    def plot_group_performance(self):
        """Performance of each observation group as box plot"""
        import plotly.express as px
        
        with st.expander("📈 Plot Observation Performance", expanded=False):
            if st.button("Generate Performance Plot", key="plot_obs_performance"):
                if not self.bayes_manager.has_response:
                    st.warning("No data available to plot.")
                    return None
                
                # Render plot inside the expander
                fig = px.box(
                    data_frame=self.bayes_manager.data,
                    x=self.bayes_manager.group_label,
                    y=self.bayes_manager.response_label,
                    points="all",
                    title="Group Performance Distribution",
                )
                fig.update_traces(boxmean="sd")
                fig.update_layout(
                    xaxis_title="Group",
                    yaxis_title=self.bayes_manager.response_label,
                    legend_title="Group",
                    margin=dict(l=10, r=10, t=60, b=40),
                )
                st.plotly_chart(
                    fig, 
                    use_container_width=True, 
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                    }
                )
                return fig
        return None

    def main_loop(self):
        self.plot_group_performance()
        self.choose_plot_coordinates()


if __name__ == "__main__":

    def load_ackley():
        auckley_path = r"data/ax_clients/ackley_client.pkl"
        import pickle

        client = pickle.load(open(auckley_path, "rb"))
        return BayesClientManager.init_from_client(client)

    def load_manual():
        """Load a manual test dataset using BayesClientManager"""
        test_df = pd.DataFrame(
            {
                "x1": [0.1, 0.1, 0.2],
                "x2": [0.2, 0.3, 0.2],
                "response": [1.0, 1.5, 2.0],
            }
        )

        bounds = {
            "x1": {"lower_bound": 0.0, "upper_bound": 1.0, "log_scale": False},
            "x2": {"lower_bound": 0.0, "upper_bound": 1.0, "log_scale": False},
        }

        return BayesClientManager(
            data=test_df,
            feature_labels=["x1", "x2"],
            response_label="response",
            bounds=bounds,
        )

    st.set_page_config(layout="wide")
    bayes_manager = load_ackley()
    plotter = UiBayesPlotter(bayes_manager)
    plotter.main_loop()
