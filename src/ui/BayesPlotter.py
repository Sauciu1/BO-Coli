import json

from src.GPVisualiser import GPVisualiserPlotly
from src.model_generation import HeteroWhiteSGP
from src.BayesClientManager import BayesClientManager
import pandas as pd
import streamlit as st
from botorch.models import SingleTaskGP
import plotly.express as px


class BayesPlotter:
    def __init__(self, bayes_manager: BayesClientManager):
        self.bayes_manager = bayes_manager

    def check_plot_ready(passed_function):
        """Decorator to ensure the bayes_manager is synced and has response data before plotting"""

        def _inner_function(self, *args, **kwargs):
            self.bayes_manager.sync_self()
            if self.bayes_manager.has_response:
                return passed_function(self, *args, **kwargs)
            else:
                st.warning("No data available for plotting.")
                return None

        return _inner_function

    @check_plot_ready
    def _set_to_best_performer(self):
        if st.button("Set to Best Performer", key="set_best_performer"):
            # Use BayesClientManager to get best coordinates and group
            best_coords = self.bayes_manager.get_best_coordinates()
            best_group = self.bayes_manager.get_best_group()

            if best_coords is not None and best_group is not None:
                # Cache the results for performance
                if "best_coords_cache" not in st.session_state:
                    st.session_state["best_coords_cache"] = {}

                cache_key = f"{len(self.bayes_manager.data)}_{hash(str(self.bayes_manager.data[self.bayes_manager.response_label].values.tobytes()))}"
                st.session_state["best_coords_cache"][cache_key] = {
                    "coords": best_coords,
                    "group": best_group,
                }

                if "group_selector" not in st.session_state:
                    st.session_state["group_selector"] = str(best_group)
                self._set_group_coords(best_group)
                st.session_state["pending_rerun"] = True
                st.rerun(scope="fragment")

    @check_plot_ready
    def _set_group_coords(self, group_name):
        """Set coordinates to a specific group's values using BayesClientManager."""
        groups_dict = self.bayes_manager.get_groups()
        if group_name in groups_dict:
            group_data = groups_dict[group_name]
            if not group_data.empty:
                # Get the first row of the group for coordinates
                group_coords = (
                    group_data[self.bayes_manager.feature_labels].iloc[0].to_dict()
                )
                # Store the coordinates to be used as slider defaults
                st.session_state["reset_to_group"] = True
                st.session_state["target_coords"] = group_coords

    @st.fragment
    def choose_plot_coordinates(self):
        """Allows user to manually set coordinates for plotting the GP"""

        columns = st.columns([0.3, 0.3, 0.5])

        with columns[1]:

            self._choose_group_for_coords()

        with columns[2]:
            self.look_coords_slider()

        with columns[0]:
            self._set_to_best_performer()
            button = st.button("Plot Gaussian Process", key="plot_the_coords", type="primary")

        if button:
            self.plot_gaussian_process(gp_model=SingleTaskGP, coords=self.plot_coords)

    @check_plot_ready
    def _choose_group_for_coords(self):
        if self.bayes_manager.has_response and self.bayes_manager.group_label:
            # Use BayesClientManager to get available groups
            available_groups = self.bayes_manager.current_group_labels
            options = [""] + [str(g) for g in available_groups]
            default_index = 0

            def group_select_callback():
                # Handle group selection when user manually changes the dropdown
                selected_group = st.session_state.get("group_selector", "")
                if selected_group and selected_group != "":
                    self._set_group_coords(int(selected_group))
                    st.session_state["pending_rerun"] = True

            selected_group = st.selectbox(
                "Select Observation Group:",
                options=options,
                index=default_index,
                key="group_selector",
                help="Choose a group to set coordinates to that group's values",
                on_change=group_select_callback,
            )

            # Display group statistics when a group is selected
            if selected_group and selected_group != "":
                groups_dict = self.bayes_manager.get_groups()
                group_id = int(selected_group)
                if group_id in groups_dict:
                    group_data = groups_dict[group_id]
                    if not group_data.empty:
                        mean_response = group_data[
                            self.bayes_manager.response_label
                        ].mean()
                        if pd.notna(mean_response):
                            st.caption(
                                f"Group {selected_group} mean response: **{mean_response:.4f}**"
                            )
                        else:
                            st.caption(
                                f"Group {selected_group}: No response data available"
                            )

    @check_plot_ready
    def look_coords_slider(self):
        """Create sliders for each parameter to set look coordinates"""
        if self.bayes_manager is None or not self.bayes_manager.feature_labels:
            st.error("Bayesian manager or feature labels not provided.")
            return

        # Handle pending rerun
        if st.session_state.get("pending_rerun", False):
            st.session_state["pending_rerun"] = False
            st.rerun()

        # Check if we need to reset coordinates
        target_coords = None
        if st.session_state.get("reset_to_best", False) or st.session_state.get(
            "reset_to_group", False
        ):
            target_coords = st.session_state.get("target_coords", {})
            # Clear the reset flags
            st.session_state["reset_to_best"] = False
            st.session_state["reset_to_group"] = False

        # Cache default values to avoid expensive recomputation
        if "default_coords_cache" not in st.session_state:
            best_coords = self.bayes_manager.get_best_coordinates()
            st.session_state["default_coords_cache"] = (
                best_coords if best_coords is not None else {}
            )

        defaults = st.session_state["default_coords_cache"]
        current_coords = {}

        for param in self.bayes_manager.feature_labels:
            bounds = self.bounds[param]

            if target_coords and param in target_coords:
                default_value = target_coords[param]
                st.session_state["default_coords_cache"][param] = default_value
            elif defaults and param in defaults:
                default_value = defaults[param]
            else:
                default_value = (bounds["lower_bound"] + bounds["upper_bound"]) / 2

            slider_value = st.slider(
                label=f"Set {param}",
                min_value=bounds["lower_bound"],
                max_value=bounds["upper_bound"],
                value=default_value,
                step=(bounds["upper_bound"] - bounds["lower_bound"]) / 100.0,
                format="%.3e",
                key=f"slider_{param}",
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
        """Get current plotting coordinates from sliders or bayes_manager."""
        coords = []
        for param in self.bayes_manager.feature_labels:
            slider_key = f"slider_{param}"
            if slider_key in st.session_state:
                coords.append(st.session_state[slider_key])
            else:
                # Fallback to bayes_manager default coordinates
                default_coords = self.bayes_manager.get_best_coordinates()
                if default_coords and param in default_coords:
                    coords.append(default_coords[param])
                else:
                    # Ultimate fallback to bounds midpoint
                    bounds = self.bounds[param]
                    coords.append((bounds["lower_bound"] + bounds["upper_bound"]) / 2)
        return coords

    @check_plot_ready
    def plot_gaussian_process(self, gp_model=SingleTaskGP, coords=None):
        """Plot the Gaussian Process using GPVisualiserPlotly"""
        if not self.bayes_manager.has_response:
            st.warning("No observations available to plot the GP.")
            return
        if self.bayes_manager is None or gp_model is None:
            st.error("Bayesian manager or GP model not provided.")
            return
        elif not self.bayes_manager.has_response:
            st.warning("No observations available to plot the GP.")
            return

        plotter = GPVisualiserPlotly(self.bayes_manager)

        fig, ax = plotter.plot_all(coordinates=coords)
        fig.update_layout(title=None)
        st.write(f"### Gaussian Process Visualization at ({', '.join(map(str, coords))})")
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
            },
        )

    @st.fragment
    @check_plot_ready
    def plot_group_performance(self):
        """Performance of each observation group as box plot"""

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
            yaxis_title="Response",
            legend_title="Group",
            margin=dict(l=10, r=10, t=60, b=40),
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
            },
        )
        return fig

    @st.fragment
    def main_loop(self):
        if st.button("Reload Data for Plotting", key="reload_data_plot"):
            self.bayes_manager.sync_self()
            st.rerun(scope="fragment")
        with st.expander("📈 Plot Group Performance", expanded=True):
            self.plot_group_performance()
        with st.expander("📊 Plot GP at Specific Coordinates", expanded=True):
            self.choose_plot_coordinates()


if __name__ == "__main__":

    def load_ackley():
        auckley_path = r"data/ax_clients/ackley_client.pkl"
        import pickle

        client = pickle.load(open(auckley_path, "rb"))
        return BayesClientManager.init_from_client(client)

    def load_manual():
        """Load a manual test dataset using BayesClientManager with group labels"""
        test_df = pd.DataFrame(
            {
                "x1": [0.1, 0.1, 0.2, 0.3, 0.4, 0.5],
                "x2": [0.2, 0.3, 0.2, 0.1, 0.5, 0.6],
                "response": [1.0, 1.5, 2.0, 2.5, 1.8, 3.0],
                "group": [1, 1, 2, 2, 3, 3],
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
    bayes_manager = load_manual()
    plotter = BayesPlotter(bayes_manager)
    plotter.main_loop()
