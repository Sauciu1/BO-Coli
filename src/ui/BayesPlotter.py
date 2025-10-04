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
            columns = st.columns([0.6, 0.4])

            def set_look_coords():
                defaults = self.bayes_manager.get_best_coordinates()
                self.bayes_manager.look_coords = pd.DataFrame([defaults]).T.rename(columns={0: "value"})

            if "manual_coords_df" not in st.session_state:
                set_look_coords()

            with columns[0]:
                if st.button("Set to Best Performer", key="set_best_performer"):
                    set_look_coords()
                plot_button = st.button("Plot Gaussian Process", key="plot_the_coords")

            with columns[1]:
                bounds_values = [self.bounds[c] for c in self.bayes_manager.feature_labels]
                self.bayes_manager.look_coords = st.data_editor(
                    self.bayes_manager.look_coords,
                    num_rows="fixed",
                    width="stretch",
                    column_config={
                        "value": st.column_config.NumberColumn(
                            format="%.3e",
                            min_value=min(b["lower_bound"] for b in bounds_values),
                            max_value=max(b["upper_bound"] for b in bounds_values),
                            help="Parameter value (will be clamped to valid range)",
                        )
                    },
                    key="manual_coords_editor",
                )

            # Clamp values to parameter ranges
            for param in self.bayes_manager.feature_labels:
                if param in self.bayes_manager.look_coords.index:
                    bounds = self.bounds[param]
                    current_val = self.bayes_manager.look_coords.loc[param, "value"]
                    self.bayes_manager.look_coords.loc[param, "value"] = max(
                        bounds["lower_bound"], min(bounds["upper_bound"], current_val)
                    )

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

    @property
    def bounds(self) -> dict:
        if self.bayes_manager.bounds:
            return self.bayes_manager.bounds
        else:
            raise ValueError("Bounds are not defined in the BayesClientManager.")

    @property
    def plot_coords(self):
 
        return list(self.bayes_manager.look_coords["value"].values)


    def plot_gaussian_process(self, gp_model=SingleTaskGP, coords=None):
        """Plot the Gaussian Process using GPVisualiserPlotly"""
        if self.bayes_manager is None or gp_model is None:
            st.error("Bayesian manager or GP model not provided.")
            return
        elif not self.bayes_manager.has_response:
            st.warning("No observations available to plot the GP.")
            return

        plotter = GPVisualiserPlotly(
            gp=gp_model,
            obs=self.bayes_manager.data,
            dim_cols=self.bayes_manager.feature_labels,
            response_col=self.bayes_manager.response_label,
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
