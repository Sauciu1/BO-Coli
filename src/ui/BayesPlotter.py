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
        st.subheader("Plot GP at Specific Coordinates")
        columns = st.columns([0.6, 0.4])

        def set_look_coords():
            defaults = self.bayes_manager.get_best_coordinates()
            df = pd.DataFrame(
                [defaults],
            ).transpose()
            df.columns = ["value"]
            self.bayes_manager.look_coords = df

        if "manual_coords_df" not in st.session_state:
            set_look_coords()

        with columns[0]:
            if st.button("Set to Best Performer", key="set_best_performer"):
                set_look_coords()
            plot_button = st.button("Plot Gaussian Process", key="plot_the_coords")

        with columns[1]:
            self.bayes_manager.look_coords = st.data_editor(
                self.bayes_manager.look_coords,
                num_rows="fixed",
                use_container_width=True,
                column_config={
                    "value": st.column_config.NumberColumn(
                        format="%.3e",
                        min_value=min(
                            self.bounds[c]["lower"]
                            for c in self.bayes_manager.feature_labels
                        ),
                        max_value=max(
                            self.bounds[c]["upper"]
                            for c in self.bayes_manager.feature_labels
                        ),
                        help="Parameter value (will be clamped to valid range)",
                    )
                },
                key="manual_coords_editor",
            )

        # Clamp values to parameter ranges
        for param in self.bayes_manager.feature_labels:
            if param in self.bayes_manager.look_coords.index:
                min_val, max_val = (
                    self.bounds[param]["lower"],
                    self.bounds[param]["upper"],
                )
                current_val = self.bayes_manager.look_coords.loc[param, "value"]
                clamped_val = max(min_val, min(max_val, current_val))
                self.bayes_manager.look_coords.loc[param, "value"] = clamped_val

        # Move the plotting logic outside the columns
        if plot_button and self.bayes_manager.has_response:
            if self.plot_coords is None or len(self.plot_coords) != len(
                self.bayes_manager.feature_labels
            ):
                st.error(
                    f"Please provide exactly {len(self.bayes_manager.feature_labels)} coordinates."
                )
            self.plot_gaussian_process(gp_model=SingleTaskGP, coords=self.plot_coords)
        else:
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
        st.plotly_chart(fig, use_container_width=True)

    def plot_group_performance(self):
        import plotly.express as px

        """Performance of each observation group as box plot"""
        df = self.bayes_manager.data

        fig = px.box(
            data_frame=df,
            x=self.bayes_manager.group_label,
            y=self.bayes_manager.response_label,
            points="all",
            #  category_orders={self.bayes_manager.group_label: order},
            # hover_data=[c for c in self.bayes_manager.input_cols if c in self.bayes_manager.df.columns],
            title="Group Performance Distribution",
        )
        fig.update_traces(boxmean="sd")
        fig.update_layout(
            xaxis_title="Group",
            yaxis_title=self.bayes_manager.response_label,
            legend_title="Group",
            margin=dict(l=10, r=10, t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
        return fig

    def main_loop(self):
        if st.button("Plot Observation Performance", key="plot_obs_performance"):
            if not self.bayes_manager.has_response:
                st.warning("No data available to plot.")
            else:
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
            "x1": {"lower": 0.0, "upper": 1.0, "log": False},
            "x2": {"lower": 0.0, "upper": 1.0, "log": False},
        }

        return BayesClientManager(
            data=test_df,
            feature_labels=["x1", "x2"],
            response_label="response",
            bounds=bounds,
        )

    bayes_manager = load_ackley()
    plotter = UiBayesPlotter(bayes_manager)
    plotter.main_loop()
