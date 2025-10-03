import json
from GPVisualiser import GPVisualiserPlotly
from model_generation import HeteroWhiteSGP
from ax_helper import BayesClientManager
import pandas as pd
import streamlit as st









class BayesPlotter:
    def __init__ (self, bayes_manager: BayesClientManager):
        self.bayes_manager = bayes_manager
        self.param_ranges = bayes_manager.get_parameter_ranges()
        self.input_cols = bayes_manager.input_cols


    @st.fragment
    def choose_plot_coordinates(self):
        """Allows user to manually set coordinates for plotting the GP"""
        st.subheader("Plot GP at Specific Coordinates")
        columns = st.columns([0.6,0.4])
    
            
        if "manual_coords_df" not in st.session_state:
            defaults = self.bayes_manager.get_best_coordinates()
            df = pd.DataFrame([defaults], ).transpose()
            df.columns = ['value']
            st.session_state.manual_coords_df = df

        param_ranges = self.bayes_manager.get_parameter_ranges()

        with columns[0]:
            if st.button("Set to Best Performer", key="set_best_performer"):
                defaults = self.bayes_manager.get_best_coordinates()
                df = pd.DataFrame([defaults]).transpose()
                df.columns = ['value']
                st.session_state.manual_coords_df = df
            plot_button = st.button("Plot Current Data", key="plot_the_coords")

        
        with columns[1]:
            st.session_state.manual_coords_df = st.data_editor(
                st.session_state.manual_coords_df,
                num_rows="fixed",
                use_container_width=True,
                column_config={
                    'value': st.column_config.NumberColumn(
                       ## format="%.4e",
                        min_value=min(param_ranges[c][0] for c in self.bayes_manager.input_cols),
                        max_value=max(param_ranges[c][1] for c in self.bayes_manager.input_cols),
                        help="Parameter value (will be clamped to valid range)ZW"
                    )
                },
                key="manual_coords_editor"
            )

        # Clamp values to parameter ranges
        for param in self.bayes_manager.input_cols:
            if param in st.session_state.manual_coords_df.index:
                min_val, max_val = param_ranges[param]
                current_val = st.session_state.manual_coords_df.loc[param, 'value']
                clamped_val = max(min_val, min(max_val, current_val))
                st.session_state.manual_coords_df.loc[param, 'value'] = clamped_val

        # Move the plotting logic outside the columns
        if plot_button:
            coords = list(st.session_state.manual_coords_df['value'].values)
            self.plot_gaussian_process(gp_model=HeteroWhiteSGP, coords=coords)


    def plot_gaussian_process(self, gp_model=HeteroWhiteSGP, coords=None):
        """Plot the Gaussian Process using GPVisualiserPlotly"""
        if self.bayes_manager is None or gp_model is None:
            st.error("Bayesian manager or GP model not provided.")
            return

        plotter = GPVisualiserPlotly(gp=gp_model, obs=self.bayes_manager.obs, dim_cols=self.bayes_manager.input_cols, response_col=self.bayes_manager.response_col)

        fig, ax = plotter.plot_all(coordinates=coords)
        st.plotly_chart(fig, use_container_width=True)


    def plot_group_performance(self):
        import plotly.express as px
        """Plot performance of different groups using GroupPerformancePlotly"""

        # Order groups by median performance (descending)
        df = self.bayes_manager.get_batch_instance_repeat()

        fig = px.box(
            data_frame=df,
            x=self.bayes_manager.group_label,
            y=self.bayes_manager.response_col,
            points="all",

          #  category_orders={self.bayes_manager.group_label: order},
            hover_data=[c for c in self.bayes_manager.input_cols if c in self.bayes_manager.df.columns],
            title="Group Performance Distribution",
        )
        fig.update_traces(boxmean="sd")
        fig.update_layout(
            xaxis_title="Group",
            yaxis_title=self.bayes_manager.response_col,
            legend_title="Group",
            margin=dict(l=10, r=10, t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
        return fig



if __name__ == "__main__":

    # Initialize manager only once using session state
    if "manager" not in st.session_state:
        test_df = pd.DataFrame(
            {
                "x1": [0.1, 0.1, 0.2],
                "x2": [0.2, 0.3, 0.2],
                "response": [1.0, 1.5, 2.0],
                "group": ["A", "A", "B"],
            }
        )

        # st.session_state.manager = GroupManager.init_from_df(test_df, ["x1", "x2"])

        st.set_page_config(layout="wide")

        json_path = r"data/ax_clients/hartmann6_runs.json"
        bayes_manager = BayesClientManager.init_from_json(json_path)
    


        plotter = BayesPlotter(bayes_manager)
        plotter.plot_group_performance()
        plotter.choose_plot_coordinates()
        




