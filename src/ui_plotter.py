import json

from ipykernel.pickleutil import can
from src.GPVisualiser import GPVisualiserPlotly
from src.model_generation import HeteroWhiteSGP
from src.ax_helper import BayesClientManager
import pandas as pd
import streamlit as st
from botorch.models import SingleTaskGP



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
            plot_button = st.button("Plot Gaussian Process", key="plot_the_coords")

        
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
        if plot_button and self.can_plot:
            coords = list(st.session_state.manual_coords_df['value'].values)
            if len(coords) != len(self.bayes_manager.input_cols):
                st.error(f"Please provide exactly {len(self.bayes_manager.input_cols)} coordinates.")
            self.plot_gaussian_process(gp_model=SingleTaskGP, coords=coords)
        else:
            st.warning("No data available to plot the GP.")


    def plot_gaussian_process(self, gp_model=SingleTaskGP, coords=None):
        """Plot the Gaussian Process using GPVisualiserPlotly"""
        if self.bayes_manager is None or gp_model is None:
            st.error("Bayesian manager or GP model not provided.")
            return
        elif not self.can_plot:
            st.warning("No observations available to plot the GP.")
            return
        

        plotter = GPVisualiserPlotly(gp=gp_model, obs=self.bayes_manager.obs, dim_cols=self.bayes_manager.input_cols, response_col=self.bayes_manager.response_col)

        fig, ax = plotter.plot_all(coordinates=coords)
        st.plotly_chart(fig, use_container_width=True)



    def plot_group_performance(self):
        import plotly.express as px
        """Performance of each observation group as box plot"""
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
    
    @property
    def can_plot(self):
        return not self.bayes_manager.df.empty
    

    def main_loop(self):
        if st.button("Plot Observation Performance", key="plot_obs_performance"):
            if not self.can_plot:
                st.warning("No data available to plot.")
            else:
                self.plot_group_performance()
        self.choose_plot_coordinates()



if __name__ == "__main__":
    def load_ackley():
        auckley_path = r"data/ax_clients/ackley_client.pkl"
        import pickle
        client = pickle.load(open(auckley_path, "rb"))
        return BayesClientManager(client=client)

    def load_manual():


        if "manager" not in st.session_state:
            test_df = pd.DataFrame(
                {
                    "x1": [0.1, 0.1, 0.2],
                    "x2": [0.2, 0.3, 0.2],
                    "response": [1.0, 1.5, 2.0],
                    "group": ["A", "C", "B"],
                }
            )

        from src.ui_group_manager import GroupManager
        return GroupManager.init_from_df(test_df, ["x1", "x2"])

        #json_path = r"data/ax_clients/hartmann6_runs.json"



    bayes_manager = load_ackley()
    plotter = BayesPlotter(bayes_manager)
    plotter.main_loop()
        
        




