from tkinter import Y

from urllib import response
import pandas as pd
import streamlit as st
from io import BytesIO
from ax import Client
import ax_helper
from ax_helper import BayesClientManager

class stBayesClientManager(BayesClientManager):
    """Extends BayesClientManager to provide Streamlit-specific functionality"""

    def get_column_config(self) -> dict:
        config = {}

        columns = ['Group'] + self.input_cols + [self.response_col]

        for col in columns:
            if col == self.response_col:
                config[col] = st.column_config.NumberColumn(format="scientific", help="Experiment result")
            elif col in self.input_cols:
                config[col] = st.column_config.NumberColumn(format="scientific", help="Input parameter")
            elif col == 'trial_name':
                config[col] = None
            elif col == 'Group':
                config[col] = st.column_config.NumberColumn(format="%d", help="Group of technical repeats")
            else:
                continue
        return config

        
    @classmethod
    def load_from_json(cls, json_path):
        """Override to return stBayesClientManager instance"""
        temp_manager = super().load_from_json(json_path)
        instance = cls(temp_manager.client, temp_manager.input_cols, temp_manager.response_col)
        instance.df = temp_manager.df
        return instance


    


json_path = r"data/ax_clients/hartmann6_runs.json"
# Load as BayesClientManager first, then convert to stBayesClientManager
temp_manager = ax_helper.BayesClientManager.load_from_json(json_path)
man_df = stBayesClientManager(temp_manager.client)
# Copy over the dataframe
man_df.df = temp_manager.df

print(man_df.get_batch_instance_repeat())



st.set_page_config(page_title="Bayesian Optimization",layout="wide")
# Initialize session state with BayesDFManager and its DataFrame

st.session_state.df = man_df.df
 


st.info(f"Loaded Bayesian DF with {len(st.session_state.df)} rows and "
        f"{len(st.session_state.df.columns)} columns.")
st.title("Excel-Style Data Sheet")
st.write("Upload a file or use sample data. Edit the data below.")

# File upload section
uploaded_file = st.file_uploader(
    "Choose a file", 
    type=['.json'],
    help="Upload a .json file to load Bayesian optimization data."
)



def df_runner(man_df):
    # Display editable dataframe
    edited_df = st.data_editor(
        st.session_state.df,
        use_container_width=True,
        num_rows="dynamic",
        column_config=man_df.get_column_config(),

        key="data_editor"
    )

    # Update session state with edited data
    st.session_state.df = edited_df

df_runner(man_df)




from GPVisualiser import GPVisualiserMatplotlib, GPVisualiserPlotly
from model_generation import HeteroWhiteSGP
import pandas as pd

def plot_gaussian_process(coords=None):
    st.session_state.show_main = True
    man_df.df = st.session_state.df
    gp = HeteroWhiteSGP
    visualiser = GPVisualiserPlotly(gp, man_df.obs, man_df.input_cols, man_df.response_col)
    fig, axs = visualiser.plot_all(coordinates=coords)
    st.plotly_chart(fig, use_container_width=True)

def manually_set_coordinates():
    """Allows user to manually set coordinates for plotting the GP"""
    st.subheader("Plot GP at Specific Coordinates")
    columns = st.columns([0.6,0.4])
   

    
        
    if "manual_coords_df" not in st.session_state:
        defaults = man_df.get_best_coordinates()
        df = pd.DataFrame([defaults], ).transpose()
        df.columns = ['value']
        st.session_state.manual_coords_df = df

    param_ranges = man_df.get_parameter_ranges()

    with columns[0]:
        if st.button("Set to Best Performer", key="set_best_performer"):
            defaults = man_df.get_best_coordinates()
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
                    format="%.4e",
                    min_value=min(param_ranges[c][0] for c in man_df.input_cols),
                    max_value=max(param_ranges[c][1] for c in man_df.input_cols),
                    help="Parameter value (will be clamped to valid range)"
                )
            },
            key="manual_coords_editor"
        )

    # Clamp values to parameter ranges
    for param in man_df.input_cols:
        if param in st.session_state.manual_coords_df.index:
            min_val, max_val = param_ranges[param]
            current_val = st.session_state.manual_coords_df.loc[param, 'value']
            clamped_val = max(min_val, min(max_val, current_val))
            st.session_state.manual_coords_df.loc[param, 'value'] = clamped_val

    # Move the plotting logic outside the columns
    if plot_button:
        coords = list(st.session_state.manual_coords_df['value'].values)
        plot_gaussian_process(coords=coords)


manually_set_coordinates()


