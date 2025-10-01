from tkinter import Y
from typing import Self
from urllib import response
import pandas as pd
import streamlit as st
from io import BytesIO
from ax import Client
import ax_helper


class BayesClientManager():
    def __init__(self, client):
        self.client = client
        self.df:pd.DataFrame = ax_helper.get_obs_from_client(client)
        self.input_cols: list[str] = list(client._experiment.parameters.keys())
        self.response_col: str = str(list(client._experiment.metrics.keys())[0])

    @staticmethod
    def load_from_json(json_path: str) -> Self:
        client = Client().load_from_json_file(json_path)

        
        return BayesClientManager(client)

    @property   
    def X(self) -> pd.DataFrame:
        return self.df[self.input_cols]

    @property
    def y(self) -> pd.Series:
        return self.df[self.response_col]
    

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
    

    def get_batch_instance_repeat(self, ):
        positions = self.X

        trial_instance = self.df.loc[:, 'trial_name'].str.split('_').map(lambda x: x[0])
        
        trial_dict = {trial:i for i, trial in enumerate(trial_instance.unique())}

        self.df['Group'] = trial_instance.map(trial_dict).astype(int)
        self.unique_trials = trial_dict
        return self.df
    
    @property
    def obs(self):
        return self.df[self.input_cols + [self.response_col]]




    
    def get_best_coordinates(self) -> dict:
        best_row = self.df.loc[self.df[self.response_col].idxmax()]
        return best_row[self.input_cols].to_dict()
    
    def get_parameter_ranges(self) -> dict:
        from ax_helper import ax_param_bounds_as_list
        bounds = list(self.client._experiment.parameters.values())
        # Convert bounds array to dictionary with parameter names as keys
        return {param_name: (bounds[i].lower, bounds[i].upper) for i, param_name in enumerate(self.input_cols)}
    


json_path = r"data/ax_clients/hartmann6_runs.json"
man_df:BayesClientManager =  BayesClientManager.load_from_json(json_path)
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




from GPVisualiser import GPVisualiserMatplotlib
from model_generation import HeteroWhiteSGP
import pandas as pd

def plot_gaussian_process(coords=None):
    st.session_state.show_main = True
    man_df.df = st.session_state.df
    gp = HeteroWhiteSGP
    visualiser = GPVisualiserMatplotlib(gp, man_df.obs, man_df.input_cols, man_df.response_col)
    fig, axs = visualiser.plot_all(coordinates=coords)
    st.pyplot(fig)




def manually_set_coordinates():
    """Allows user to manually set coordinates for plotting the GP"""
    columns = st.columns([0.6,0.4])
    st.subheader("Manually Set Coordinates")

    
        
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


