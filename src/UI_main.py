from tkinter import Y
from typing import Self
from urllib import response
import pandas as pd
import streamlit as st
from io import BytesIO
from ax import Client
import ax_helper


class BayesDFManager():
    def __init__(self, df:pd.DataFrame, input_cols:list[str], response_col:list[str]):
        self.df:pd.DataFrame = df
        self.input_cols: list[str] = input_cols
        self.response_col: str = response_col

    @property   
    def X(self) -> pd.DataFrame:
        return self.df[self.input_cols]

    @property
    def y(self) -> pd.Series:
        return self.df[self.response_col]
    

    def get_column_config(self) -> dict:
        config = {}

        columns = ['Group'] + self.input_cols + self.response_col

        for col in columns:
            dtype = self.df[col].dtype

            if col in self.response_col:
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
        return self.df[self.input_cols + self.response_col]



    @staticmethod
    def load_from_json(json_path: str) -> Self:
        client = Client().load_from_json_file(json_path)
        input_cols: list[str] = list(client._experiment.parameters.keys())
        response_col: list[str] = list(client._experiment.metrics.keys())
        df = ax_helper.get_obs_from_client(client)
        return BayesDFManager(df, input_cols, response_col)





json_path = r"data/ax_clients/hartmann6_runs.json"
man_df =  BayesDFManager.load_from_json(json_path)
print(man_df.get_batch_instance_repeat())



st.set_page_config(page_title="Excel-Style Data Editor", layout="wide")
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



from GPVisualiser import GPVisualiser
from model_generation import HeteroWhiteSGP

def show_gp_visualisation():
    if st.button("Show GP Visualisation"):
        gp = HeteroWhiteSGP
        visualiser = GPVisualiser(gp, man_df.obs, man_df.input_cols, man_df.response_col)
        fig, axs = visualiser.plot_all()
        st.pyplot(fig)
