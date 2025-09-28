from tkinter import Y
from typing import Self
from urllib import response
import pandas as pd
import streamlit as st
from io import BytesIO
from ax import Client
import ax_helper

toy_df = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [25, 30, 35, 28, 32],
    'Department': ['Engineering', 'Marketing', 'Sales', 'HR', 'Engineering'],
    'Salary': [70000, 65000, 60000, 55000, 75000],
    'Comments': ['', '', '', '', '']  # Empty last column
})


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
        #if len(trial_instance.unique()) != len(self.df):
        #    batch = 


        return self.df


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

# Show summary
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Data Summary")
    st.write(f"Total rows: {len(edited_df)}")
    st.write(f"Total columns: {len(edited_df.columns)}")

with col2:
    st.subheader("Column Info")
    for col in edited_df.columns:
        non_null = edited_df[col].notna().sum()
        st.write(f"{col}: {non_null}/{len(edited_df)} filled")

with col3:
    st.subheader("Actions")
    
    # Add empty column button
    if st.button("➕ Add Empty Column"):
        new_col_name = f"New_Column_{len(edited_df.columns) + 1}"
        st.session_state.df[new_col_name] = ''
        st.rerun()
    
    # Download button
    if st.button("Download as Excel"):
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            edited_df.to_excel(writer, index=False, sheet_name='Data')
        
        st.download_button(
            label="📥 Download Excel File",
            data=buffer.getvalue(),
            file_name="edited_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )