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
    """Display data in grouped sub-windows with add/delete functionality"""
    
    # Group the dataframe by 'Group' column
    if 'Group' not in st.session_state.df.columns:
        st.error("No 'Group' column found in the data")
        return
    
    grouped = st.session_state.df.groupby('Group')
    
    # Display each group in its own container
    for group_id, group_data in grouped:
        with st.container():
            st.subheader(f"Group {group_id}")
            
            # Create the group interface
            display_group_interface(group_id, group_data, man_df)
            
            st.divider()
    
    # Button to add a new group
    if st.button("➕ Add New Group", key="add_new_group"):
        add_new_group(man_df)

def display_group_interface(group_id, group_data, man_df):
    """Display interface for a single group"""
    
    col1, col2 = st.columns([0.9, 0.1])
    
    with col2:
        # Delete group button
        if st.button("🗑️", key=f"delete_group_{group_id}", help="Delete this group"):
            delete_group(group_id)
            st.rerun()
    
    with col1:
        # Parameters row (should be the same for all rows in the group)
        if not group_data.empty:
            first_row = group_data.iloc[0]
            
            # Display parameters
            param_cols = st.columns(len(man_df.input_cols) + 1)
            param_cols[0].write("**Parameters**")
            
            for i, col in enumerate(man_df.input_cols):
                param_cols[i + 1].write(f"**{col}**")
                param_cols[i + 1].write(f"{first_row[col]:.4e}")
        
        # Responses section - individual input cells
        st.write("**Responses**")
        
        responses = group_data[man_df.response_col].values
        response_cols = st.columns(len(responses) + 1)
        
        updated_responses = []
        for i, response_val in enumerate(responses):
            with response_cols[i]:
                # Individual response input with delete button
                col_input, col_delete = st.columns([0.8, 0.2])
                
                with col_input:
                    new_val = st.number_input(
                        f"r{i+1}",
                        value=float(response_val),
                        format="%.6e",
                        key=f"response_{group_id}_{i}"
                    )
                    updated_responses.append(new_val)
                
                with col_delete:
                    if st.button("❌", key=f"delete_response_{group_id}_{i}", help="Delete this response"):
                        delete_response_from_group(group_id, i, man_df)
                        st.rerun()
        
        # Add response button
        with response_cols[len(responses)]:
            if st.button("➕", key=f"add_response_{group_id}", help="Add response"):
                add_response_to_group(group_id, group_data, man_df)
                st.rerun()
        
        # Update responses if they changed
        if updated_responses != list(responses):
            update_responses_from_inputs(group_id, updated_responses, group_data, man_df)

def update_responses_from_inputs(group_id, response_values, original_group_data, man_df):
    """Update responses from individual input changes"""
    st.session_state.df = st.session_state.df[st.session_state.df['Group'] != group_id]
    
    if response_values and not original_group_data.empty:
        first_row = original_group_data.iloc[0]
        new_rows = []
        for response_value in response_values:
            new_row = first_row.copy()
            new_row[man_df.response_col] = response_value
            new_rows.append(new_row)
        
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)

def delete_response_from_group(group_id, response_index, man_df):
    """Delete a specific response from a group"""
    group_data = st.session_state.df[st.session_state.df['Group'] == group_id]
    if len(group_data) > response_index:
        # Remove the specific response row
        row_to_remove = group_data.iloc[response_index].name
        st.session_state.df = st.session_state.df.drop(row_to_remove)

def update_group_responses_horizontal(group_id, response_values, original_group_data, man_df):
    """Update the main dataframe with horizontally edited responses"""
    
    # Remove old group data from session state
    st.session_state.df = st.session_state.df[st.session_state.df['Group'] != group_id]
    
    # Create new rows for the updated responses
    if len(response_values) > 0 and not original_group_data.empty:
        first_row = original_group_data.iloc[0]
        
        new_rows = []
        for response_value in response_values:
            new_row = first_row.copy()
            new_row[man_df.response_col] = response_value
            new_rows.append(new_row)
        
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)

def update_group_responses(group_id, edited_responses, original_group_data, man_df):
    """Update the main dataframe with edited responses"""
    
    # Get the indices of the original group data
    group_indices = original_group_data.index.tolist()
    
    # Remove old group data from session state
    st.session_state.df = st.session_state.df[st.session_state.df['Group'] != group_id]
    
    # Create new rows for the updated responses
    if not edited_responses.empty and not original_group_data.empty:
        first_row = original_group_data.iloc[0]
        
        new_rows = []
        for _, response_row in edited_responses.iterrows():
            new_row = first_row.copy()
            new_row[man_df.response_col] = response_row[man_df.response_col]
            new_rows.append(new_row)
        
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)

def add_response_to_group(group_id, group_data, man_df):
    """Add a new response to an existing group"""
    if not group_data.empty:
        # Create a new row based on the first row of the group
        new_row = group_data.iloc[0].copy()
        new_row[man_df.response_col] = 0.0  # Default response value
        
        # Add to session state
        new_df = pd.DataFrame([new_row])
        st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)
        st.rerun()

def delete_group(group_id):
    """Delete an entire group"""
    st.session_state.df = st.session_state.df[st.session_state.df['Group'] != group_id]

def add_new_group(man_df):
    """Add a new group with default parameter values"""
    
    # Find the next available group ID
    if st.session_state.df.empty:
        new_group_id = 1
    else:
        new_group_id = st.session_state.df['Group'].max() + 1
    
    # Get parameter ranges for default values
    param_ranges = man_df.get_parameter_ranges()
    
    # Create new row with default parameter values (midpoint of ranges)
    new_row = {'Group': new_group_id}
    
    for param in man_df.input_cols:
        if param in param_ranges:
            min_val, max_val = param_ranges[param]
            new_row[param] = (min_val + max_val) / 2
        else:
            new_row[param] = 0.0
    
    new_row[man_df.response_col] = 0.0
    
    # Add other required columns if they exist
    for col in st.session_state.df.columns:
        if col not in new_row:
            new_row[col] = None
    
    # Add to session state
    new_df = pd.DataFrame([new_row])
    st.session_state.df = pd.concat([st.session_state.df, new_df], ignore_index=True)
    st.rerun()

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


