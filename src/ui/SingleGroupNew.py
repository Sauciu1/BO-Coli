import streamlit as st
import pandas as pd
import numpy as np


class SingleGroup:
    def __init__(self, group_df, group_label, feature_labels=['x1', 'x2'], response_label='y'):
        self.group_df = group_df
        self.group_label = group_label
        self.feature_labels = feature_labels
        self.response_label = response_label
    
    @st.fragment
    def render(self):
        st.markdown(f"### Group {self.group_label}")
        st.caption(f"{len(self.group_df)} trial(s)")
        
        cols = st.columns([1, 2])
        
        with cols[0]:
            st.write("**Parameters:**")
            if not self.group_df.empty:
                # Show X values (assuming all trials in group have same X values)
                x_values = [self.group_df.iloc[0][param] for param in self.feature_labels]
                display_x = [0.0 if pd.isna(val) else val for val in x_values]
                
                st.data_editor(
                    pd.DataFrame([display_x], columns=self.feature_labels),
                    num_rows="fixed",
                    key=f"x_values_group_{self.group_label}",
                    hide_index=True,
                    disabled=True  # Read-only for simplicity
                )
        
        with cols[1]:
            st.write("**Response Values:**")
            if not self.group_df.empty:
                responses = self.group_df[self.response_label].tolist()
                response_data = {f"Trial {i+1}": [resp if not pd.isna(resp) else None] 
                               for i, resp in enumerate(responses)}
                
                st.data_editor(
                    pd.DataFrame(response_data),
                    key=f"responses_group_{self.group_label}",
                    hide_index=True,
                    num_rows="fixed",
                    disabled=True  # Read-only for simplicity
                )
            else:
                st.info("No trials in this group")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Single Group Example")

    # Create sample data directly
    df = pd.DataFrame({
        "x1": [0.1, 0.4, 0.5, 0.7, 0.1],
        "x2": [1.0, 0.9, 0.8, 0.6, 1.0],
        "y": [0.5, 0.6, 0.55, np.nan, 0.45],
        "group": [0, 1, 2, 3, 0]  # Group 0 has two trials
    })
    
    # Get groups from the dataframe
    for group_label in sorted(df['group'].unique()):
        group_df = df[df['group'] == group_label]
        group = SingleGroup(group_df, group_label, feature_labels=['x1', 'x2'], response_label='y')
        group.render()
        st.divider()
    
    # Show raw data
    with st.expander("Raw Data", expanded=False):
        st.dataframe(df, use_container_width=True)