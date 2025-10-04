
import streamlit as st
import pandas as pd
import numpy as np
import uuid
from src.BayesClientManager import BayesClientManager
from src.ui.SingleGroup import SingleGroup


class GroupUi:
    def __init__(self, bayes_manager: BayesClientManager):
        self.bayes_manager = bayes_manager
        st.session_state.setdefault("show_pending_only", True)
        st.session_state.setdefault("groups", {})
    
    @property
    def group_data(self):
        """Get groups data from bayes manager"""
        return self.bayes_manager.get_groups()
    
    @property
    def groups(self):
        """Get or create SingleGroup instances with proper data synchronization"""
        current_groups = {}
        
        # Get fresh data from bayes_manager each time
        for group_label, group_df in self.group_data.items():
            if group_label not in st.session_state.groups:
                st.session_state.groups[group_label] = SingleGroup(
                    group_df, group_label, self.bayes_manager
                )
            else:
                # Update existing group with fresh data from manager
                existing_group = st.session_state.groups[group_label]
                existing_group.group_df = group_df.copy()
            
            current_groups[group_label] = st.session_state.groups[group_label]
        
        # Clean up removed groups
        removed_groups = set(st.session_state.groups.keys()) - set(current_groups.keys())
        for group_label in removed_groups:
            del st.session_state.groups[group_label]
        
        return current_groups
    
    @property
    def pending_groups(self):
        """Get groups that have pending (NaN) responses"""
        return {label: group for label, group in self.groups.items() 
                if not group.group_df[self.bayes_manager.response_label].notna().all()}
    
    @property
    def has_data(self):
        """Check if there's any data available"""
        return not self.bayes_manager.data.empty
        

    def add_manual_group(self):
        if self.bayes_manager.data.empty:
            next_group = 0
        else:
            max_g = self.bayes_manager.data[self.bayes_manager.group_label].max()
            next_group = max_g + 1 if not pd.isna(max_g) else 0
        
        new_row = {**{param: np.nan for param in self.bayes_manager.feature_labels},
                   self.bayes_manager.response_label: np.nan, self.bayes_manager.group_label: next_group,
                   self.bayes_manager.id_label: f"manual_{str(uuid.uuid4())[:8]}"}
        
        self.bayes_manager.data = pd.concat([self.bayes_manager.data, pd.DataFrame([new_row])], ignore_index=True)
   

    def remove_group(self, group_label: int):
        """Remove a group from the data and session state"""
        # Remove from bayes manager data
        self.bayes_manager.data = self.bayes_manager.data[
            self.bayes_manager.data[self.bayes_manager.group_label] != group_label].reset_index(drop=True)
        
        # Remove from session state
        if group_label in st.session_state.groups:
            del st.session_state.groups[group_label]


    @st.fragment
    def render_all(self):
        # Control buttons
        self._render_controls()

        st.divider()
        
        # Determine which groups to show
        groups_to_show: dict[str, SingleGroup] = self.pending_groups if st.session_state["show_pending_only"] else self.groups
        
        if st.session_state["show_pending_only"]:
            st.caption(f"Showing {len(groups_to_show)} pending of {len(self.groups)} total groups")
        
        # Render each group
        for group_label, group in groups_to_show.items():
            with st.container():
                cols = st.columns([0.05, 1])
                
                with cols[1]:
                    group.render()
                    # Ensure data is synced back to manager after rendering
                    group.write_data_to_manager()
                
                with cols[0]:
                    st.write("")
                    st.write("")
                    if st.button("🗑️", key=f"delete_group_{group_label}", help="Delete group"):
                        self.remove_group(group_label)
                        st.rerun(scope="fragment")
                
                st.divider()
    
    def _render_controls(self):
        """Render control buttons"""
        cols = st.columns([1, 1, 2, 2])
        
        with cols[0]:
            if st.button("Show Pending Only"):
                st.session_state["show_pending_only"] = True
                st.rerun(scope="fragment")
            
            pending_count = len(self.pending_groups)
            if pending_count == 0:
                st.caption("No pending groups")
            else:
                st.caption(f"{pending_count} pending")
        
        with cols[1]:
            if st.button("Show All"):
                st.session_state["show_pending_only"] = False
                st.rerun(scope="fragment")
        
        with cols[2]:
            self._get_new_targets()
        
        with cols[3]:
            if st.button("Add Manual Group"):
                self.add_manual_group()
                st.rerun()

    def _get_new_targets(self):
        cols = st.columns([1, 1.7, 0.2])
        with cols[0]:
            batch_size = st.number_input("Batch Size", min_value=1, value=1, step=1)
        with cols[1]:
            if target_button := st.button("Get New Targets", disabled=False):
                with st.spinner("Generating targets..."):
                    try:
                        # Get new targets from BayesClientManager
                        self.bayes_manager.get_batch_targets(batch_size)
                        # Clear cached groups to force refresh with new data
                        st.session_state.groups = {}
                        st.rerun(scope='fragment')
                    except Exception as e:
                        st.error(f"Error generating targets: {str(e)}")
                        st.error("The model needs more data or has fully explored the space.")

            
    

    @st.fragment
    def show_data_stats(self):
        """Display data statistics and raw data views"""
        if not self.has_data:
            st.info("No data available.")
            return
        
        with st.expander("Data Statistics", expanded=False):
            try:
                stats = self.bayes_manager.agg_stats
                if not stats.empty:
                    st.dataframe(stats, width="stretch")
                else:
                    st.info("No aggregated statistics available.")
            except Exception as e:
                st.error(f"Error calculating statistics: {str(e)}")
        
        with st.expander("Raw Data", expanded=False):
            st.dataframe(self.bayes_manager.data, width="stretch")
        
        # Data export and analysis buttons
        cols = st.columns([1, 1, 1])
        
        with cols[0]:
            if st.button("Get All Group Data"):
                st.write("### Current Data from All Groups:")
                df = self.get_current_data()
                st.dataframe(df, width="stretch")
        
        with cols[1]:
            if st.button("Sync All Groups"):
                self.sync_all_groups_to_manager()
                st.success("All groups synchronized to manager")
        
        with cols[2]:
            if self.bayes_manager.has_response:
                pending_count = len(self.bayes_manager.pending_targets)
                st.metric("Pending Trials", pending_count)

    def get_current_data(self):
        """Get current data from BayesClientManager (source of truth)"""
        return self.bayes_manager.data.copy()
    
    def sync_all_groups_to_manager(self):
        """Synchronize all group changes back to the BayesClientManager"""
        for group in self.groups.values():
            group.write_data_to_manager()


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Group Manager")
    
    # Initialize session state
    if "bayes_manager" not in st.session_state:
        # Initialize test data
        df = pd.DataFrame({
            'x1': [0.1, 0.4, 0.5, 0.7, 0.1],
            'x2': [1.0, 0.9, 0.8, 0.6, 1.0],
            'response': [0.5, 0.6, 0.55, np.nan, 0.45]
        })
        
        bounds = {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False},
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }
        
        st.session_state.bayes_manager = BayesClientManager(
            data=df,
            feature_labels=['x1', 'x2'],
            bounds=bounds,
            response_label='response'
        )


    
    # Create and render GroupUi
    ui = GroupUi(st.session_state.bayes_manager)
    ui.render_all()
    ui.show_data_stats()
    

