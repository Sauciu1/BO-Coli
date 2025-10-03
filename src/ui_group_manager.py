from email.policy import default
from pytest import param
import streamlit as st
import pandas as pd
import uuid
import time
from ax_helper import BayesClientManager
import numpy as np


class Group:
    def __init__(
        self,
        X: list[float],
        parameters,
        label: str | None = None,
        responses: list[float] = None,
        group_id: str = None,
    ):
        self.parameters = parameters
        if group_id:
            self.id = group_id
        else:
            # Create a unique ID using UUID to avoid collisions
            unique_suffix = str(uuid.uuid4())[:8]
            timestamp = str(int(time.time() * 1000))[
                -6:
            ]  # Last 6 digits of timestamp in milliseconds
            self.id = f"group_{unique_suffix}_{timestamp}"
        self.label = label or f"Group {self.id}"

        # Initialize in session state if not exists
        if f"group_{self.id}_X" not in st.session_state:
            st.session_state[f"group_{self.id}_X"] = X
        if f"group_{self.id}_responses" not in st.session_state:
            st.session_state[f"group_{self.id}_responses"] = responses or [0.0]



    @property
    def X(self):
        return st.session_state[f"group_{self.id}_X"]

    @X.setter
    def X(self, value):
        st.session_state[f"group_{self.id}_X"] = value

    @property
    def responses(self):
        return st.session_state[f"group_{self.id}_responses"]

    @responses.setter
    def responses(self, value):
        st.session_state[f"group_{self.id}_responses"] = value

    @st.fragment
    def _render_X_row(self, parameters_visible = False) -> list[float]:
        """Render the X values row with editable fields"""
        x_df = pd.DataFrame([self.X], columns=self.parameters)

        edited_x = st.data_editor(
            x_df,
            # width=True,
            num_rows="fixed",
            column_config={
                col: st.column_config.NumberColumn(
                        
                    # format="%.4e",
                    help="Parameter value"
                )
                for col in x_df.columns
            },
            key=f"x_values_{self.id}",
            hide_index=True,
        )

        # Update X values if changed
        if not edited_x.empty:
            self.X = edited_x.iloc[0].tolist()

    @st.fragment
    def render(self):
        """Render the group interface with editable X values and response boxes"""
        cols = st.columns([1, 1.4])  # Make observations column wider
        with cols[0]:
            self._render_X_row()
        with cols[1]:
            self._render_y_row()

        return "updated"

    @st.fragment
    def _render_y_row(self):
        # Create columns for responses + control buttons
        n_resp = len(self.responses)
        response_cols = st.columns([1]*n_resp + [0.1*n_resp, 0.1*n_resp], gap=None, width='stretch')

        # Response inputs
        updated_responses = []

        for i, response_val in enumerate(self.responses):
            with response_cols[i]:
                new_val = st.number_input(
                    f"obs {i+1}",
                    value=float(response_val),
                    placeholder=np.nan,
                    key=f"response_{self.id}_{i}",
                    label_visibility="collapsed",
                    step=1e-10,
                    format="%.6e"
                )
                updated_responses.append(new_val)

        # Delete button (only show if more than 1 response)
        delete_clicked = False
        if len(self.responses) > 1:
            with response_cols[len(self.responses)]:
                delete_clicked = st.button(
                    "❌",
                    key=f"delete_response_{self.id}",
                    help="Delete last observation",
                    width="content"
                )

        # Add button
        with response_cols[len(self.responses) + 1]:
            add_clicked = st.button(
                "➕", key=f"add_response_{self.id}", help="Add observation",
                width="content"
            )

        if add_clicked:
            current_responses = self.responses.copy()
            current_responses.append(np.nan)
            self.responses = current_responses
            st.rerun(scope="fragment")

        if delete_clicked:
            current_responses = self.responses.copy()
            current_responses.pop()  # Remove last element
            self.responses = current_responses
            st.rerun(scope="fragment")

        # Update responses if they changed (only if no deletion occurred)
        if not delete_clicked and len(updated_responses) == len(self.responses):
            self.responses = updated_responses


class GroupManager:
    def __init__(self, parameter_labels: list[str], bayes_manager: BayesClientManager = None):
        self.bayes_manager = bayes_manager
        self.parameter_labels = parameter_labels
        if "groups" not in st.session_state:
            st.session_state["groups"] = []
        if "show_pending_only" not in st.session_state:
            st.session_state["show_pending_only"] = True

    @property
    def groups(self) -> list[Group]:
        return st.session_state["groups"]

    def add_group(self, group: Group):
        st.session_state["groups"].append(group)

    def remove_group(self, group: Group):
        st.session_state["groups"] = [g for g in self.groups if g.id != group.id]

    def render_add_group_button(self):
        """Render the add new group button"""
        if st.button("Manually Add New Group"):
            new_group = Group(
                X=[np.nan] * len(self.parameter_labels),
                parameters=self.parameter_labels,
                responses=[np.nan],
                label=f"Group {len(self.groups) + 1}",
            )
            self.add_group(new_group)
            st.rerun()

    def has_pending_values(self, group: Group) -> bool:
        """Check if a group has any NaN values that might be considered pending"""
        def is_nan_safe(value):
            """Safely check if a value is NaN, handling non-numeric types"""
            try:
                return pd.isna(value) or (isinstance(value, (int, float)) and np.isnan(value))
            except (TypeError, ValueError):
                return isinstance(value, str)  # Treat strings as "pending"
        
        # Check if any X values are NaN (might indicate pending input)
        has_nan_x = any(is_nan_safe(x) for x in group.X)
        
        # Check if any responses are NaN (might indicate pending input)
        has_nan_response = any(is_nan_safe(r) for r in group.responses)
        
        return has_nan_x or has_nan_response

    @st.fragment
    def render_all(self):
        # Filter control buttons at the top
        cols = st.columns([1, 1, 0.5, 1, 2])

        with cols[0]:
            pending_count = sum(self.has_pending_values(g) for g in self.groups)
            if st.button("Show Pending Only", key="show_pending_btn"):
                st.session_state["show_pending_only"] = True
                st.rerun(scope="fragment")
            if pending_count == 0:
                st.caption("No currently pending groups")

        with cols[1]:
            if st.button("Show All", key="show_all_btn"):
                st.session_state["show_pending_only"] = False
                st.rerun(scope="fragment")

        with cols[3]:
            st.number_input("Batch Size", min_value=1, value=1, step=1, key="num_new_groups")
            if st.button("Get New Targets"):
                with st.spinner("Generating the new targets... please wait."):

                    # Store current number of trials to know which ones are new
                    current_trials = len(self.bayes_manager.client._experiment.trials)
                    
                    # Get new targets (this adds them to the client)
                    self.bayes_manager.client.get_next_trials(max_trials=st.session_state["num_new_groups"])
                    
                    # Get only the newly added trials
                    new_trials = self.bayes_manager.client._experiment.trials
                    for trial_idx in range(current_trials, len(new_trials)):
                        trial = new_trials[trial_idx]
                        parameter_values = [trial.arm.parameters[param] for param in self.parameter_labels]
                        new_group = Group(
                            X=parameter_values,
                            parameters=self.parameter_labels,
                            responses=[np.nan],
                            label=f"Group {len(self.groups) + 1}",
                        )
                        self.add_group(new_group)
                
                    st.rerun(scope='fragment')
        with cols[4]:
            st.write("#")
            self.render_add_group_button()


                


        # Column headers
        cols = st.columns([1, 1])  # Make observations column wider
        with cols[0]:
            st.write("**Parameters:**")
        with cols[1]:
            st.write("**Observations:**")

        # Filter groups based on current mode
        groups_to_show = self.groups[:]
        if st.session_state.get("show_pending_only", False):
            groups_to_show = [g for g in groups_to_show if self.has_pending_values(g)]

        # Display status
        if st.session_state.get("show_pending_only", False):
            st.caption(f"Showing {len(groups_to_show)} pending groups out of {len(self.groups)} total")

        for group in groups_to_show:  # Use filtered list
            col0, col1, col2 = st.columns([0.08, 1, 0.05], gap=None)  # Reduce delete button column
            with col0:
                st.markdown(f"**{group.label}**")

            with col1:
                group.render()

            with col2:
                if st.button(
                    "🗑️", key=f"delete_group_{group.id}", help="Delete this group"
                ):
                    self.remove_group(group)
                    st.rerun(scope="fragment")

    def get_full_data(self) -> pd.DataFrame:
        """Get a DataFrame with all groups' X values and responses"""
        records = []
        for group in self.groups:
            for response in group.responses:
                record = {f"x{i+1}": val for i, val in enumerate(group.X)}
                record["response"] = response
                record["group_label"] = group.label
                records.append(record)
        return pd.DataFrame(records)
    
    @property
    def agg_stats(self):
        """Get aggregated statistics of the current data"""
        df = self.get_full_data()
        if df.empty:
            return pd.DataFrame()
        else:
            # Filter out NaN values for statistics
            df_clean = df.dropna()
            if not df_clean.empty:
                stats = self.bayes_manager.get_agg_info()
                return stats
            else:
                return None

    @st.fragment
    def show_data_stats(self):
        """Show aggregated statistics of the current data"""
        with st.expander("Group Statistics", expanded=False):
            if self.agg_stats is not None:
                st.dataframe(self.agg_stats, use_container_width=True)
            else:
                st.info("No valid data for statistics.")

    @staticmethod
    def init_from_df(df: pd.DataFrame, parameter_labels: list[str]):
        """Initialize groups from a DataFrame with columns x1, x2, ..., response"""
        manager = GroupManager(parameter_labels)

        # Group by the actual parameter values
        param_columns = [col for col in parameter_labels if col in df.columns]
        if not param_columns:
            return manager

        grouped = df.groupby(param_columns)

        for params, group_df in grouped:
            # Handle single parameter case where params is not a tuple
            if len(param_columns) == 1:
                X = [params]
            else:
                X = list(params)

            responses = group_df["response"].tolist()

            label = (
                group_df["group"].iloc[0]
                if "group" in group_df.columns
                else f"Group {len(manager.groups) + 1}"
            )

            # Use the existing add_group method to simulate user interaction
            new_group = Group(
                X=X, parameters=parameter_labels, label=label, responses=responses
            )
            manager.add_group(new_group)

        return manager
    

    def reload_from_manager(self):
        """Reload all groups from the current BayesClientManager state"""
        # Clear existing groups
        st.session_state["groups"] = []
        
        # Reload from the manager
        df = self.bayes_manager.get_batch_instance_repeat()
        param_columns = [col for col in self.parameter_labels if col in df.columns]
        if not param_columns:
            return
        
        grouped = df.groupby(param_columns)
        for params, group_df in grouped:
            # Handle single parameter case where params is not a tuple
            if len(param_columns) == 1:
                X = [params]
            else:
                X = list(params)

            responses = group_df[self.bayes_manager.response_col].tolist()
            
            label = f"Group {len(self.groups) + 1}"
            new_group = Group(
                X=X, parameters=self.parameter_labels, label=label, responses=responses
            )
            self.add_group(new_group)

    def get_targets_from_client(self):
        """Get target values from the BayesClientManager"""


    @staticmethod
    def init_from_manager(bayes_manager: BayesClientManager):
        """Initialize groups from a BayesClientManager"""
        df = bayes_manager.get_batch_instance_repeat()
        parameter_labels = bayes_manager.input_cols
        group_manager = GroupManager.init_from_df(df, parameter_labels)
        group_manager.bayes_manager = bayes_manager
        return group_manager

    def write_to_manager(self, manager: BayesClientManager):
        """Write all group data to a BayesClientManager"""
        df = self.get_full_data()
        if df.empty:
            return

        # Convert to the format expected by BayesClientManager
        for _, row in df.iterrows():
            # Extract parameter values
            params = {col: row[col] for col in self.parameter_labels if col in row}

            # Add the trial to the manager
            manager.attach_trial(parameters=params, raw_data=row["response"])


if __name__ == "__main__":

    st.title("Group Manager Test")
    st.set_page_config(layout="wide")
    
    # Initialize manager only once using session state
    if "manager" not in st.session_state:
        json_path = r"data/ax_clients/hartmann6_runs.json"
        bayes_manager = BayesClientManager.init_from_json(json_path)
        st.session_state.manager = GroupManager.init_from_manager(bayes_manager)

    manager = st.session_state.manager
    

    
    manager.render_all()
    manager.show_data_stats()

    st.write("### Full Data")
    st.dataframe(manager.get_full_data(), use_container_width=True)
