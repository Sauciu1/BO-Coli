import streamlit as st
import pandas as pd
import uuid
import time


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
    def _render_X_row(self) -> list[float]:
        col1, col2 = st.columns([0.2, 0.8])

        with col1:
            st.write("**Parameters:**")

        with col2:
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

        with st.expander(self.label, expanded=True):
            self._render_X_row()
            self._render_y_row()

        return "updated"

    @st.fragment
    def _render_y_row(self):
        # Response input boxes
        col1, col2 = st.columns([0.2, 0.8])

        with col1:
            st.write("**Observations:**")

        with col2:
            # Create columns: one for add button + response columns
            response_cols = st.columns(1 + len(self.responses))

            # Response inputs in remaining columns
            delete_index = None
            updated_responses = []

            for i, response_val in enumerate(self.responses):
                with response_cols[i]:
                    # Individual response input with delete button
                    new_val = st.number_input(
                        f"obs {i+1}",
                        value=float(response_val),
                        # format="%.4e",
                        key=f"response_{self.id}_{i}",
                        label_visibility="hidden",
                    )
                    updated_responses.append(new_val)

                    if (
                        len(self.responses) > 1
                    ):  # Don't allow deleting the last response
                        if st.button(
                            "❌",
                            key=f"delete_response_{self.id}_{i}",
                            help="Delete this observation",
                        ):
                            delete_index = i

            with response_cols[-1]:
                add_clicked = st.button(
                    "➕", key=f"add_response_{self.id}", help="Add response"
                )

                if add_clicked:
                    current_responses = self.responses.copy()
                    current_responses.append(0.0)
                    self.responses = current_responses
                    st.rerun(scope="fragment")

            # Handle deletion after the loop to avoid index issues
            if delete_index is not None:
                current_responses = self.responses.copy()
                current_responses.pop(delete_index)
                self.responses = current_responses
                st.rerun(scope="fragment")

            # Update responses if they changed (only if no deletion occurred)
            if delete_index is None and len(updated_responses) == len(self.responses):
                self.responses = updated_responses


class GroupManager:
    def __init__(self, parameter_labels: list[str]):
        self.parameter_labels = parameter_labels
        if "groups" not in st.session_state:
            st.session_state["groups"] = []

    @property
    def groups(self) -> list[Group]:
        return st.session_state["groups"]

    def add_group(self, group: Group):
        st.session_state["groups"].append(group)

    def remove_group(self, group: Group):
        st.session_state["groups"] = [g for g in self.groups if g.id != group.id]

    def render_add_group_button(self):
        """Render the add new group button"""
        if st.button("Add New Group"):
            new_group = Group(
                X=[0.0] * len(self.parameter_labels),
                parameters=self.parameter_labels,
                label=f"Group {len(self.groups) + 1}",
            )
            self.add_group(new_group)
            st.rerun()

    @st.fragment
    def render_all(self):
        for group in self.groups[:]:  # Use slice to avoid modification during iteration
            col1, col2 = st.columns([0.9, 0.1])

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

    @st.fragment
    def show_full_data(self):
        with st.expander("Group Statistics", expanded=False):
            df = self.get_full_data()
            if df.empty:
                st.info("No data available yet.")
            else:
                stats = (
                    df.groupby("group_label")["response"]
                    .agg(["count", "mean", "std"])
                    .reset_index()
                    .rename(
                        columns={
                            "group_label": "Group",
                            "count": "N",
                            "mean": "Mean",
                            "std": "Std",
                        }
                    )
                )
                st.dataframe(stats, use_container_width=True)

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

            label = group_df["group"].iloc[0] if "group" in group_df.columns else f"Group {len(manager.groups) + 1}"

            # Use the existing add_group method to simulate user interaction
            new_group = Group(
                X=X, parameters=parameter_labels, label=label, responses=responses
            )
            manager.add_group(new_group)

        return manager


if __name__ == "__main__":

    st.title("Group Manager Test")

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
        st.session_state.manager = GroupManager.init_from_df(test_df, ["x1", "x2"])

    manager = st.session_state.manager
    manager.get_full_data()

    manager.render_add_group_button()

    manager.show_full_data()

    manager.render_all()
