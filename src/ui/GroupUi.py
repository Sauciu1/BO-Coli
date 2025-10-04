




class GroupUi:
    def __init__(self, bayes_manager: BayesClientManager):
        self.bayes_manager = bayes_manager
        st.session_state.setdefault("show_pending_only", True)
        st.session_state.setdefault("data_version", 0)

    @property
    def groups(self) -> list[Group]:
        if self.bayes_manager.data.empty: return []
        unique_groups = sorted(self.bayes_manager.data[self.bayes_manager.group_label].unique())
        groups = [Group(self, g) for g in unique_groups if not pd.isna(g)]
        [g.invalidate_cache() for g in groups]
        return groups
    
    def _notify_data_change(self):
        st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1
        """TODO explicitly update bayes manager"""
        [g.invalidate_cache() for g in self.groups]

    def update_bayes_manager_from_self(self):
        self.bayes_manager.data = self.bayes_manager.data.copy(deep=True).reset_index(drop=True)
        




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
        self._notify_data_change()

    def remove_group(self, group_number: int):
        self.bayes_manager.data = self.bayes_manager.data[
            self.bayes_manager.data[self.bayes_manager.group_label] != group_number].reset_index(drop=True)
        self._notify_data_change()


    @st.fragment
    def render_all(self):
        cols = st.columns([1, 1, 1, 2])
        
        with cols[0]:
            if st.button("Show Pending Only"):
                st.session_state["show_pending_only"] = True
                st.rerun(scope="fragment")
            if (pending_count := sum(g.has_pending for g in self.groups)) == 0:
                st.caption("No pending groups")
        
        with cols[1]:
            if st.button("Show All"):
                st.session_state["show_pending_only"] = False
                st.rerun(scope="fragment")
        
        with cols[2]:
            batch_size = st.number_input("Batch Size", min_value=1, value=1, step=1)
            if st.button("Get New Targets"):
                self._notify_data_change()
                with st.spinner("Generating targets..."):
                    self.bayes_manager.get_batch_targets(batch_size)
                st.rerun(scope='fragment')
        
        with cols[3]:
            if st.button("Add Manual Group"):
                self.add_manual_group()
                st.rerun()

        groups_to_show = [g for g in self.groups if not st.session_state["show_pending_only"] or g.has_pending]
        
        if st.session_state["show_pending_only"]:
            st.caption(f"Showing {len(groups_to_show)} pending of {len(self.groups)} total groups")

        for group in groups_to_show:
            with st.container():
                col0, col1, col2 = st.columns([0.08, 1, 0.05])
                
                with col0:
                    st.markdown(f"**Group {group.group_number}**")
                    st.caption(f"{len(group.trials)} trial(s)")
                
                with col1: group.render()
                
                with col2:
                    if st.button("🗑️", key=f"delete_group_{group.group_number}", help="Delete group"):
                        self.remove_group(group.group_number)
                        st.rerun(scope="fragment")
                
                st.divider()

    @st.fragment  
    def show_data_stats(self):
        with st.expander("Data Statistics", expanded=False):
            if not self.bayes_manager.data.empty:
                st.dataframe(self.bayes_manager.agg_stats, use_container_width=True)
            else:
                st.info("No data available.")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Group Manager Test")
    
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
    
    if "bayes_manager" not in st.session_state:
        st.session_state.bayes_manager = BayesClientManager(
            data=df,
            feature_labels=['x1', 'x2'],
            bounds=bounds,
            response_label='response'
        )
    
    ui = GroupUi(st.session_state.bayes_manager)
    ui.render_all()
    ui.show_data_stats()
    

