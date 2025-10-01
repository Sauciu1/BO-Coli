from GPVisualiser import GPVisualiserMatplotlib, GPVisualiserPlotly
from model_generation import HeteroWhiteSGP
import pandas as pd
import streamlit as st

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