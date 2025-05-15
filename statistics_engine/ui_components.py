import streamlit as st
import pandas as pd
from matplotlib.figure import Figure
from typing import Any

def display_progress(step: int, total_steps: int) -> None:
    """
    Display a progress bar for the analysis workflow.

    Args:
        step (int): Current step.
        total_steps (int): Total number of steps.
    """
    progress = step / total_steps
    st.progress(progress)
    st.write(f"Step {step}/{total_steps}")

def display_result_section(title: str, content: Any, export_label: str, export_data: bytes, mime: str, filename: str) -> None:
    """
    Display a result section with export option.

    Args:
        title (str): Section title.
        content (Any): Content to display (e.g., DataFrame, plot).
        export_label (str): Label for export button.
        export_data (bytes): Data to export.
        mime (str): MIME type for export.
        filename (str): Export filename.
    """
    st.write(f"**{title}:**")
    if isinstance(content, pd.DataFrame):
        st.dataframe(content)
    elif isinstance(content, Figure):
        st.pyplot(content)
    else:
        st.write(content)
    st.download_button(
        label=export_label,
        data=export_data,
        file_name=f"{filename}.{'csv' if mime == 'text/csv' else 'png'}",
        mime=mime
    )

def display_explanation(title: str, explanation: dict) -> None:
    """
    Display an explanation section.

    Args:
        title (str): Explanation title.
        explanation (dict): Explanation content.
    """
    with st.expander(title):
        for key, value in explanation.items():
            st.markdown(f'<div class="explanation-bubble"><strong>{key}:</strong> {value}</div>', unsafe_allow_html=True)

def toggle_outputs() -> tuple[bool, bool, bool]:
    """
    Allow users to toggle which outputs to display.

    Returns:
        tuple[bool, bool, bool]: Booleans indicating whether to show descriptive stats, LS Means, and plots.
    """
    show_descriptive = st.checkbox("Show Descriptive Statistics", value=True)
    show_ls_means = st.checkbox("Show LS Means Table and Plot", value=True)
    show_plot = st.checkbox("Show Box/Interaction Plot", value=True)
    return show_descriptive, show_ls_means, show_plot