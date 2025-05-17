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
    with st.expander(title, expanded=True):
        for key, value in explanation.items():
            st.markdown(f'<div class="explanation-bubble"><strong>{key}:</strong> {value}</div>', unsafe_allow_html=True)

def toggle_outputs() -> tuple[bool, bool, bool]:
    """
    Allow users to toggle which outputs to display with descriptive labels.

    Returns:
        tuple[bool, bool, bool]: Booleans indicating whether to show descriptive stats, LS Means, and plots.
    """
    show_descriptive = st.checkbox(
        "Show a table of basic statistics (like averages and spreads)",
        value=True,
        help="This shows a table with summary stats like means, standard deviations, and counts for each group."
    )
    show_ls_means = st.checkbox(
        "Show a table and chart of adjusted averages (LS Means)",
        value=True,
        help="This shows a table and chart of least squares means, which are adjusted averages that account for other factors in the model."
    )
    show_plot = st.checkbox(
        "Show a chart comparing groups or trends (Box/Interaction Plot)",
        value=True,
        help="This shows a visual plot to compare groups (box plot) or trends over time (interaction plot), depending on the model."
    )
    return show_descriptive, show_ls_means, show_plot