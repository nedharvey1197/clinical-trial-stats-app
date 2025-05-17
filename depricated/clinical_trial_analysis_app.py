import streamlit as st
import sys
import os
import pandas as pd
import io
import warnings
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary modules from statistics_engine
from statistics_engine.analysis import AnalysisOrchestrator
from statistics_engine.data import get_canned_example, between_factors_dict, repeated_factors_dict
from statistics_engine.ui_components import display_progress, display_result_section, display_explanation, toggle_outputs
from statistics_engine.ui_explanations import get_explanation

# Run the main app (directly import Clinical_trial_analysis_app_v1.py content)
if __name__ == "__main__":
    # Set page config with wide layout
    st.set_page_config(page_title="Clinical Trial Analysis", layout="wide")
    
    # Executing the main app
    exec(open("statistics_engine/Clinical_trial_analysis_app_v1.py").read()) 