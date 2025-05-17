import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import warnings
import plotly.express as px
import json
from visualization import AdvancedVisualization
from base_models import ClinicalTrialAnalysis
from enhanced_models import EnhancedClinicalTrialAnalysis

# Configure page
st.set_page_config(page_title="Clinical Trial Interactive Dashboard", layout="wide")

# Add custom CSS
st.markdown("""
    <style>
    .stButton > button, .stDownloadButton > button {
        background-color: #1976d2;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5em 1.5em;
        font-size: 1.1em;
        font-weight: bold;
        margin: 0.5em 0;
        transition: background 0.2s;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #1565c0;
        color: #fff;
    }
    .insight-box {
        background-color: #f8f9fa;
        border-left: 3px solid #1976d2;
        padding: 1em;
        margin: 1em 0;
        border-radius: 0 5px 5px 0;
    }
    .dash-title {
        color: #1976d2;
        text-align: center;
        font-size: 2em;
        margin-bottom: 1em;
    }
    </style>
""", unsafe_allow_html=True)

# Import data from the statistics engine
from data import get_canned_example, between_factors_dict, repeated_factors_dict


def main():
    """Main function to run the dashboard app."""
    st.markdown("<h1 class='dash-title'>Interactive Clinical Trial Data Explorer</h1>", unsafe_allow_html=True)
    
    # Initialize data variable to ensure it exists
    data = None
    model_type = "T-test"  # Default
    description = ""
    
    # Sidebar for data selection and controls
    with st.sidebar:
        st.title("Data Selection")
        
        # Data source options
        data_source = st.radio(
            "Choose Data Source",
            ["Use Canned Example", "Upload Your Data"]
        )
        
        if data_source == "Use Canned Example":
            model_type = st.selectbox(
                "Select Example Dataset",
                list(between_factors_dict.keys())
            )
            
            # Get data and description from canned examples
            try:
                example = get_canned_example(model_type)
                data = example["data"]
                description = example["description"]
                
                # Display brief sample of the data
                st.write("Sample data:")
                st.dataframe(data.head(3))
            except Exception as e:
                st.error(f"Error loading example data: {str(e)}")
                # Fallback to default data
                data = pd.DataFrame({
                    'Subject': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'Drug': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
                    'Outcome': [120, 115, 110, 105, 100, 130, 125, 120, 115, 110]
                })
                description = "Default example data"
            
        else:  # Upload data
            st.write("Upload a CSV file with your clinical trial data")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    st.success("Data loaded successfully!")
                    
                    # Display brief sample
                    st.write("Sample data:")
                    st.dataframe(data.head(3))
                    
                    # Ask for description
                    description = st.text_area(
                        "Enter a brief description of your data (optional)",
                        "Custom uploaded clinical trial data."
                    )
                    model_type = "Custom Data"
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
                    # Fallback to default data
                    data = pd.DataFrame({
                        'Subject': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'Drug': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
                        'Outcome': [120, 115, 110, 105, 100, 130, 125, 120, 115, 110]
                    })
                    description = "Default example data"
            else:
                # If no data uploaded, use default example
                st.info("Please upload data or select a canned example.")
                try:
                    example = get_canned_example("T-test")  # Default
                    data = example["data"]
                    description = example["description"]
                except Exception as e:
                    # Fallback to default data
                    data = pd.DataFrame({
                        'Subject': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        'Drug': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
                        'Outcome': [120, 115, 110, 105, 100, 130, 125, 120, 115, 110]
                    })
                    description = "Default example data"

        # Visualization controls
        st.title("Visualization Options")
        
        # Get column names for selection
        # Ensure data is defined before using it
        if data is not None:
            columns = data.columns.tolist()
            
            # Default to 'Outcome' if it exists, otherwise use first column that's not 'Subject'
            default_outcome = "Outcome" if "Outcome" in columns else next((col for col in columns if col != "Subject"), columns[0])
            default_outcome_index = columns.index(default_outcome) if default_outcome in columns else 0
            
            outcome_options = [col for col in columns if col != 'Subject']
            if not outcome_options:
                outcome_options = ["No outcome variable available"]
                
            outcome_var = st.selectbox(
                "Select Outcome Variable",
                outcome_options,
                index=min(default_outcome_index, len(outcome_options)-1)
            )
            
            # Create factor options list
            factor_options = [col for col in columns if col not in [outcome_var, 'Subject']]
            
            if factor_options:
                primary_factor = st.selectbox(
                    "Select Primary Grouping Factor",
                    factor_options,
                    index=0
                )
                
                secondary_factor_options = [col for col in factor_options if col != primary_factor]
                if secondary_factor_options:
                    secondary_factor = st.selectbox(
                        "Select Secondary Factor (optional)",
                        ["None"] + secondary_factor_options
                    )
                else:
                    secondary_factor = "None"
            else:
                primary_factor = "None"
                secondary_factor = "None"
                
            # Plot types
            plot_types = st.multiselect(
                "Select Visualizations to Display",
                ["Box Plot", "Distribution Plot", "Q-Q Plot", "Interaction Plot", "Statistical Summary"],
                default=["Box Plot", "Distribution Plot"]
            )
            
            # Create Viz button
            visualize_clicked = st.button("Generate Visualizations")
            
            # Statistical parameters
            st.title("Statistical Options")
            
            show_advanced = st.checkbox("Show Advanced Statistics")
            
            if show_advanced:
                alpha_level = st.slider(
                    "Alpha Level (Significance)",
                    min_value=0.01,
                    max_value=0.10,
                    value=0.05,
                    step=0.01,
                    format="%.2f"
                )
                
                mcid_value = st.number_input(
                    "Minimal Clinically Important Difference (MCID)",
                    min_value=0.0,
                    value=0.0,
                    step=0.1
                )
        else:
            st.error("No data available. Please upload a file or select a canned example.")
            visualize_clicked = False
    
    # Main content area
    if data is not None and visualize_clicked:
        try:
            # Setup the visualization module
            viz = AdvancedVisualization(data)
            
            # Determine layout based on number of plots
            col1, col2 = st.columns(2)
            
            # Title and description
            st.markdown(f"## Analysis of {model_type} Data")
            st.markdown(f"<div class='insight-box'>{description}</div>", unsafe_allow_html=True)
            
            # Data summary
            st.markdown("### Dataset Summary")
            st.write(f"Number of observations: {len(data)}")
            
            # Create viz based on selections
            if "Box Plot" in plot_types:
                with col1:
                    st.markdown("### Box Plot")
                    if primary_factor != "None":
                        try:
                            fig = viz.create_interactive_boxplot(outcome_var, [primary_factor], secondary_factor if secondary_factor != "None" else None)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add statistical insights
                            if show_advanced and primary_factor != "None":
                                st.markdown("#### Statistical Insights")
                                group_stats = data.groupby(primary_factor)[outcome_var].agg(['mean', 'std', 'count']).reset_index()
                                st.dataframe(group_stats.round(2))
                        except Exception as e:
                            st.error(f"Error creating box plot: {str(e)}")
                    else:
                        st.info("Box plot requires at least one grouping factor")
            
            if "Distribution Plot" in plot_types:
                with col2:
                    st.markdown("### Distribution Plot")
                    if primary_factor != "None":
                        try:
                            fig = viz.create_distribution_plot(outcome_var, primary_factor)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add statistical insights
                            if show_advanced:
                                st.markdown("#### Normality Test")
                                normality_results = viz.test_normality(outcome_var, primary_factor)
                                st.dataframe(normality_results)
                        except Exception as e:
                            st.error(f"Error creating distribution plot: {str(e)}")
                    else:
                        try:
                            fig = viz.create_distribution_plot(outcome_var)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating distribution plot: {str(e)}")
            
            if "Q-Q Plot" in plot_types:
                with col1:
                    st.markdown("### Q-Q Plot")
                    if primary_factor != "None":
                        figs = viz.create_qq_plot(outcome_var, primary_factor)
                        for i, (group, fig) in enumerate(figs.items()):
                            st.subheader(f"Q-Q Plot for {group}")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = viz.create_qq_plot(outcome_var)
                        st.plotly_chart(fig, use_container_width=True)
            
            if "Interaction Plot" in plot_types and secondary_factor != "None":
                with col2:
                    st.markdown("### Interaction Plot")
                    try:
                        fig = viz.create_interactive_interaction_plot(outcome_var, primary_factor, secondary_factor)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        if show_advanced:
                            st.markdown("#### Interaction Effect")
                            interaction_results = viz.test_interaction(outcome_var, primary_factor, secondary_factor)
                            st.write(f"Interaction F-value: {interaction_results['F']:.2f}, p-value: {interaction_results['p']:.4f}")
                            if interaction_results['p'] < alpha_level:
                                st.markdown("**Significant interaction detected**")
                            else:
                                st.markdown("No significant interaction detected")
                    except Exception as e:
                        st.error(f"Error creating interaction plot: {str(e)}")
            
            if "Statistical Summary" in plot_types:
                st.markdown("### Statistical Summary")
                if primary_factor != "None":
                    analysis = EnhancedClinicalTrialAnalysis(data, mcid_value if mcid_value > 0 else None)
                    
                    if secondary_factor != "None":
                        model_type = "Two-way ANOVA"
                        results = analysis.run_analysis(model_type, outcome_var, [primary_factor, secondary_factor])
                    else:
                        if len(data[primary_factor].unique()) == 2:
                            model_type = "T-test"
                        else:
                            model_type = "One-way ANOVA"
                        results = analysis.run_analysis(model_type, outcome_var, [primary_factor])
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Main Effects")
                        if "ANOVA" in results:
                            st.dataframe(results["ANOVA"])
                        elif "T-test" in results:
                            st.write(f"T-statistic: {results['T-test']['t_stat']:.2f}")
                            st.write(f"P-value: {results['T-test']['p_value']:.4f}")
                            sig = results['T-test']['p_value'] < alpha_level
                            st.markdown(f"**Significant difference: {sig}**")
                    
                    with col2:
                        st.markdown("#### Effect Sizes")
                        if "Effect Sizes" in results:
                            for k, v in results["Effect Sizes"].items():
                                if isinstance(v, float):
                                    st.write(f"{k}: {v:.4f}")
                                else:
                                    st.write(f"{k}: {v}")
                            
                            if "Clinical Significance" in results["Effect Sizes"]:
                                is_clinical = "significant" in results["Effect Sizes"]["Clinical Significance"].lower()
                                if is_clinical:
                                    st.markdown("✅ **Clinically Significant**")
                                else:
                                    st.markdown("❌ **Not Clinically Significant**")
                else:
                    st.info("Statistical analysis requires at least one grouping factor")
            
            # Add download capability
            if data is not None:
                # Save settings as JSON
                settings = {
                    "model_type": model_type,
                    "primary_factor": primary_factor,
                    "secondary_factor": secondary_factor,
                    "outcome_var": outcome_var,
                    "plot_types": plot_types,
                    "alpha_level": alpha_level if 'alpha_level' in locals() else 0.05,
                    "mcid_value": mcid_value if 'mcid_value' in locals() else 0.0
                }
                
                settings_json = json.dumps(settings, indent=2)
                
                st.markdown("### Export")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="Download Data (CSV)",
                        data=data.to_csv(index=False),
                        file_name=f"clinical_trial_data_{model_type.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    st.download_button(
                        label="Download Settings (JSON)",
                        data=settings_json,
                        file_name=f"visualization_settings_{model_type.replace(' ', '_')}.json",
                        mime="application/json"
                    )
        except Exception as e:
            st.error(f"An error occurred while generating visualizations: {str(e)}")
            st.info("Try selecting different variables or visualization types.")

if __name__ == "__main__":
    main() 