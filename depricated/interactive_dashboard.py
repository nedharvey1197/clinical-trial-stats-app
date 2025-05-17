import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import warnings
import plotly.express as px
import json
from statistics_engine.visualization import AdvancedVisualization
from statistics_engine.base_models import ClinicalTrialAnalysis
from statistics_engine.enhanced_models import EnhancedClinicalTrialAnalysis

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
from statistics_engine.data import get_canned_example, between_factors_dict, repeated_factors_dict


def main():
    """Main function to run the dashboard app."""
    st.markdown("<h1 class='dash-title'>Interactive Clinical Trial Data Explorer</h1>", unsafe_allow_html=True)
    
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
            example = get_canned_example(model_type)
            data = example["data"]
            description = example["description"]
            
            # Display brief sample of the data
            st.write("Sample data:")
            st.dataframe(data.head(3))
            
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
                    return
            else:
                # If no data uploaded, use default example
                st.info("Please upload data or select a canned example.")
                if 'data' not in locals():
                    model_type = "T-test"  # Default
                    example = get_canned_example(model_type)
                    data = example["data"]
                    description = example["description"]

        # Visualization controls
        st.title("Visualization Options")
        
        # Get column names for selection
        if 'data' in locals():
            columns = data.columns.tolist()
            
            outcome_var = st.selectbox(
                "Select Outcome Variable",
                [col for col in columns if col not in ['Subject']],
                index=columns.index("Outcome") if "Outcome" in columns else 0
            )
            
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
    
    # Main content area
    if 'data' in locals() and visualize_clicked:
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
                    factors = [primary_factor]
                    if secondary_factor != "None":
                        factors.append(secondary_factor)
                    fig = viz.create_interactive_boxplot(
                        outcome_var,
                        factors,
                        secondary_factor if secondary_factor != "None" else None
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Box plot requires at least one grouping factor.")
                    
        if "Distribution Plot" in plot_types:
            with col2:
                st.markdown("### Distribution Plot")
                fig = viz.create_distribution_plot(
                    outcome_var,
                    primary_factor if primary_factor != "None" else None
                )
                st.plotly_chart(fig, use_container_width=True)
                
        if "Q-Q Plot" in plot_types:
            with col1:
                st.markdown("### Q-Q Plot (Normality Check)")
                fig = viz.create_qq_plot(
                    outcome_var,
                    primary_factor if primary_factor != "None" else None
                )
                st.plotly_chart(fig, use_container_width=True)
                
        if "Interaction Plot" in plot_types and primary_factor != "None" and secondary_factor != "None":
            with col2:
                st.markdown("### Interaction Plot")
                fig = viz.create_interactive_interaction_plot(
                    outcome_var,
                    primary_factor,
                    secondary_factor
                )
                st.plotly_chart(fig, use_container_width=True)
                
        if "Statistical Summary" in plot_types:
            st.markdown("### Statistical Analysis")
            
            # Run the analysis using the enhanced model if requested
            if show_advanced and mcid_value > 0:
                analyzer = EnhancedClinicalTrialAnalysis(data, mcid=mcid_value)
            else:
                analyzer = ClinicalTrialAnalysis(data)
            
            # Determine factors for analysis
            between_factors = []
            repeated_factors = []
            
            if primary_factor != "None":
                between_factors.append(primary_factor)
            if secondary_factor != "None":
                # Simplified approach - assume second factor is between
                between_factors.append(secondary_factor)
            
            # For repeated measures, would need more logic here
            if 'Time' in data.columns:
                repeated_factors.append('Time')
                
            # Capture warnings during analysis
            warning_buffer = io.StringIO()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # Run appropriate analysis based on model type and factors
                if len(between_factors) == 1 and len(repeated_factors) == 0:
                    if len(data[between_factors[0]].unique()) == 2:
                        # Two groups - run t-test
                        run_model = "T-test"
                    else:
                        # More than two groups - run one-way ANOVA
                        run_model = "One-way ANOVA"
                elif len(between_factors) == 2 and len(repeated_factors) == 0:
                    run_model = "Two-way ANOVA"
                elif len(between_factors) == 1 and len(repeated_factors) == 1:
                    run_model = "Mixed ANOVA (One Between, One Repeated)"
                else:
                    # For demo, default to model_type from data source
                    run_model = model_type
                
                try:
                    results = analyzer.run_analysis(run_model, outcome_var, between_factors, repeated_factors)
                    
                    # Process warnings
                    for warn in w:
                        if not issubclass(warn.category, (DeprecationWarning, PendingDeprecationWarning,)):
                            warning_buffer.write(f"\n{warn.category.__name__}: {warn.message}\n\n")
                            
                    # Display results
                    st.write(f"Analysis performed: **{run_model}**")
                    
                    # Show statistical test results
                    if run_model == "T-test" and 'T-test' in results:
                        st.write(f"T-value: {results['T-test']['t_stat']:.4f}, p-value: {results['T-test']['p_value']:.4f}")
                        if results['T-test']['p_value'] < (alpha_level if show_advanced else 0.05):
                            st.success("Statistically significant difference found.")
                        else:
                            st.info("No statistically significant difference found.")
                    elif "ANOVA" in results:
                        st.subheader("ANOVA Results")
                        st.dataframe(results["ANOVA"])
                        
                    # Show effect sizes if available (from enhanced model)
                    if 'Effect Sizes' in results:
                        st.subheader("Effect Sizes")
                        for key, value in results['Effect Sizes'].items():
                            st.write(f"**{key}:** {value}")
                            
                    # Assumptions
                    st.subheader("Statistical Assumptions")
                    for key, value in results['Assumptions'].items():
                        if isinstance(value, tuple):
                            st.write(f"**{key}:** statistic = {value[0]:.4f}, p-value = {value[1]:.4f}")
                            if key == "Shapiro-Wilk" and value[1] < 0.05:
                                st.warning("Normality assumption may be violated. Consider non-parametric alternatives.")
                            elif key == "Levene" and value[1] < 0.05:
                                st.warning("Equal variance assumption may be violated. Consider robust methods.")
                        else:
                            st.write(f"**{key}:** {value}")
                            
                    # Display Descriptive Statistics
                    st.subheader("Descriptive Statistics")
                    st.dataframe(results["Descriptive Stats"])
                    
                    # Display LS Means
                    st.subheader("LS Means")
                    st.dataframe(results["LS Means"])
                    
                    # Display LS Means Plot
                    st.pyplot(results["LS Means Plot"])
                    
                    # Display warnings
                    warnings_text = warning_buffer.getvalue()
                    if warnings_text:
                        st.subheader("Analysis Warnings")
                        st.markdown(
                            f'<div style="background:#fff3cd; color:#856404; border:1px solid #ffeeba; '
                            f'border-radius:8px; padding:1em; margin:1em 0;">'
                            f'<strong>Statistical Warnings:</strong><br><pre style="white-space:pre-wrap;">{warnings_text}\n</pre>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.markdown(
                        f'<div style="background:#f8d7da; color:#721c24; border:1px solid #f5c6cb; '
                        f'border-radius:8px; padding:1em; margin:1em 0;">'
                        f'<strong>Analysis Error:</strong><br>{str(e)}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        
        # Download options
        st.markdown("### Download Options")
        
        # Create CSV of the data
        csv_data = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv_data,
            file_name=f"{model_type.replace(' ', '_')}_data.csv",
            mime="text/csv",
        )
        
        # Create a JSON of the visualization settings
        viz_settings = {
            "model_type": model_type,
            "outcome_variable": outcome_var,
            "primary_factor": primary_factor,
            "secondary_factor": secondary_factor,
            "plot_types": plot_types,
            "show_advanced": show_advanced
        }
        
        if show_advanced:
            viz_settings["alpha_level"] = alpha_level
            viz_settings["mcid_value"] = mcid_value
            
        viz_json = json.dumps(viz_settings, indent=2)
        st.download_button(
            label="Download Visualization Settings (JSON)",
            data=viz_json,
            file_name=f"{model_type.replace(' ', '_')}_viz_settings.json",
            mime="application/json",
        )

# Run the app
if __name__ == "__main__":
    main() 