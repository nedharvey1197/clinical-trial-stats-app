import streamlit as st
import pandas as pd
import numpy as np
from clinical_trial_analysis_app_V0 import ClinicalTrialAnalysis

# Set page config with wide layout
st.set_page_config(page_title="Clinical Trial Analysis", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .explanation-text {
        font-size: 1.2em;
        line-height: 1.6;
        padding: 1em;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin: 1em 0;
    }
    .section-divider {
        border-top: 2px solid #e0e0e0;
        margin: 2em 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Clinical Trial Statistical Analysis")

# Create two main columns with different widths
analysis_col, explanation_col = st.columns([1, 2])  # Give more space to explanations

with analysis_col:
    st.header("Analysis Workflow")
    # File uploader
    uploaded_file = st.file_uploader("Upload your clinical trial data (CSV)", type=['csv'])

    if uploaded_file is not None:
        # Read the data
        data = pd.read_csv(uploaded_file)
        st.write("Preview of your data:")
        st.dataframe(data.head())

        # Initialize the analysis class
        analysis = ClinicalTrialAnalysis(data)

        # Sidebar for analysis options
        st.sidebar.header("Analysis Options")

        # Select outcome variable
        outcome = st.sidebar.selectbox("Select Outcome Variable", data.columns)

        # Select model type
        model_type = st.sidebar.selectbox(
            "Select Analysis Type",
            ["T-test", "One-way ANOVA", "Two-way ANOVA", "Three-way ANOVA",
             "One-way Repeated Measures ANOVA", "Two-way Repeated Measures ANOVA",
             "Three-way Repeated Measures ANOVA", "Mixed ANOVA (One Between, One Repeated)",
             "Mixed ANOVA (Two Between, One Repeated)", "Mixed ANOVA (One Between, Two Repeated)",
             "Complex Mixed ANOVA"]
        )

        # Select between-subjects factors
        between_factors = st.sidebar.multiselect(
            "Select Between-Subjects Factors",
            data.columns,
            max_selections=3
        )

        # Select repeated measures factors
        repeated_factors = st.sidebar.multiselect(
            "Select Repeated Measures Factors",
            data.columns,
            max_selections=2
        )

        # Run analysis button
        if st.sidebar.button("Run Analysis"):
            try:
                # Run the analysis
                results = analysis.run_analysis(model_type, outcome, between_factors, repeated_factors)

                # Display results
                st.header("Analysis Results")

                # Display assumptions
                st.subheader("Statistical Assumptions")
                assumptions = results['Assumptions']
                st.write("Shapiro-Wilk Test for Normality:", assumptions['Shapiro-Wilk'])
                if 'Levene' in assumptions:
                    st.write("Levene's Test for Equal Variances:", assumptions['Levene'])
                if 'Mauchly-Sphericity' in assumptions:
                    st.write("Mauchly's Test for Sphericity:", assumptions['Mauchly-Sphericity'])

                # Display descriptive statistics
                st.subheader("Descriptive Statistics")
                st.dataframe(results['Descriptive Stats'])

                # Display LS Means
                st.subheader("Least Squares Means")
                st.dataframe(results['LS Means'])

                # Display plots
                st.subheader("Plots")
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(results['Plot'])
                with col2:
                    st.pyplot(results['LS Means Plot'])

                # Display statistical test results
                st.subheader("Statistical Test Results")
                if 'T-test' in results:
                    st.write("T-test Results:", results['T-test'])
                elif 'ANOVA' in results:
                    st.write("ANOVA Results:")
                    st.dataframe(results['ANOVA'])
                    if 'Post-Hoc' in results:
                        st.write("Post-Hoc Analysis:")
                        st.dataframe(results['Post-Hoc'].summary())
                elif 'Repeated Measures ANOVA' in results:
                    st.write("Repeated Measures ANOVA Results:")
                    st.write(results['Repeated Measures ANOVA']['Summary'])
                elif 'Mixed ANOVA' in results:
                    st.write("Mixed ANOVA Results:")
                    st.write(results['Mixed ANOVA']['Summary'])
                elif 'Complex Mixed ANOVA' in results:
                    st.write("Complex Mixed ANOVA Results:")
                    st.write(results['Complex Mixed ANOVA']['Summary'])

                # Display alternative test results if assumptions were violated
                if 'Alternative Test' in results:
                    st.subheader("Alternative Test Results (Assumptions Violated)")
                    st.write(results['Alternative Test'])

            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
    else:
        st.info("Please upload a CSV file to begin analysis.")

with explanation_col:
    st.header("Analysis Explanation")

    # Introduction
    st.markdown('<div class="explanation-text">', unsafe_allow_html=True)
    st.markdown("""
    This tool helps you analyze clinical trial data using various statistical methods.
    The analysis workflow is designed to guide you through the process step by step.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Data Upload Explanation
    st.markdown('<div class="explanation-text">', unsafe_allow_html=True)
    st.markdown("""
    ### Data Upload
    Start by uploading your CSV file containing the clinical trial data.
    The file should include:
    - A 'Subject' column for repeated measures analyses
    - Columns for your outcome variable
    - Columns for your factors (categorical variables)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Analysis Type Explanation
    st.markdown('<div class="explanation-text">', unsafe_allow_html=True)
    st.markdown("""
    ### Analysis Type
    Choose the appropriate statistical test based on your study design:
    - **T-test**: Compare means between two groups
    - **ANOVA**: Compare means across multiple groups
    - **Repeated Measures**: Analyze data collected over time
    - **Mixed ANOVA**: Combine between-subjects and repeated measures
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Results Interpretation
    st.markdown('<div class="explanation-text">', unsafe_allow_html=True)
    st.markdown("""
    ### Results Interpretation
    The analysis provides:
    1. **Assumptions Check**: Tests for normality, equal variances, and sphericity
    2. **Descriptive Statistics**: Summary of your data
    3. **Statistical Tests**: Results of your chosen analysis
    4. **Visualizations**: Plots to help understand the data
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Additional Notes
    st.markdown('<div class="explanation-text">', unsafe_allow_html=True)
    st.markdown("""
    ### Additional Notes
    - If assumptions are violated, alternative tests will be suggested
    - Post-hoc analyses are provided for ANOVA tests
    - Confidence intervals are shown in the plots
    """)
    st.markdown('</div>', unsafe_allow_html=True)
