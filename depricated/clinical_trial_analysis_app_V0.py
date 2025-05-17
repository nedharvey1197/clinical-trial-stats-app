import pandas as pd
import streamlit as st
from models.base_models import ClinicalTrialAnalysis
import warnings
import io
warnings.simplefilter("always")


# Set page config with wide layout
st.set_page_config(page_title="Clinical Trial Analysis", layout="wide")
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
    </style>
""", unsafe_allow_html=True)

# Canned Example Datasets with descriptions
canned_examples = {
    "T-test": {
        "data": pd.DataFrame({
            'Subject': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Drug': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
            'Outcome': [120, 115, 110, 105, 100, 130, 125, 120, 115, 110]
        }),
        "description": "A Phase II randomized controlled trial (RCT) comparing the efficacy of a new antihypertensive drug (Drug A) versus placebo (Drug B) in patients with mild hypertension. The outcome is the reduction in systolic blood pressure (mmHg) after 8 weeks of treatment, with higher reductions indicating better efficacy. The trial aims to determine if Drug A significantly lowers blood pressure compared to placebo."
    },
    "One-way ANOVA": {
        "data": pd.DataFrame({
            'Subject': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            'Drug': ['Placebo', 'Placebo', 'Placebo', 'Placebo', 'Placebo',
                     'Low Dose', 'Low Dose', 'Low Dose', 'Low Dose', 'Low Dose',
                     'High Dose', 'High Dose', 'High Dose', 'High Dose', 'High Dose'],
            'Outcome': [120, 115, 110, 105, 100, 130, 125, 120, 115, 110, 140, 135, 130, 125, 120]
        }),
        "description": "A Phase II dose-ranging trial evaluating a new cholesterol-lowering drug in patients with hyperlipidemia. Patients are randomized to receive placebo, low dose, or high dose of the drug for 12 weeks. The outcome is the percentage reduction in LDL cholesterol levels (mg/dL), with larger reductions indicating better efficacy. The trial tests whether different doses lead to varying cholesterol reductions."
    },
    "Two-way ANOVA": {
        "data": pd.DataFrame({
            'Subject': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'Drug': ['A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'B'],
            'Age_Group': ['Young', 'Young', 'Young', 'Young', 'Young', 'Young',
                          'Old', 'Old', 'Old', 'Old', 'Old', 'Old'],
            'Outcome': [120, 115, 110, 130, 125, 120, 125, 120, 115, 135, 130, 125]
        }),
        "description": "A Phase III trial investigating a new anti-diabetic drug (Drug A vs. Drug B) in patients with type 2 diabetes, stratified by age (Young: <50 years, Old: ≥50 years). The outcome is the change in HbA1c levels (%) after 6 months, with lower values indicating better glycemic control. The trial assesses the main effects of drug type and age, as well as their interaction, on HbA1c reduction."
    },
    "Three-way ANOVA": {
        "data": pd.DataFrame({
            'Subject': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            'Drug': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
            'Age_Group': ['Young', 'Young', 'Old', 'Old', 'Young', 'Young', 'Old', 'Old', 'Young', 'Young', 'Old', 'Old', 'Young', 'Young', 'Old', 'Old'],
            'Site': ['1', '1', '1', '1', '2', '2', '2', '2', '1', '1', '1', '1', '2', '2', '2', '2'],
            'Outcome': [120, 122, 130, 128, 115, 117, 125, 123, 135, 133, 140, 138, 130, 128, 135, 133]
        }),
        "description": "A multicenter Phase III trial evaluating a new pain relief medication (Drug A vs. Drug B) in patients with osteoarthritis, across two age groups (Young: <50 years, Old: ≥50 years) and two clinical sites (Site 1, Site 2). The outcome is the reduction in pain score (VAS, 0-100 scale) after 4 weeks, with larger reductions indicating better pain relief. The trial examines the effects of drug, age, site, and their interactions on pain reduction."
    },
    "One-way Repeated Measures ANOVA": {
        "data": pd.DataFrame({
            'Subject': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'Time': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'Outcome': [120, 115, 110, 130, 125, 120, 140, 135, 130]
        }),
        "description": "A Phase II single-arm trial assessing the effect of a new anti-inflammatory drug on C-reactive protein (CRP) levels (mg/L) in patients with rheumatoid arthritis. Measurements are taken at baseline (Time 1), 4 weeks (Time 2), and 8 weeks (Time 3), with lower CRP levels indicating reduced inflammation. The trial evaluates the change in CRP over time within the same patients."
    },
    "Two-way Repeated Measures ANOVA": {
        "data": pd.DataFrame({
            'Subject': [1, 1, 1, 1, 2, 2, 2, 2],
            'Time': [1, 1, 2, 2, 1, 1, 2, 2],
            'Condition': ['Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard'],
            'Outcome': [120, 115, 110, 105, 130, 125, 120, 115]
        }),
        "description": "A Phase II crossover trial evaluating a cognitive enhancer in healthy volunteers under two task conditions (Easy, Hard). Cognitive performance (reaction time in milliseconds, lower is better) is measured at two time points (Time 1: before treatment, Time 2: after treatment) for each condition. The trial assesses the effects of time and task difficulty on cognitive performance within subjects."
    },
    "Three-way Repeated Measures ANOVA": {
        "data": pd.DataFrame({
            'Subject': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            'Drug': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A'],
            'Time': [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
            'Condition': ['Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard'],
            'Outcome': [120, 115, 110, 105, 130, 125, 120, 115, 125, 120, 115, 110]
        }),
        "description": "A Phase II trial comparing two anxiolytic drugs (Drug A, Drug B) in patients with generalized anxiety disorder (GAD). Anxiety levels (HAM-A score, lower is better) are measured under two stress conditions (Easy, Hard) at two time points (Time 1: baseline, Time 2: 6 weeks post-treatment). The trial examines the effects of drug, stress condition, and time on anxiety reduction."
    },
    "Mixed ANOVA (One Between, One Repeated)": {
        "data": pd.DataFrame({
            'Subject': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            'Drug': ['A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            'Outcome': [120, 115, 110, 130, 125, 120, 125, 120, 115, 135, 130, 125]
        }),
        "description": "A Phase III trial comparing a new antidepressant (Drug A) versus placebo (Drug B) in patients with major depressive disorder. Depression scores (MADRS, lower is better) are measured at baseline (Time 1), 4 weeks (Time 2), and 8 weeks (Time 3). The trial assesses the effect of the drug over time on depression severity."
    },
    "Mixed ANOVA (Two Between, One Repeated)": {
        "data": pd.DataFrame({
            'Subject': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            'Drug': ['A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'B'],
            'Age_Group': ['Young', 'Young', 'Young', 'Young', 'Young', 'Young', 'Old', 'Old', 'Old', 'Old', 'Old', 'Old'],
            'Time': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            'Outcome': [120, 115, 110, 130, 125, 120, 125, 120, 115, 135, 130, 125]
        }),
        "description": "A Phase III trial evaluating a new migraine treatment (Drug A vs. Drug B) in patients, stratified by age (Young: <50 years, Old: ≥50 years). Migraine frequency (number of attacks per month, lower is better) is measured at baseline (Time 1), 3 months (Time 2), and 6 months (Time 3). The trial investigates the effects of drug, age, and time on migraine frequency."
    },
    "Mixed ANOVA (One Between, Two Repeated)": {
        "data": pd.DataFrame({
            'Subject': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            'Drug': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A'],
            'Time': [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
            'Condition': ['Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard'],
            'Outcome': [120, 115, 110, 105, 130, 125, 120, 115, 125, 120, 115, 110]
        }),
        "description": "A Phase II trial comparing a new asthma medication (Drug A vs. Drug B) in patients with asthma. Lung function (FEV1, higher is better) is measured under two exercise conditions (Easy, Hard) at two time points (Time 1: baseline, Time 2: 4 weeks post-treatment). The trial assesses the effects of drug, exercise condition, and time on lung function."
    },
    "Complex Mixed ANOVA": {
        "data": pd.DataFrame({
            'Subject': [i+1 for i in range(32)],
            'Drug': sum([[d]*16 for d in ['A','B']],[]),
            'Age_Group': sum([[a]*8 for a in ['Young','Old']]*2,[]),
            'Time': sum([[t]*4 for t in [1,2]]*4,[]),
            'Condition': (['Easy','Easy','Hard','Hard']*8),
            'Outcome': [120, 122, 130, 128, 115, 117, 125, 123, 135, 133, 140, 138, 130, 128, 135, 133,
                        121, 123, 131, 129, 116, 118, 126, 124, 136, 134, 141, 139, 131, 129, 136, 134]
        }),
        "description": "A Phase III trial evaluating a new antipsychotic drug (Drug A vs. Drug B) in patients with schizophrenia, stratified by age (Young: <50 years, Old: ≥50 years). Symptom severity (PANSS score, lower is better) is measured under two social stress conditions (Easy, Hard) at two time points (Time 1: baseline, Time 2: 12 weeks post-treatment). The trial examines the effects of drug, age, stress condition, and time on symptom severity."
    }
}

# Determine factors based on the model
between_factors_dict = {
    "T-test": ["Drug"],
    "One-way ANOVA": ["Drug"],
    "Two-way ANOVA": ["Drug", "Age_Group"],
    "Three-way ANOVA": ["Drug", "Age_Group", "Site"],
    "One-way Repeated Measures ANOVA": [],
    "Two-way Repeated Measures ANOVA": [],
    "Three-way Repeated Measures ANOVA": ["Drug"],
    "Mixed ANOVA (One Between, One Repeated)": ["Drug"],
    "Mixed ANOVA (Two Between, One Repeated)": ["Drug", "Age_Group"],
    "Mixed ANOVA (One Between, Two Repeated)": ["Drug"],
    "Complex Mixed ANOVA": ["Drug", "Age_Group"]
}

repeated_factors_dict = {
    "T-test": [],
    "One-way ANOVA": [],
    "Two-way ANOVA": [],
    "Three-way ANOVA": [],
    "One-way Repeated Measures ANOVA": ["Time"],
    "Two-way Repeated Measures ANOVA": ["Time", "Condition"],
    "Three-way Repeated Measures ANOVA": ["Time", "Condition"],
    "Mixed ANOVA (One Between, One Repeated)": ["Time"],
    "Mixed ANOVA (Two Between, One Repeated)": ["Time"],
    "Mixed ANOVA (One Between, Two Repeated)": ["Time", "Condition"],
    "Complex Mixed ANOVA": ["Time", "Condition"]
}

# Inject custom CSS for explanation bubbles and font size
st.markdown(
    """
    <style>
    .explanation-bubble {
        background: #f5f7fa;
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 18px;
        font-size: 1.18rem;
        color: #222;
        border: 1px solid #e0e4ea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    .explanation-bubble strong {
        color: #1a4d8f;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar: App-Level Selections
with st.sidebar:
    st.title("Clinical Trial Analysis")
    st.write("Select a statistical model and input your data to analyze.")

    # Step 1: Model Selection
    st.header("Step 1: Select Model")
    models = list(canned_examples.keys())
    model_type = st.selectbox("Choose a model to analyze your data:", ["Select a model..."] + models,
                              help="Select a statistical model to apply. Each model corresponds to a specific experimental design.")
    if model_type == "Select a model...":
        model_type = None
        between_factors = []
        repeated_factors = []
        data = None
    else:
        between_factors = between_factors_dict[model_type]
        repeated_factors = repeated_factors_dict[model_type]
        data = canned_examples[model_type]["data"]

    # Checkbox for automatically running the canned example
    auto_run = st.checkbox("Automatically run the canned example", value=True,
                           help="Check this to run the example immediately after selecting a model.")

    # Step 2: Data Input Method
    st.header("Step 2: Choose Data Input")
    data_option = st.radio("Data Input Method", ["Use Canned Example", "Upload CSV", "Manual Entry"],
                           help="Choose to run a preloaded example, upload your own data, or enter data manually.")

# Main Content and Explanation Columns
main_col, explain_col = st.columns([2, 1])

with main_col:
    st.title(f"Analysis Workflow" + (f": {model_type}" if model_type else ""))
    st.write("Follow the steps below to analyze your clinical trial data.")

    # Display clinical description for the selected model
    if model_type:
        st.markdown(f"<h4>Clinical Context:</h4>", unsafe_allow_html=True)
        st.markdown(f"<p>{canned_examples[model_type]['description']}</p>", unsafe_allow_html=True)
        st.markdown("---")

    # Step 3: Run Canned Example
    st.header("Step 3: Explore a Canned Example")
    if model_type:
        st.write(f"A preloaded example dataset is available for the {model_type} model.")
        st.write("**Example Dataset:**")
        st.dataframe(data)
    else:
        st.write("\n\n---\n\n")
        st.write("**No model selected. Please select a model to begin analysis.**")

with explain_col:
    st.header("Explanation")
    if model_type:
        st.markdown(
            f'<div class="explanation-bubble"><strong>What:</strong> Displaying a preloaded dataset for the selected model: <b>{model_type}</b>.</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="explanation-bubble"><strong>Why:</strong> This allows you to see the expected data format and explore the analysis without entering your own data. The dataset is tailored to the model\'s requirements (e.g., 2 groups for T-test, repeated measures for Mixed ANOVA).</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="explanation-bubble"><strong>Source:</strong> The data is predefined in the app, similar to how statisticians might use example datasets in SAS or SPSS to demonstrate analysis.</div>', unsafe_allow_html=True)
        st.divider()
    else:
        st.markdown('<div class="explanation-bubble">No model selected. Please select a model to see the explanation.</div>', unsafe_allow_html=True)

# Run Canned Example
results = None  # Ensure results is always defined and in scope for all code paths
if model_type:
    with main_col:
        if auto_run:
            analysis = ClinicalTrialAnalysis(data)
            warning_buffer = io.StringIO()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # Run your analysis
                results = analysis.run_analysis(model_type, "Outcome", between_factors, repeated_factors)
                # Collect warning messages
                for warn in w:
                    if not issubclass(warn.category, (DeprecationWarning, PendingDeprecationWarning,)):
                        warning_buffer.write(f"\n{warn.category.__name__}: {warn.message}\n\n")
            st.subheader("Canned Example Results")
        else:
            if st.button("Run Canned Example"):
                analysis = ClinicalTrialAnalysis(data)
                warning_buffer = io.StringIO()
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    results = analysis.run_analysis(model_type, "Outcome", between_factors, repeated_factors)
                    for warn in w:
                        # Only show warnings that are not Deprecation or PendingDeprecation
                        if not issubclass(warn.category, (DeprecationWarning, PendingDeprecationWarning)):
                            warning_buffer.write(f"\n{warn.category.__name__}: {warn.message}\n")
                st.subheader("Canned Example Results")

    with explain_col:
        if auto_run or st.session_state.get("ran_canned_example", False):
            st.markdown(
                '<div class="explanation-bubble"><strong>What:</strong> Running the statistical analysis on the canned example.</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="explanation-bubble"><strong>Why:</strong> This demonstrates how the selected model processes the data, providing a baseline for understanding the outputs (e.g., assumption checks, test results, plots).</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="explanation-bubble"><strong>Source:</strong> The analysis uses <code>statsmodels</code> and <code>scipy</code> in Python, equivalent to SAS PROC GLM or PROC MIXED for ANOVA and mixed models.</div>', unsafe_allow_html=True)
            st.divider()

# Display Canned Example Results
if results is not None:
    with main_col:
        # Assumption Checks
        st.write("**Assumption Checks:**")
        for key, value in results['Assumptions'].items():
            st.write(f"{key}: {value}")

        # Descriptive Statistics
        st.write("**Descriptive Statistics:**")
        st.dataframe(results['Descriptive Stats'])

        # LS Means
        st.write("**LS Means Table:**")
        st.dataframe(results['LS Means'])
        st.write("**LS Means Plot:**")
        st.pyplot(results['LS Means Plot'])

        # Expected Mean Squares (for ANOVA models)
        if results['Expected Mean Squares'].size > 0:
            st.write("**Expected Mean Squares:**")
            st.dataframe(results['Expected Mean Squares'])

        # Statistical Test Results
        st.write("**Statistical Test Results:**")
        if 'Alternative Test' in results:
            st.markdown(f"<span style='color:red; font-weight:bold;'>Alternative Test Used (Assumptions Failed): {results['Alternative Test']}</span>", unsafe_allow_html=True)
        elif model_type == "T-test":
            st.write(f"T-test: t = {results['T-test']['t_stat']:.2f}, p = {results['T-test']['p_value']:.4f}")
        elif "ANOVA" in results:
            st.write("ANOVA Table:")
            anova_table = results['ANOVA']
            # If it's a statsmodels table, use .summary(), else display as DataFrame
            if hasattr(anova_table, 'summary'):
                st.write(anova_table.summary())
            elif isinstance(anova_table, pd.DataFrame):
                st.dataframe(anova_table)
            else:
                try:
                    # Try to convert to DataFrame if it's a list of lists of Cell objects
                    import statsmodels.api as sm
                    if hasattr(anova_table, '_results_table'):
                        st.dataframe(pd.DataFrame(anova_table._results_table.data[1:], columns=anova_table._results_table.data[0]))
                    else:
                        st.write(anova_table)
                except Exception:
                    st.write(anova_table)
            if 'Post-Hoc' in results:
                st.write("Post-Hoc (Tukey HSD):")
                posthoc = results['Post-Hoc']
                if hasattr(posthoc, 'summary'):
                    st.text(posthoc.summary())
                elif hasattr(posthoc, '_results_table'):
                    st.dataframe(pd.DataFrame(posthoc._results_table.data[1:], columns=posthoc._results_table.data[0]))
                else:
                    st.write(posthoc)
        elif "Repeated Measures ANOVA" in results or "Mixed ANOVA" in results:
            st.write("Mixed Model Summary:")
            st.text(results.get('Repeated Measures ANOVA', results.get('Mixed ANOVA'))['Summary'])
            st.write("**Run Summary:**")
            run_summary = results.get('Repeated Measures ANOVA', results.get('Mixed ANOVA'))['Run Summary']
            for key, value in run_summary.items():
                st.write(f"{key}: {value}")
            st.write("**Variance Estimates:**")
            st.write(pd.DataFrame(results.get('Repeated Measures ANOVA', results.get('Mixed ANOVA'))['Variance Estimates'].items(),
                                 columns=['Metric', 'Value']))
        # Highlight sphericity or other warnings in assumptions
        if 'Assumptions' in results and 'Sphericity Note' in results['Assumptions']:
            st.markdown(f"<span style='color:orange; font-weight:bold;'>⚠️ {results['Assumptions']['Sphericity Note']}</span>", unsafe_allow_html=True)

    with explain_col:
        st.markdown(
            '<div class="explanation-bubble"><strong>What:</strong> Running the statistical test and displaying results.</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="explanation-bubble"><strong>Why:</strong> This tests the main hypotheses (e.g., differences between groups, effects over time). For ANOVA, an F-test is used; for T-test, a t-statistic. Mixed models provide detailed fit statistics (e.g., Log-Likelihood, AIC).</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="explanation-bubble"><strong>Source:</strong> Using <code>statsmodels</code> for ANOVA and mixed models, <code>scipy.stats</code> for alternative tests (e.g., Kruskal-Wallis). Equivalent to SAS PROC GLM or PROC MIXED.</div>', unsafe_allow_html=True)
        # General explanatory text for statistical warnings
        st.markdown("""
        <div style="background:#e9ecef; color:#333; border-radius:8px; padding:1em; margin:1em 0;">
        <strong>About Statistical Warnings</strong><br>
        During statistical analysis, you may see warnings related to model fitting, convergence, or assumption checks. These are a normal part of the process, especially with small or synthetic datasets.<br>
        <ul>
            <li>Some warnings indicate the model had difficulty fitting the data or that certain assumptions were not fully met.</li>
            <li>The app will automatically apply corrections when appropriate.</li>
            <li>If you see these warnings with real data, consider reviewing your data quality, sample size, or model complexity.</li>
        </ul>
        <b>These warnings are shown for transparency and to help you better understand the analysis process.</b>
        </div>
        """, unsafe_allow_html=True)

        # Display captured warnings (assuming you have warning_buffer as described earlier)
        warnings_text = warning_buffer.getvalue()
        if warnings_text:
            st.markdown(
                f'<div style="background:#fff3cd; color:#856404; border:1px solid #ffeeba; border-radius:8px; padding:1em; margin:1em 0;">'
                f'<strong>Statistical Warnings:</strong><br><pre style="white-space:pre-wrap;">{warnings_text}\n</pre>'
                f'</div>',
                unsafe_allow_html=True
            )
    with main_col:
        # Visualization
        st.write("**Box/Interaction Plot:**")
        st.pyplot(results['Plot'])

    with explain_col:
        st.markdown(
            '<div class="explanation-bubble"><strong>What:</strong> Generating a box plot (non-repeated) or interaction plot (repeated measures).</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="explanation-bubble"><strong>Why:</strong> Visualizations help interpret the data and results. Box plots show group distributions, while interaction plots show trends over time or across factors.</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="explanation-bubble"><strong>Source:</strong> Using <code>seaborn</code> and <code>matplotlib</code>, similar to SAS ODS Graphics output.</div>', unsafe_allow_html=True)

    with main_col:
    # Download analysis results and explanation as HTML
        if model_type and 'results' in locals() and results:
            import io
            from datetime import datetime
            html_content = io.StringIO()
            html_content.write(f"<h2>Analysis Workflow: {model_type}</h2>")
            html_content.write("<h3>Step 3: Explore a Canned Example</h3>")
            if data is not None:
                html_content.write(data.to_html(index=False))
            html_content.write("<h3>Explanation</h3>")
            html_content.write("<b>What:</b> Displaying a preloaded dataset for the selected model.<br>")
            html_content.write("<b>Why:</b> This allows you to see the expected data format and explore the analysis without entering your own data. The dataset is tailored to the model's requirements.<br>")
            html_content.write("<b>Source:</b> The data is predefined in the app.<hr>")
            html_content.write("<h3>Canned Example Results</h3>")
            for key, value in results['Assumptions'].items():
                html_content.write(f"<b>{key}:</b> {value}<br>")
            html_content.write(results['Descriptive Stats'].to_html(index=False))
            html_content.write(results['LS Means'].to_html(index=False))
            # Add more as needed
            st.download_button(
                label="Download Analysis & Explanation (HTML)",
                data=html_content.getvalue(),
                file_name=f"analysis_{model_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime='text/html',
            )

# Step 4: Custom Data Input
if 'results' in locals() and results:
    with main_col:
        st.header("Step 4: Analyze Your Own Data")
        st.write("Now that you've seen the example, you can input your own data to analyze.")

        # Download template for data upload
        st.subheader("Download Data Template")
        # Generate template DataFrame
        template_cols = ['Subject', 'Outcome'] + between_factors + repeated_factors
        template_df = pd.DataFrame(columns=template_cols)
        csv = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download {model_type if model_type else ''} Data Template (CSV)",
            data=csv,
            file_name=f"{model_type.replace(' ', '_') if model_type else 'template'}_template.csv",
            mime='text/csv',
        )

        data = None
        if data_option == "Upload CSV":
            st.write("Upload a CSV file with columns: Subject, Outcome, and relevant factors (e.g., Drug, Time).")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file:
                data = pd.read_csv(uploaded_file)
                st.write("Uploaded Data:")
                st.dataframe(data)
                # Prompt for factors based on model
                between_factors = st.text_input(f"Between-subjects factors for {model_type} (comma-separated, e.g., Drug,Age)", ",".join(between_factors)).split(",")
                between_factors = [f.strip() for f in between_factors if f.strip()]
                repeated_factors = st.text_input(f"Repeated-measures factors for {model_type} (comma-separated, e.g., Time)", ",".join(repeated_factors)).split(",")
                repeated_factors = [f.strip() for f in repeated_factors if f.strip()]

        elif data_option == "Manual Entry":
            st.write("Enter data for each group.")
            num_groups = st.number_input("Number of groups", min_value=1, value=2, step=1)
            between_factors = st.text_input(f"Between-subjects factors for {model_type} (comma-separated, e.g., Drug,Age)", ",".join(between_factors)).split(",")
            between_factors = [f.strip() for f in between_factors if f.strip()]
            repeated_factors = st.text_input(f"Repeated-measures factors for {model_type} (comma-separated, e.g., Time)", ",".join(repeated_factors)).split(",")
            repeated_factors = [f.strip() for f in repeated_factors if f.strip()]

            data_dict = {'Subject': [], 'Outcome': []}
            for factor in between_factors + repeated_factors:
                data_dict[factor] = []

            subject_id = 1
            for group in range(num_groups):
                group_name = st.text_input(f"Name of group {group+1} (e.g., DrugA)", f"Group{group+1}")
                num_subjects = st.number_input(f"Number of subjects in {group_name}", min_value=1, value=5, step=1, key=f"subjects_{group}")
                for s in range(num_subjects):
                    for t in range(len(repeated_factors) if repeated_factors else 1):
                        outcome = st.number_input(
                            f"Outcome for subject {s+1} in {group_name}" + (f" at {repeated_factors[0]} {t+1}" if repeated_factors else ""),
                            value=100.0, step=1.0, key=f"outcome_{group}_{s}_{t}"
                        )
                        data_dict['Subject'].append(subject_id)
                        data_dict['Outcome'].append(outcome)
                        for factor in between_factors:
                            if factor == between_factors[0]:
                                data_dict[factor].append(group_name)
                            else:
                                data_dict[factor].append(
                                    st.text_input(f"{factor} for subject {s+1} in {group_name}", "Value", key=f"{factor}_{group}_{s}")
                                )
                        for i, factor in enumerate(repeated_factors):
                            data_dict[factor].append(t + 1)
                    subject_id += 1

            if st.button("Submit Data"):
                data = pd.DataFrame(data_dict)
                st.write("Entered Data:")
                st.dataframe(data)

    with explain_col:
        st.write("**What:** Allowing you to input your own data via CSV upload or manual entry.")
        st.write("**Why:** This lets you analyze custom datasets, ensuring the tool is flexible for real-world use. CSV upload is efficient for large datasets, while manual entry suits smaller studies.")
        st.write("**Source:** The app uses `pandas` to handle data input, similar to how SAS imports data with PROC IMPORT or DATA steps.")

# Run Custom Analysis
if data is not None:
    if st.button("Run Analysis with Custom Data"):
        try:
            analysis = ClinicalTrialAnalysis(data)
            warning_buffer = io.StringIO()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                results = analysis.run_analysis(model_type, "Outcome", between_factors, repeated_factors)
                for warn in w:
                    # Only show warnings that are not Deprecation or PendingDeprecation
                    if not issubclass(warn.category, (DeprecationWarning, PendingDeprecationWarning,)):
                        warning_buffer.write(f"\n{warn.category.__name__}: {warn.message}\n\n")

            st.subheader("Custom Data Results")
            # Assumption Checks
            st.write("**Assumption Checks:**")
            for key, value in results['Assumptions'].items():
                st.write(f"{key}: {value}")

            # Descriptive Statistics
            st.write("**Descriptive Statistics:**")
            st.dataframe(results['Descriptive Stats'])

            # LS Means
            st.write("**LS Means Table:**")
            st.dataframe(results['LS Means'])
            st.write("**LS Means Plot:**")
            st.pyplot(results['LS Means Plot'])

            # Expected Mean Squares
            if results['Expected Mean Squares'].size > 0:
                st.write("**Expected Mean Squares:**")
                st.dataframe(results['Expected Mean Squares'])

            # Statistical Test Results
            st.write("**Statistical Test Results:**")
            if 'Alternative Test' in results:
                st.markdown(f"<span style='color:red; font-weight:bold;'>Alternative Test Used (Assumptions Failed): {results['Alternative Test']}</span>", unsafe_allow_html=True)
            elif model_type == "T-test":
                st.write(f"T-test: t = {results['T-test']['t_stat']:.2f}, p = {results['T-test']['p_value']:.4f}")
            elif "ANOVA" in results:
                st.write("ANOVA Table:")
                st.write(results['ANOVA'])
                if 'Post-Hoc' in results:
                    st.write("Post-Hoc (Tukey HSD):")
                    posthoc = results['Post-Hoc']
                    if hasattr(posthoc, 'summary'):
                        st.write(posthoc.summary())
                    elif hasattr(posthoc, '_results_table'):
                        st.dataframe(pd.DataFrame(posthoc._results_table.data[1:], columns=posthoc._results_table.data[0]))
                    else:
                        st.write(posthoc)
            elif "Repeated Measures ANOVA" in results or "Mixed ANOVA" in results:
                st.write("Mixed Model Summary:")
                st.text(results.get('Repeated Measures ANOVA', results.get('Mixed ANOVA'))['Summary'])
                st.write("**Run Summary:**")
                run_summary = results.get('Repeated Measures ANOVA', results.get('Mixed ANOVA'))['Run Summary']
                for key, value in run_summary.items():
                    st.write(f"{key}: {value}")
                st.write("**Variance Estimates:**")
                st.write(pd.DataFrame(results.get('Repeated Measures ANOVA', results.get('Mixed ANOVA'))['Variance Estimates'].items(),
                                     columns=['Metric', 'Value']))

            # Visualization
            st.write("**Box/Interaction Plot:**")
            st.pyplot(results['Plot'])

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}. Please check your data and factors.")

with explain_col:
    if st.session_state.get("ran_custom_analysis", False):
        st.write("**What:** Running the statistical analysis on your custom data.")
        st.write("**Why:** This applies the selected model to your data, producing the same detailed outputs as the canned example for direct comparison.")
        st.write("**Source:** The analysis uses `statsmodels` and `scipy`, equivalent to SAS PROC GLM or PROC MIXED.")
        # General explanatory text for statistical warnings
        st.markdown("""
        <div style="background:#e9ecef; color:#333; border-radius:8px; padding:1em; margin:1em 0;">
        <strong>About Statistical Warnings</strong><br>
        During statistical analysis, you may see warnings related to model fitting, convergence, or assumption checks. These are a normal part of the process, especially with small or synthetic datasets.<br>
        <ul>
            <li>Some warnings indicate the model had difficulty fitting the data or that certain assumptions were not fully met.</li>
            <li>The app will automatically apply corrections when appropriate.</li>
            <li>If you see these warnings with real data, consider reviewing your data quality, sample size, or model complexity.</li>
        </ul>
        <b>These warnings are shown for transparency and to help you better understand the analysis process.</b>
        </div>
        """, unsafe_allow_html=True)

        # Display captured warnings (assuming you have warning_buffer as described earlier)
        warnings_text = warning_buffer.getvalue()
        if warnings_text:
            st.markdown(
                f'<div style="background:#fff3cd; color:#856404; border:1px solid #ffeeba; border-radius:8px; padding:1em; margin:1em 0;">'
                f'<strong>Statistical Warnings:</strong><br><pre style="white-space:pre-wrap;">{warnings_text}\n</pre>'
                f'</div>',
                unsafe_allow_html=True
            )
