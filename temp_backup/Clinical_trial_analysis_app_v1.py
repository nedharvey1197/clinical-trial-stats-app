import pandas as pd
import streamlit as st
from analysis import AnalysisOrchestrator
from data import get_canned_example, between_factors_dict, repeated_factors_dict
from ui_components import display_progress, display_result_section, display_explanation, toggle_outputs
from ui_explanations import get_explanation
import warnings
import io
from datetime import datetime

def main():
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
    """, unsafe_allow_html=True)

    # Sidebar: App-Level Selections
    with st.sidebar:
        # Add return to launcher button
        if st.button("← Return to App Launcher"):
            st.switch_page("app_launcher.py")
            
        st.title("Clinical Trial Analysis")
        st.write("Select a statistical model and input your data to analyze.")

        # Step 1: Model Selection
        st.header("Step 1: Select Model")
        display_progress(1, 3)
        models = list(between_factors_dict.keys())
        model_type = st.selectbox(
            "Choose a model to analyze your data:",
            ["Select a model..."] + models,
            help="Select a statistical model to apply. Each model corresponds to a specific experimental design."
        )
        if model_type == "Select a model...":
            model_type = None
            between_factors = []
            repeated_factors = []
            data = None
            description = None
        else:
            example = get_canned_example(model_type)
            data = example["data"]
            description = example["description"]
            between_factors = between_factors_dict[model_type]
            repeated_factors = repeated_factors_dict[model_type]

        # Checkbox for automatically running the canned example
        auto_run = st.checkbox(
            "Automatically run the canned example",
            value=True,
            help="Check this to run the example immediately after selecting a model."
        )

        # Analyze Your Own Data (Combines Step 2 and Step 4)
        st.header("Analyze Your Own Data (Optional)")
        # Step 2: Choose Data Input Method
        data_option = st.radio(
            "Data Input Method",
            ["Use Canned Example", "Upload CSV", "Manual Entry"],
            help="Choose to run a preloaded example, upload your own data, or enter data manually."
        )

        # Initialize data as None
        custom_data = None  # Renamed to avoid conflict with canned example data
        if data_option == "Upload CSV":
            st.write("Upload a CSV file with columns: Subject, Outcome, and relevant factors (e.g., Drug, Time).")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file:
                try:
                    custom_data = pd.read_csv(uploaded_file)
                    st.write("Uploaded Data:")
                    st.dataframe(custom_data)
                    between_factors = st.text_input(
                        f"Between-subjects factors for {model_type} (comma-separated, e.g., Drug,Age)",
                        ",".join(between_factors)
                    ).split(",")
                    between_factors = [f.strip() for f in between_factors if f.strip()]
                    repeated_factors = st.text_input(
                        f"Repeated-measures factors for {model_type} (comma-separated, e.g., Time)",
                        ",".join(repeated_factors)
                    ).split(",")
                    repeated_factors = [f.strip() for f in repeated_factors if f.strip()]
                except Exception as e:
                    st.error(f"Error loading CSV: {str(e)}")

        elif data_option == "Manual Entry":
            st.write("Enter data for each group.")
            num_groups = st.number_input("Number of groups", min_value=1, value=2, step=1)
            between_factors = st.text_input(
                f"Between-subjects factors for {model_type} (comma-separated, e.g., Drug,Age)",
                ",".join(between_factors)
            ).split(",")
            between_factors = [f.strip() for f in between_factors if f.strip()]
            repeated_factors = st.text_input(
                f"Repeated-measures factors for {model_type} (comma-separated, e.g., Time)",
                ",".join(repeated_factors)
            ).split(",")
            repeated_factors = [f.strip() for f in repeated_factors if f.strip()]

            data_dict = {'Subject': [], 'Outcome': []}
            for factor in between_factors + repeated_factors:
                data_dict[factor] = []

            subject_id = 1
            for group in range(num_groups):
                group_name = st.text_input(f"Name of group {group+1} (e.g., DrugA)", f"Group{group+1}")
                num_subjects = st.number_input(
                    f"Number of subjects in {group_name}",
                    min_value=1,
                    value=5,
                    step=1,
                    key=f"subjects_{group}"
                )
                for s in range(num_subjects):
                    for t in range(len(repeated_factors) if repeated_factors else 1):
                        outcome = st.number_input(
                            f"Outcome for subject {s+1} in {group_name}" + (f" at {repeated_factors[0]} {t+1}" if repeated_factors else ""),
                            value=100.0,
                            step=1.0,
                            key=f"outcome_{group}_{s}_{t}"
                        )
                        data_dict['Subject'].append(subject_id)
                        data_dict['Outcome'].append(outcome)
                        for factor in between_factors:
                            if factor == between_factors[0]:
                                data_dict[factor].append(group_name)
                            else:
                                data_dict[factor].append(
                                    st.text_input(
                                        f"{factor} for subject {s+1} in {group_name}",
                                        "Value",
                                        key=f"{factor}_{group}_{s}"
                                    )
                                )
                        for i, factor in enumerate(repeated_factors):
                            data_dict[factor].append(t + 1)
                    subject_id += 1

            if st.button("Submit Data"):
                try:
                    custom_data = pd.DataFrame(data_dict)
                    st.write("Entered Data:")
                    st.dataframe(custom_data)
                except Exception as e:
                    st.error(f"Error processing manual data: {str(e)}")

        # Analysis Settings Form (Moved from Step 4)
        with st.form(key="analysis_settings_form"):
            st.subheader("Run Your Analysis")
            imputation_method = st.selectbox(
                "Imputation Method for Missing Values",
                options=["mean", "median", "none"],
                help="Choose how to handle missing values in the Outcome column. 'none' will skip imputation."
            )
            submit_button = st.form_submit_button(
                label="Run Analysis with Custom Data",
                disabled=custom_data is None
            )

        if custom_data is None and data_option != "Use Canned Example":
            st.info("Please upload a CSV file or submit manual data to enable the analysis.")

        # Visibility Toggles for Custom Data Results
        st.subheader("Customize Your Results")
        st.write("Choose which results to display after running the analysis:")
        show_descriptive, show_ls_means, show_plot = toggle_outputs()

    # Main Content and Explanation Columns
    main_col, explain_col = st.columns([2, 1])

    with main_col:
        st.title(f"Analysis Workflow" + (f": {model_type}" if model_type else ""))
        st.write("Follow the steps below to analyze your clinical trial data.")

        # Step 3: Explore a Canned Example
        st.header("Step 3: Explore a Canned Example")
        display_progress(3, 4)
        if model_type:
            st.write(description)
            st.write("**Example Dataset:**")
            st.dataframe(data)
        else:
            st.write("\n\n---\n\n")
            st.write("**No model selected. Please select a model to begin analysis.**")

    with explain_col:
        st.header("Explanation")
        if model_type:
            display_explanation("Data Input", get_explanation("data_input", model_type, results=None, quality_report=None))
        else:
            st.markdown('<div class="explanation-bubble">No model selected. Please select a model to see the explanation.</div>', unsafe_allow_html=True)

    # Run Canned Example
    results = None
    quality_report = None
    if model_type:
        with main_col:
            orchestrator = AnalysisOrchestrator(model_type, "Outcome", between_factors, repeated_factors, mcid=5.0)
            if auto_run:
                warning_buffer = io.StringIO()
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    try:
                        results, quality_report = orchestrator.run_pipeline(data, imputation_method="mean")
                    except Exception as e:
                        st.error(f"Error running canned example: {str(e)}")
                    for warn in w:
                        if not issubclass(warn.category, (DeprecationWarning, PendingDeprecationWarning)):
                            warning_buffer.write(f"\n{warn.category.__name__}: {warn.message}\n\n")
                if results:
                    st.subheader("Canned Example Results")
            else:
                if st.button("Run Canned Example"):
                    warning_buffer = io.StringIO()
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        try:
                            results, quality_report = orchestrator.run_pipeline(data, imputation_method="mean")
                        except Exception as e:
                            st.error(f"Error running canned example: {str(e)}")
                        for warn in w:
                            if not issubclass(warn.category, (DeprecationWarning, PendingDeprecationWarning)):
                                warning_buffer.write(f"\n{warn.category.__name__}: {warn.message}\n\n")
                    if results:
                        st.subheader("Canned Example Results")

        with explain_col:
            if (auto_run or st.session_state.get("ran_canned_example", False)) and results:
                display_explanation("Analysis", get_explanation("results", model_type))

    # Display Canned Example Results
    if results is not None:
        with main_col:
            # Data Quality Report
            st.write("**Data Quality Report:**")
            for key, value in quality_report.items():
                st.write(f"{key}: {value}")

            # Assumption Checks
            display_result_section(
                "Assumption Checks",
                results['Assumptions'],
                "Download Assumptions (Text)",
                str(results['Assumptions']).encode('utf-8'),
                "text/plain",
                "assumptions"
            )

        with explain_col:
            display_explanation("Assumption Checks", get_explanation("assumption_checks", model_type, results, quality_report))

        with main_col:
            # Descriptive Statistics
            display_result_section(
                "Descriptive Statistics",
                results['Descriptive Stats'],
                "Download Descriptive Stats (CSV)",
                results['Exports']['Descriptive Stats CSV'],
                "text/csv",
                "descriptive_stats"
            )

            # LS Means
            display_result_section(
                "LS Means Table",
                results['LS Means'],
                "Download LS Means (CSV)",
                results['Exports']['LS Means CSV'],
                "text/csv",
                "ls_means"
            )
            display_result_section(
                "LS Means Plot",
                results['LS Means Plot'],
                "Download LS Means Plot (PNG)",
                results['Exports']['LS Means Plot PNG'],
                "image/png",
                "ls_means_plot"
            )

            # Expected Mean Squares
            if results['Expected Mean Squares'].size > 0:
                display_result_section(
                    "Expected Mean Squares",
                    results['Expected Mean Squares'],
                    "Download Expected Mean Squares (CSV)",
                    results['Exports']['Expected Mean Squares CSV'],
                    "text/csv",
                    "expected_mean_squares"
                )

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

            # Effect Sizes
            st.write("**Effect Sizes:**")
            st.write(results['Effect Sizes'])

            # Visualization
            display_result_section(
                "Box/Interaction Plot",
                results['Plot'],
                "Download Plot (PNG)",
                results['Exports']['Plot PNG'],
                "image/png",
                "plot"
            )

        with explain_col:
            display_explanation("Results", get_explanation("results", model_type, results, quality_report))
            display_explanation("Visualization", get_explanation("visualization", model_type, results, quality_report))
        
        with main_col:
            # Download full analysis as HTML
            html_content = io.StringIO()
            html_content.write(f"<h2>Analysis Workflow: {model_type}</h2>")
            html_content.write("<h3>Step 3: Explore a Canned Example</h3>")
            if data is not None:
                html_content.write(data.to_html(index=False))
            html_content.write("<h3>Canned Example Results</h3>")
            html_content.write("<h4>Data Quality Report</h4>")
            for key, value in quality_report.items():
                html_content.write(f"<p><b>{key}:</b> {value}</p>")
            html_content.write("<h4>Assumption Checks</h4>")
            for key, value in results['Assumptions'].items():
                html_content.write(f"<p><b>{key}:</b> {value}</p>")
            html_content.write("<h4>Descriptive Statistics</h4>")
            html_content.write(results['Descriptive Stats'].to_html(index=False))
            html_content.write("<h4>LS Means Table</h4>")
            html_content.write(results['LS Means'].to_html(index=False))
            st.download_button(
                label="Download Analysis & Explanation (HTML)",
                data=html_content.getvalue(),
                file_name=f"analysis_{model_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime='text/html',
            )

    if results is not None:
        with main_col:
            # New Section: Try Another Model or Experiment with Your Own Data
            st.header("Try Another Model or Experiment with Your Own Data")
            st.write("You can select a different model in the sidebar to try another analysis, or use the 'Analyze Your Own Data' section in the sidebar to input and analyze your own data.")

    # Run Custom Analysis
    if submit_button and custom_data is not None:
        with main_col:
            try:
                orchestrator = AnalysisOrchestrator(model_type, "Outcome", between_factors, repeated_factors, mcid=5.0)
                warning_buffer = io.StringIO()
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    custom_results, custom_quality_report = orchestrator.run_pipeline(custom_data, imputation_method=imputation_method)
                    for warn in w:
                        if not issubclass(warn.category, (DeprecationWarning, PendingDeprecationWarning)):
                            warning_buffer.write(f"\n{warn.category.__name__}: {warn.message}\n\n")

                st.subheader("Custom Data Results")
                # Data Quality Report
                st.write("**Data Quality Report:**")
                for key, value in custom_quality_report.items():
                    st.write(f"{key}: {value}")

                # Assumption Checks
                display_result_section(
                    "Assumption Checks",
                    custom_results['Assumptions'],
                    "Download Assumptions (Text)",
                    str(custom_results['Assumptions']).encode('utf-8'),
                    "text/plain",
                    "assumptions"
                )

                # Descriptive Statistics (Controlled by Sidebar Checkbox)
                if show_descriptive:
                    display_result_section(
                        "Descriptive Statistics",
                        custom_results['Descriptive Stats'],
                        "Download Descriptive Stats (CSV)",
                        custom_results['Exports']['Descriptive Stats CSV'],
                        "text/csv",
                        "descriptive_stats"
                    )

                # LS Means (Controlled by Sidebar Checkbox)
                if show_ls_means:
                    display_result_section(
                        "LS Means Table",
                        custom_results['LS Means'],
                        "Download LS Means (CSV)",
                        custom_results['Exports']['LS Means CSV'],
                        "text/csv",
                        "ls_means"
                    )
                    display_result_section(
                        "LS Means Plot",
                        custom_results['LS Means Plot'],
                        "Download LS Means Plot (PNG)",
                        custom_results['Exports']['LS Means Plot PNG'],
                        "image/png",
                        "ls_means_plot"
                    )

                # Expected Mean Squares
                if custom_results['Expected Mean Squares'].size > 0:
                    st.write("This section shows the expected mean squares for the model.")
                    display_result_section(
                        "Expected Mean Squares",
                        custom_results['Expected Mean Squares'],
                        "Download Expected Mean Squares (CSV)",
                        custom_results['Exports']['Expected Mean Squares CSV'],
                        "text/csv",
                        "expected_mean_squares"
                    )

                # Statistical Test Results
                st.write("**Statistical Test Results:**")
                if 'Alternative Test' in custom_results:
                    st.markdown(f"<span style='color:red; font-weight:bold;'>Alternative Test Used (Assumptions Failed): {custom_results['Alternative Test']}</span>", unsafe_allow_html=True)
                elif model_type == "T-test":
                    st.write(f"T-test: t = {custom_results['T-test']['t_stat']:.2f}, p = {custom_results['T-test']['p_value']:.4f}")
                elif "ANOVA" in model_type:
                    st.write("ANOVA Table:")
                    st.write(custom_results['ANOVA'])
                    if 'Post-Hoc' in custom_results:
                        st.write("Post-Hoc (Tukey HSD):")
                        posthoc = custom_results['Post-Hoc']
                        if hasattr(posthoc, 'summary'):
                            st.text(posthoc.summary())
                        elif hasattr(posthoc, '_results_table'):
                            st.dataframe(pd.DataFrame(posthoc._results_table.data[1:], columns=posthoc._results_table.data[0]))
                        else:
                            st.write(posthoc)
                elif "Repeated Measures ANOVA" in model_type or "Mixed ANOVA" in model_type:
                    st.write("Mixed Model Summary:")
                    st.text(custom_results.get('Repeated Measures ANOVA', custom_results.get('Mixed ANOVA'))['Summary'])
                    st.write("**Run Summary:**")
                    run_summary = custom_results.get('Repeated Measures ANOVA', custom_results.get('Mixed ANOVA'))['Run Summary']
                    for key, value in run_summary.items():
                        st.write(f"{key}: {value}")
                    st.write("**Variance Estimates:**")
                    st.write(pd.DataFrame(custom_results.get('Repeated Measures ANOVA', custom_results.get('Mixed ANOVA'))['Variance Estimates'].items(),
                                         columns=['Metric', 'Value']))

                # Effect Sizes
                st.write("**Effect Sizes:**")
                st.write(custom_results['Effect Sizes'])

                # Visualization (Controlled by Sidebar Checkbox)
                if show_plot:
                    display_result_section(
                        "Box/Interaction Plot",
                        custom_results['Plot'],
                        "Download Plot (PNG)",
                        custom_results['Exports']['Plot PNG'],
                        "image/png",
                        "plot"
                    )

            except ValueError as e:
                if "NaN values detected" in str(e):
                    st.error(f"Analysis failed: {str(e)}. Try selecting 'mean' or 'median' for imputation, or clean your dataset to remove missing values.")
                else:
                    st.error(f"Error during analysis: {str(e)}. Please check your data and factors.")
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}. Please check your data and factors.")

        with explain_col:
            if custom_results:
                display_explanation("Assumption Checks", get_explanation("assumption_checks", model_type, custom_results, custom_quality_report))
                display_explanation("Results", get_explanation("results", model_type, custom_results, custom_quality_report))
                if show_plot:
                    display_explanation("Visualization", get_explanation("visualization", model_type, custom_results, custom_quality_report))
                # Display warnings
                warnings_text = warning_buffer.getvalue()
                if warnings_text:
                    st.markdown(
                        f'<div style="background:#fff3cd; color:#856404; border:1px solid #ffeeba; border-radius:8px; padding:1em; margin:1em 0;">'
                        f'<strong>Statistical Warnings:</strong><br><pre style="white-space:pre-wrap;">{warnings_text}\n</pre>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

if __name__ == "__main__":
    main()