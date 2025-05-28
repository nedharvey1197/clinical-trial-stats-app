# 03_Copilot_Trial_Confidence_Explorer.py
import streamlit as st
import pandas as pd
import numpy as np
import logging

# Core app components from existing lib/
from lib.analysis import AnalysisOrchestrator
from lib.data import get_canned_example, get_example_metadata, get_example_description, canned_examples_with_desc, get_selector_metadata, repeated_factors_dict, between_factors_dict
from lib.ui_components import display_progress, display_result_section, toggle_outputs
from lib.ui_explanations import get_explanation
from lib.helpers.test_selector import recommend_tests
from lib.explainer.explain_test_selection import explain_test
from lib.helpers.metaMapping import map_clinical_context_to_meta
from lib.helpers.clinical_mapper import (
    df_clinical_context,
    get_valid_intervention_types,
    get_valid_moas,
    get_valid_endpoints
)
from lib.model_registry import MODEL_REGISTRY
from lib.statistical_dispatcher import run_statistical_model
from lib.test_recommender import recommend_statistical_test

# Future helper modules to be added
# from lib.helpers.optimizer import generate_design_suggestions
# from lib.helpers.model_comparator import compare_models
total_steps = 5
# App Config
st.set_page_config(page_title="Copilot: Trial Confidence Explorer", page_icon="🧠", layout="wide")
st.title("🧠 Copilot for Clinical Trial Confidence Analysis")

# At the top of your file, set default progress state
if "progress_step" not in st.session_state:
    st.session_state["progress_step"] = 1

def reset_downstream_state():
    for k in ["meta", "selected_model_id", "trial_meta"]:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state["progress_step"] = 1  # Reset progress to Step 1

# Step 1: Trial Setup
st.header("1️⃣ Define Your Trial Scenario")
display_progress(1, total_steps)

trial_type = st.radio(
    "How would you like to proceed?",
    ["Use a known trial example", "Enter trial details manually"],
    key="trial_type_selection"
)

# Define curated trial archetypes (11 known canned examples)
example_options = [
    {
        "model_id": "t_test",
        "label": "Simple two-group comparison (T-test)",
        "summary": "A trial comparing the efficacy of a new drug vs placebo in patients with mild hypertension.",
        "example_key": "T-test"
    },
    {
        "model_id": "one_way_anova",
        "label": "Dose-ranging trial (One-way ANOVA)",
        "summary": "A trial comparing three dosage levels (placebo, low, high) of a cholesterol-lowering drug.",
        "example_key": "One-way ANOVA"
    },
    {
        "model_id": "two_way_anova",
        "label": "Stratified comparison by age (Two-way ANOVA)",
        "summary": "A trial comparing Drug A vs B for diabetes, stratified by patient age (Young vs Old).",
        "example_key": "Two-way ANOVA"
    },
    {
        "model_id": "three_way_anova",
        "label": "Multicenter stratified trial (Three-way ANOVA)",
        "summary": "A trial evaluating pain relief across drug, age group, and trial site.",
        "example_key": "Three-way ANOVA"
    },
    {
        "model_id": "one_way_rm_anova",
        "label": "Single-arm longitudinal trial (One-way RM ANOVA)",
        "summary": "A within-subject trial measuring inflammation reduction over three timepoints.",
        "example_key": "One-way Repeated Measures ANOVA"
    },
    {
        "model_id": "two_way_rm_anova",
        "label": "Crossover trial with repeated tasks (Two-way RM ANOVA)",
        "summary": "A study evaluating cognitive performance under two task conditions across time.",
        "example_key": "Two-way Repeated Measures ANOVA"
    },
    {
        "model_id": "three_way_rm_anova",
        "label": "Stress response over time (Three-way RM ANOVA)",
        "summary": "A repeated measures study examining anxiety under drug, time, and stress condition.",
        "example_key": "Three-way Repeated Measures ANOVA"
    },
    {
        "model_id": "mixed_anova_1b1r",
        "label": "Treatment effect over time (Mixed ANOVA: 1 Between, 1 Repeated)",
        "summary": "A trial comparing antidepressant vs placebo, tracking changes in symptoms over time.",
        "example_key": "Mixed ANOVA (One Between, One Repeated)"
    },
    {
        "model_id": "mixed_anova_2b1r",
        "label": "Stratified repeated measures trial (Mixed ANOVA: 2 Between, 1 Repeated)",
        "summary": "A migraine study comparing drugs across age groups over multiple visits.",
        "example_key": "Mixed ANOVA (Two Between, One Repeated)"
    },
    {
        "model_id": "mixed_anova_1b2r",
        "label": "Dual-condition repeated trial (Mixed ANOVA: 1 Between, 2 Repeated)",
        "summary": "A crossover trial comparing drug effects on lung function across time and exercise intensity.",
        "example_key": "Mixed ANOVA (One Between, Two Repeated)"
    },
    {
        "model_id": "complex_mixed_anova",
        "label": "Multifactorial psychiatric trial (Complex Mixed ANOVA)",
        "summary": "A schizophrenia trial analyzing drug, age, stress, and time on symptom severity.",
        "example_key": "Complex Mixed ANOVA"
    }
]

# Show selection menu if using an example
if trial_type == "Use a known trial example":
    st.markdown("### Select a representative trial example")
    select_options = [
        f"{ex['label']} – {ex['summary']}"
        for ex in example_options
    ]
    # Render dropdown
    selected_combined = st.selectbox(
        "Choose an example trial type:",
        select_options,
        key="selected_example_label",
        on_change=reset_downstream_state
    )

    # Find the matching example based on combined label
    selected_example = next(
        (ex for ex in example_options if selected_combined.startswith(ex["label"])),
        None
    )
    # Save to session state
    if selected_example:
        st.session_state["use_example"] = True
        st.session_state["example_model_id"] = selected_example["model_id"]
        st.session_state["example_key"] = selected_example["example_key"]
        st.markdown(f"**Scenario:** {selected_example['summary']}")
        
        st.markdown("The following structured table describes your trial. Copilot will infer statistical structure from clinical inputs.")

        # Get selector metadata for the selected example
        selector_metadata = get_selector_metadata(selected_example["example_key"])
        print("[LOG] Selected example key:", selected_example["example_key"])
        print("[LOG] Selector metadata:", selector_metadata)

        with st.form("clinical_trial_selector_form"):
            st.markdown(f"""
            <style>
            .copilot-meta-table {{
                width: 100%;
                border-collapse: separate;
                border-spacing: 0 0.5em;
                margin-bottom: 1.5em;
            }}
            .copilot-meta-table th {{
                background: #f8fafc;
                color: #222;
                font-size: 1.1em;
                font-weight: 700;
                padding: 0.75em 1em;
                border-bottom: 2px solid #4CAF50;
                border-radius: 8px 8px 0 0;
            }}
            .copilot-meta-table td {{
                background: #fff;
                color: #222;
                font-size: 1em;
                padding: 0.7em 1em;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
            }}
            .copilot-meta-table tr:not(:last-child) td {{
                border-bottom: none;
            }}
            </style>
            <table class="copilot-meta-table">
                <tr>
                    <th>Trial Characteristics</th>
                    <th>Intervention Details</th>
                    <th>Trial Parameters</th>
                </tr>
                <tr>
                    <td><b>Therapeutic Area:</b> {selector_metadata['therapeutic_area']}</td>
                    <td><b>Intervention Type:</b> {selector_metadata['intervention_type']}</td>
                    <td><b>Primary Endpoint:</b> {selector_metadata['endpoint']}</td>
                </tr>
                <tr>
                    <td><b>Trial Phase:</b> {selector_metadata['phase']}</td>
                    <td><b>Mechanism of Action (MoA):</b> {selector_metadata['moa']}</td>
                    <td><b>Sample Size Per Arm:</b> {selector_metadata['n_per_arm']}</td>
                </tr>
                <tr>
                    <td><b>Trial Design Type:</b> {selector_metadata['design_type']}</td>
                    <td><b>Control Type:</b> {selector_metadata['control_type']}</td>
                    <td><b>Randomization:</b> {selector_metadata['randomization']}</td>
                </tr>
            </table>
            """, unsafe_allow_html=True)
            submitted = st.form_submit_button("Continue to Step 2")
            if submitted:
                st.session_state["therapeutic_area"] = selector_metadata["therapeutic_area"]
                st.session_state["phase"] = selector_metadata["phase"]
                st.session_state["design_type"] = selector_metadata["design_type"]
                st.session_state["intervention_type"] = selector_metadata["intervention_type"]
                st.session_state["moa"] = selector_metadata["moa"]
                st.session_state["control_type"] = selector_metadata["control_type"]
                st.session_state["endpoint"] = selector_metadata["endpoint"]
                st.session_state["n_per_arm"] = selector_metadata["n_per_arm"]
                st.session_state["randomization"] = selector_metadata["randomization"]
                print("[LOG] Set session_state for canned example:", {
                    "therapeutic_area": selector_metadata["therapeutic_area"],
                    "phase": selector_metadata["phase"],
                    "design_type": selector_metadata["design_type"],
                    "intervention_type": selector_metadata["intervention_type"],
                    "moa": selector_metadata["moa"],
                    "control_type": selector_metadata["control_type"],
                    "endpoint": selector_metadata["endpoint"],
                    "n_per_arm": selector_metadata["n_per_arm"],
                    "randomization": selector_metadata["randomization"]
                })
                # Set meta for canned example
                example_key = selected_example["example_key"]
                example_dict = canned_examples_with_desc[example_key]
                outcome_type = example_dict["metadata"]["outcome_type"]
                repeated_factors = repeated_factors_dict.get(example_key, [])
                st.session_state["meta"] = {
                    "outcome_type": outcome_type,
                    "repeated_factors": repeated_factors
                }
                print("[LOG] Set meta for canned example:", st.session_state["meta"])
                st.session_state["progress_step"] = 2  # Move to Step 2
else:
    st.session_state["use_example"] = False
    st.session_state["example_model_id"] = None
    st.session_state["example_key"] = None

    st.markdown("Use the structured form below to describe your trial. Copilot will infer statistical structure from clinical inputs.")

    # Add this CSS before the form section (only once, near the top or before the form)
    st.markdown("""
        <style>
        .copilot-form-header {
            font-size: 1.1em;
            font-weight: 700;
            margin-bottom: 0.5em;
            color: #222;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.form("clinical_trial_selector_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="copilot-form-header">Trial Characteristics</div>', unsafe_allow_html=True)
            therapeutic_area = st.selectbox("Therapeutic Area", sorted(df_clinical_context["therapeutic_area"].unique()), key="therapeutic_area_select")
            phase = st.selectbox("Trial Phase", ["Phase I", "Phase II", "Phase III", "Phase IV", "Not Applicable"], key="phase_select")
            design_type = st.selectbox("Trial Design Type", ["Parallel Group", "Crossover", "Single Arm", "Factorial", "Longitudinal"], key="design_type_select")

        with col2:
            st.markdown('<div class="copilot-form-header">Intervention Details</div>', unsafe_allow_html=True)
            intervention_type = st.selectbox("Intervention Type", get_valid_intervention_types(therapeutic_area), key="intervention_type_select")
            moa = st.selectbox("Mechanism of Action (MoA)", get_valid_moas(therapeutic_area, intervention_type), key="moa_select", )
            control_type = st.selectbox("Control Type", ["Placebo", "Standard of Care", "Active Comparator", "No Control"], key="control_type_select")

        with col3:
            st.markdown('<div class="copilot-form-header">Trial Parameters</div>', unsafe_allow_html=True)
            endpoint = st.selectbox("Primary Endpoint", get_valid_endpoints(therapeutic_area, intervention_type, moa), key="endpoint_select")
            n_per_arm = st.selectbox("Sample Size Per Arm", ["<20", "20–50", "51–100", ">100"], key="n_per_arm_select")
            randomization = st.selectbox("Randomization", ["Randomized", "Non-Randomized", "Stratified Randomization"], key="randomization_select")

        submitted = st.form_submit_button("Continue to Step 2")
        if submitted:
            st.session_state["therapeutic_area"] = therapeutic_area
            st.session_state["phase"] = phase
            st.session_state["design_type"] = design_type
            st.session_state["intervention_type"] = intervention_type
            st.session_state["moa"] = moa
            st.session_state["control_type"] = control_type
            st.session_state["endpoint"] = endpoint
            st.session_state["n_per_arm"] = n_per_arm
            st.session_state["randomization"] = randomization
            print("[DEBUG] Set session_state for manual entry:", {
                "therapeutic_area": therapeutic_area,
                "phase": phase,
                "design_type": design_type,
                "intervention_type": intervention_type,
                "moa": moa,
                "control_type": control_type,
                "endpoint": endpoint,
                "n_per_arm": n_per_arm,
                "randomization": randomization
            })
            # Set meta for manual entry
            meta_dict = map_clinical_context_to_meta(
                endpoint,
                design_type,
                phase,
                n_per_arm
            )
            outcome_type = meta_dict.get("outcome_type", "continuous")
            repeated_factors = repeated_factors_dict.get(design_type, [])
            st.session_state["meta"] = {
                "outcome_type": outcome_type,
                "repeated_factors": repeated_factors
            }
            print("[DEBUG] Set meta for manual entry:", st.session_state["meta"])
            st.session_state["progress_step"] = 2  # Move to Step 2

# Only proceed if form is submitted
if submitted:
    meta = map_clinical_context_to_meta(
        st.session_state["endpoint"],
        st.session_state["design_type"],
        st.session_state["phase"],
        st.session_state["n_per_arm"]
    )
    st.success("Trial configuration submitted.")



# --- Step 2: Copilot Recommends Statistical Model ---
if st.session_state.get("progress_step", 1) >= 2:
    step = 2
    st.header("2️⃣ Copilot Recommends Statistical Approach")
    display_progress(step, total_steps)
    
    st.markdown("### 🔍 Copilot-Inferred Trial Metadata")
    # Display a human-readable summary of the trial design
    summary_lines = [
        f"- Outcome type: {st.session_state['meta']['outcome_type']}",
        f"- Number of groups: {n_groups if 'n_groups' in locals() else 'N/A'}",
        f"- Between-subjects factors: {n_between if 'n_between' in locals() else 'N/A'}",
        f"- Repeated measures factors: {n_within if 'n_within' in locals() else 'N/A'} ({', '.join(st.session_state['meta']['repeated_factors']) if st.session_state['meta']['repeated_factors'] else 'None'})"
    ]
    st.success("**Trial Design Summary**\n" + "\n".join(summary_lines))

    # Gather design info from meta/session_state
    meta = st.session_state["meta"]
    # You may need to infer these from your meta/session_state or data:
    # For canned examples, you can get the grouping variable and count unique values
    # For manual entry, you may need to ask the user or infer from uploaded data
    # Example for canned:
    if st.session_state.get("use_example"):
        example_key = st.session_state["example_key"]
        example_dict = canned_examples_with_desc[example_key]
        # Assume 'Drug' is the main grouping variable for between-subjects
        group_col = "Drug" if "Drug" in example_dict["data"].columns else example_dict["data"].columns[1]
        n_groups = example_dict["data"][group_col].nunique()
        n_between = len([f for f in example_dict["data"].columns if f in between_factors_dict[example_key]])
        n_within = len(meta.get("repeated_factors", []))
        n_within_levels = [example_dict["data"][f].nunique() for f in meta.get("repeated_factors", [])]
    else:
        # For manual, you may need to infer or ask for these
        # Here we use placeholders; you should replace with real logic
        n_between = 1  # TODO: infer from user input
        n_within = len(meta.get("repeated_factors", []))
        n_groups = 2   # TODO: infer from user input or uploaded data
        n_within_levels = [2 for _ in range(n_within)]  # TODO: infer from user input or uploaded data

    model_id = recommend_statistical_test(n_between, n_within, n_groups, n_within_levels)
    print(f"[DEBUG] Recommended model_id: {model_id}")

    # Now use model_id to select the model from MODEL_REGISTRY
    from lib.model_registry import MODEL_REGISTRY
    matched_models = [m for m in MODEL_REGISTRY if m.model_id == model_id]
    # Build explanation string for why the test was chosen
    gating_explanation = []
    if n_between == 0 and n_within == 1:
        gating_explanation.append("No between-subjects factors and 1 repeated factor: One-way Repeated Measures ANOVA.")
    elif n_between == 0 and n_within == 2:
        gating_explanation.append("No between-subjects factors and 2 repeated factors: Two-way Repeated Measures ANOVA.")
    elif n_between == 0 and n_within == 3:
        gating_explanation.append("No between-subjects factors and 3 repeated factors: Three-way Repeated Measures ANOVA.")
    elif n_between == 1 and n_within == 0 and n_groups == 2:
        gating_explanation.append("1 between-subjects factor with 2 groups: T-test.")
    elif n_between == 1 and n_within == 0 and n_groups > 2:
        gating_explanation.append("1 between-subjects factor with >2 groups: One-way ANOVA.")
    elif n_between == 2 and n_within == 0:
        gating_explanation.append("2 between-subjects factors: Two-way ANOVA.")
    elif n_between == 3 and n_within == 0:
        gating_explanation.append("3 between-subjects factors: Three-way ANOVA.")
    elif n_between == 1 and n_within == 1:
        gating_explanation.append("1 between and 1 repeated factor: Mixed ANOVA (One Between, One Repeated).")
    elif n_between == 2 and n_within == 1:
        gating_explanation.append("2 between and 1 repeated factor: Mixed ANOVA (Two Between, One Repeated).")
    elif n_between == 1 and n_within == 2:
        gating_explanation.append("1 between and 2 repeated factors: Mixed ANOVA (One Between, Two Repeated).")
    elif n_between >= 2 and n_within >= 2:
        gating_explanation.append("2+ between and 2+ repeated factors: Complex Mixed ANOVA.")
    else:
        gating_explanation.append("No appropriate test found for this design.")
        
    if matched_models:
        best_model = matched_models[0]
        st.session_state["selected_model_id"] = best_model.model_id
        st.success(f"Recommended Model: **{best_model.label}**")
        st.markdown(f"**Why?** {gating_explanation[0]}")
        st.markdown(f"**Description**: {best_model.description}")
        st.markdown(f"**Design Tags**: `{', '.join(best_model.design_tags)}`")
        if st.button("Continue to Calculations", key="Continue2_3"):
            st.session_state["progress_step"] = 3
    else:
        st.warning("No matching model found for this design. Please check the design or expand the model registry.")


    # DEPRECATED: Old matching logic by repeated_factors/outcome_type is no longer used.


# --- Step 3: Run Selected Model ---
if st.session_state.get("progress_step", 1) >= 3:
    step = 3
    st.header("3️⃣ Run Selected Model")
    display_progress(step, total_steps)

    model_id = st.session_state.get("selected_model_id", None)

    if model_id is None:
        st.warning("Please complete Step 2 before running analysis.")
    else:
        st.markdown(f"**Selected Model ID:** `{model_id}`")

        uploaded_file = st.file_uploader("Upload trial dataset (CSV)", type=["csv"])
        use_example = st.checkbox("Use canned example", value=True)

        if uploaded_file or use_example:
            if uploaded_file:
                data = pd.read_csv(uploaded_file)
            else:
                example_dict = get_canned_example(model_id)
                data = example_dict['data']  # Extract the DataFrame from the dictionary

            st.markdown("✅ Data Preview:")
            st.dataframe(data.head())

            outcome = st.selectbox("Select outcome variable", data.columns)
            between_factors = st.multiselect("Between-subject factors", data.columns)
            within_factors = st.multiselect("Within-subject (repeated) factors", data.columns)

            if st.button("Run Analysis", key="RunAnalysis"):
                try:
                    result = run_statistical_model(
                        model_id=model_id,
                        data=data,
                        outcome=outcome,
                        between_factors=between_factors,
                        within_factors=within_factors,
                    )
                    st.success("✅ Model run complete.")
                    st.markdown("### 📊 Results")
                    st.json(result)

                except Exception as e:
                    st.error(f"❌ Model execution failed: {str(e)}")