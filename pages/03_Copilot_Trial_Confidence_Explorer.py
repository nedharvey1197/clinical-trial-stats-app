# 03_Copilot_Trial_Confidence_Explorer.py
import streamlit as st
import pandas as pd
import numpy as np

# Core app components from existing lib/
from lib.analysis import AnalysisOrchestrator
from lib.data import get_canned_example, get_example_metadata, get_example_description, canned_examples_with_desc
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

# Future helper modules to be added
# from lib.helpers.optimizer import generate_design_suggestions
# from lib.helpers.model_comparator import compare_models

# App Config
st.set_page_config(page_title="Copilot: Trial Confidence Explorer", page_icon="🧠", layout="wide")
st.title("🧠 Copilot for Clinical Trial Confidence Analysis")

# Step Tracker
step = 1
total_steps = 5


# Step 1: Trial Setup
st.header("1️⃣ Define Your Trial Scenario")
display_progress(step, total_steps)


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
    selected_label = st.selectbox(
        "Choose an example trial type:",
        [f"{ex['label']}" for ex in example_options],
        key="selected_example_label"
    )

    selected_example = next((ex for ex in example_options if ex["label"] == selected_label), None)
    if selected_example:
        st.session_state["use_example"] = True
        st.session_state["example_model_id"] = selected_example["model_id"]
        st.session_state["example_key"] = selected_example["example_key"]
        st.markdown(f"**Scenario:** {selected_example['summary']}")
else:
    st.session_state["use_example"] = False
    st.session_state["example_model_id"] = None
    st.session_state["example_key"] = None

st.markdown("Use the structured form below to describe your trial. Copilot will infer statistical structure from clinical inputs.")

with st.form("clinical_trial_selector_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        therapeutic_area = st.selectbox("Therapeutic Area", sorted(df_clinical_context["therapeutic_area"].unique()))
        phase = st.selectbox("Trial Phase", ["Phase I", "Phase II", "Phase III", "Phase IV", "Not Applicable"])
        design_type = st.selectbox("Trial Design Type", ["Parallel Group", "Crossover", "Single Arm", "Factorial", "Longitudinal"])

    with col2:
        intervention_type = st.selectbox("Intervention Type", get_valid_intervention_types(therapeutic_area))
        moa = st.selectbox("Mechanism of Action (MoA)", get_valid_moas(therapeutic_area, intervention_type))
        control_type = st.selectbox("Control Type", ["Placebo", "Standard of Care", "Active Comparator", "No Control"])

    with col3:
        endpoint = st.selectbox("Primary Endpoint", get_valid_endpoints(therapeutic_area, intervention_type, moa))
        n_per_arm = st.selectbox("Sample Size Per Arm", ["<20", "20–50", "51–100", ">100"])
        randomization = st.selectbox("Randomization", ["Randomized", "Non-Randomized", "Stratified Randomization"])

    submitted = st.form_submit_button("Continue to Step 2")

# Only proceed if form is submitted
if submitted:
    meta = map_clinical_context_to_meta(endpoint, design_type, phase, n_per_arm)
    st.success("Trial configuration submitted.")
    st.markdown("### 🔍 Copilot-Inferred Trial Metadata")
    st.json(meta)


# --- Step 2: Copilot Recommends Statistical Models ---
step += 1
st.header("2️⃣ Copilot Recommends Statistical Approach")
display_progress(step, total_steps)

from lib.model_registry import MODEL_REGISTRY

# Extract matching fields from meta
outcome_type = meta.get("outcome_type", "continuous")
repeated_factors = meta.get("repeated_factors", [])

# Filter based on outcome + repeated measures
recommended_models = [
    m for m in MODEL_REGISTRY
    if outcome_type in m.outcome_types
    and all(f in m.repeated_factors for f in repeated_factors)
]

st.markdown("### Recommended Statistical Models")

if recommended_models:
    model_labels = [f"{m.label} ({m.model_id})" for m in recommended_models]
    selected_label = st.radio("Select a model to run", model_labels)

    # Store selected model_id in session
    selected_model = next((m for m in recommended_models if f"{m.label} ({m.model_id})" == selected_label), None)
    if selected_model:
        st.session_state["selected_model_id"] = selected_model.model_id
        st.success(f"Model selected: {selected_model.label}")
        st.markdown(f"**Description:** {selected_model.description}")
        st.markdown(f"**Design Tags:** `{', '.join(selected_model.design_tags)}`")
else:
    st.warning("⚠️ No models match the current trial structure.")
    
## --- Step 3: Run Selected Model ---
step += 1
st.header("3️⃣ Run Selected Model")
display_progress(step, total_steps)

from lib.statistical_dispatcher import run_statistical_model

model_id = st.session_state.get("selected_model_id", None)

if model_id is None:
    st.warning("Please select a model in Step 2 to continue.")
else:
    st.markdown(f"**Selected Model ID:** `{model_id}`")

    # Data Upload or Example
    st.markdown("#### Upload trial data or use canned example")
    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
    use_example = st.checkbox("Use example dataset", value=True)

    if uploaded_file or use_example:
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
        else:
            example = get_canned_example(model_id)
            data = example["data"]
            if not isinstance(data, pd.DataFrame):
                st.error("Loaded example did not return a valid DataFrame.")
            description = example.get("description", "")
        st.markdown(f"**Trial Context:** {description}")
        st.dataframe(data.head())

        # Collect user mappings
        outcome = st.selectbox("Select outcome variable", data.columns)
        between_factors = st.multiselect("Between-subject factors", data.columns)
        within_factors = st.multiselect("Within-subject (repeated) factors", data.columns)

        if st.button("Run Statistical Model"):
            try:
                result = run_statistical_model(
                    model_id=model_id,
                    data=data,
                    outcome=outcome,
                    between_factors=between_factors,
                    within_factors=within_factors,
                )

                st.success("✅ Model run completed successfully")
                st.markdown("### 📊 Results:")
                st.json(result)

            except Exception as e:
                st.error(f"❌ Failed to run model: {str(e)}")

# Step 4: Visual & Comparative Dashboard
step += 1
st.header("4️⃣ Visualize & Compare")
display_progress(step, total_steps)

st.markdown("> Comparison coming soon: Bootstrap vs T-test vs Mann-Whitney")
# compare_models(data, ["T-test", "Bootstrap", "Mann-Whitney"])

# Step 5: Copilot Design Optimizer
step += 1
st.header("5️⃣ Copilot Design Suggestions")
display_progress(step, total_steps)

# Placeholder — insert real logic
st.markdown("Copilot suggests the following design optimizations:")
st.markdown("- Consider endpoint swap: continuous biomarker over subjective score")
st.markdown("- Enrich population to reduce variance")
st.markdown("- Apply covariate-adjusted model")

# Final Export or Summary Panel
st.success("Analysis complete. You may export your results or share a summary with your team.")

# Step 2: Copilot Analyzes Your Trial Design
step += 1
st.header("2️⃣ Copilot Analyzes Your Trial Design")
display_progress(step, total_steps)

outcome_map = {
    "continuous": "a continuous outcome",
    "binary": "a binary (yes/no) outcome",
    "time-to-event": "a time-to-event outcome"
}

sample_map = {
    "independent": "independent samples",
    "related": "related samples",
    "mixed": "mixed samples"
}

if input_mode == "Use a canned example":
    st.markdown("Copilot has analyzed the selected example and inferred the following trial structure:")
    
    summary_parts = [
        f"- Outcome type: **{outcome_map.get(meta['outcome_type'], meta['outcome_type'])}**",
        f"- Sample structure: **{sample_map.get(meta['sample_type'], meta['sample_type'])}**",
        f"- Paired or repeated measures: **{'Yes' if meta['paired'] else 'No'}**",
        f"- Distribution assumption: **{'Normal' if meta['normal'] else 'Non-normal or unknown'}**",
        f"- Sample size condition: **{'Small sample' if meta['small_n'] else 'Adequate sample size'}**"
    ]
    st.markdown("\n".join(summary_parts))
else:
    st.markdown("Based on your inputs, Copilot recommends the following test(s):")

# Recommend tests based on metadata
recommended = recommend_tests(
    outcome_type=meta["outcome_type"],
    sample_type=meta["sample_type"],
    paired=meta["paired"],
    normal=meta["normal"],
    small_n=meta["small_n"]
)

st.markdown("---")
st.markdown("### 🔍 Recommended Test(s):")

for rec in recommended:
    exp = explain_test(rec["test"])
    st.subheader(f"📌 {exp['title']}")
    st.markdown(f"**Summary:** {exp['summary']}")
    st.markdown(f"**When to Use:** {exp['when_to_use']}")
    st.markdown(f"**Assumptions:** {exp['assumptions']}")
    if exp['reference']:
        st.markdown(f"[📚 Learn more]({exp['reference']})")

# Step 3: Run Models
step += 1
st.header("3️⃣ Run Selected Analyses")
display_progress(step, total_steps)

orchestrator = AnalysisOrchestrator()
results = orchestrator.run_analysis(data, model_type="T-test")
st.write("**Results:**")
st.write(results)

# Step 4: Visual & Comparative Dashboard
step += 1
st.header("4️⃣ Visualize & Compare")
display_progress(step, total_steps)

st.markdown("> Comparison coming soon: Bootstrap vs T-test vs Mann-Whitney")
# compare_models(data, ["T-test", "Bootstrap", "Mann-Whitney"])

# Step 5: Copilot Design Optimizer
step += 1
st.header("5️⃣ Copilot Design Suggestions")
display_progress(step, total_steps)

# Placeholder — insert real logic
st.markdown("Copilot suggests the following design optimizations:")
st.markdown("- Consider endpoint swap: continuous biomarker over subjective score")
st.markdown("- Enrich population to reduce variance")
st.markdown("- Apply covariate-adjusted model")

# Final Export or Summary Panel
st.success("Analysis complete. You may export your results or share a summary with your team.")
