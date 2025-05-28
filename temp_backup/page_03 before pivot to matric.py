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
st.header("1️⃣ Define Trial Scenario")
display_progress(step, total_steps)

explanation_map = {
    "T-test": (
        "The following trial example illustrates the use of a **T-test**.\n"
        "\nIt compares two independent groups on a continuous outcome under the assumption of normal distribution\n"
        "\nThis test is commonly used in two-arm Phase II trials."
    ),
    "One-way ANOVA": (
        "The following trial example demonstrates a **one-way ANOVA**, used to compare three or more independent groups on a continuous outcome.\n"
        "\nIt helps detect overall group differences, often used in dose-finding studies."
    ),
    "Two-way ANOVA": (
        "The following trial example reflects a **two-way ANOVA**, where two independent factors (e.g., treatment and site) are evaluated for their main effects and interaction on a continuous outcome."
    ),
    "Three-way ANOVA": (
        "The following trial example applies a **three-way ANOVA**.\n"
        "It is ideal for assessing the combined effects and interactions of three between-subject factors (e.g., drug, site, demographic group) on a continuous outcome."
    ),
    "One-way Repeated Measures ANOVA": (
        "The following trial example illustrates a **one-way repeated measures ANOVA**.\n"
        "It is appropriate for comparing multiple timepoints within the same subjects.\n"
        "It is ideal for detecting longitudinal trends in within-subject designs."
    ),
    "Two-way Repeated Measures ANOVA": (
        "The following trial example is a **two-way repeated measures ANOVA**.\n"
        "It is where repeated measures across time are analyzed alongside a second within-subject condition (e.g., rest vs. stress)."
    ),
    "Three-way Repeated Measures ANOVA": (
        "The following trial example demonstrates a **three-way repeated measures ANOVA**.\n"
        "It evaluates interactions between multiple within-subject factors over time."
    ),
    "Mixed ANOVA (One Between, One Repeated)": (
        "The following trial example is based on a **mixed ANOVA** design, combining one between-subject factor (e.g., treatment group) and one repeated measure (e.g., pre vs. post).\n"
        "It is useful for testing group × time interactions."
    ),
    "Mixed ANOVA (Two Between, One Repeated)": (
        "The following trial example reflects a **mixed factorial ANOVA** with two between-subjects factors and one repeated measure.\n"
        "It is suited for more complex interaction testing across subject groups and timepoints."
    ),
    "Mixed ANOVA (One Between, Two Repeated)": (
        "The following trial example illustrates a **mixed ANOVA** involving one between-subject factor and two repeated measures (e.g., time and condition). "
        "It allows for rich within- and between-subject interaction analysis."
    ),
    "Complex Mixed ANOVA": (
        "The following trial example represents a **full factorial mixed model**, combining multiple between-subject and repeated factors. "
        "It is ideal for multi-phase or multi-arm trials requiring interaction modeling across dimensions."
    )
}

input_mode = st.radio("Choose input method", ["Use a canned example", "Describe my own trial"])

if input_mode == "Use a canned example":
    model_names = list(canned_examples_with_desc.keys())
    selected_model = st.selectbox("Select a canned trial example", model_names)
    example = get_canned_example(selected_model)
    # Safely pull explanation
    copilot_note = explanation_map.get(selected_model, None)
    if copilot_note:
        st.markdown(f"## 💡 Lets look at an Example Trial")
        st.markdown(copilot_note)
    st.markdown(f"### Trial Description:\n\n{example['description']}")
    st.dataframe(example["data"].head())
    meta = get_example_metadata(selected_model)
else:
    st.markdown("### Custom Trial Design")
    outcome_type = st.selectbox("What type of outcome are you measuring?", ["continuous", "binary", "time-to-event"])
    sample_type = st.selectbox("How are samples structured?", ["independent", "related", "mixed"])
    paired = st.radio("Is the data paired or repeated (e.g., pre/post)?", [True, False])
    normal = st.radio("Do you expect a normal distribution?", [True, False])
    small_n = st.checkbox("Is the sample size small (e.g., <30 per group)?")
    meta = {
        "outcome_type": outcome_type,
        "sample_type": sample_type,
        "paired": paired,
        "normal": normal,
        "small_n": small_n
    }




# --- Step 2: Copilot Recommends Statistical Models ---
step += 1
st.header("2️⃣ Copilot Recommends Statistical Approach")
display_progress(step, total_steps)

# Load curated models
from lib.model_registry import MODEL_REGISTRY

# Extract parameters from meta
outcome_type = meta.get("outcome_type", "continuous")
repeated_factors = meta.get("repeated_factors", [])  # Should be provided by meta mapper logic

# Filter model registry
filtered_models = [
    model for model in MODEL_REGISTRY
    if outcome_type in model.outcome_types
    and all(rf in model.repeated_factors for rf in repeated_factors)
]

st.markdown("### Recommended Statistical Models")

if filtered_models:
    for model in filtered_models:
        with st.expander(f"📊 {model.label}"):
            st.markdown(f"**Description:** {model.description}")
            st.markdown(f"**Repeated Factors:** `{', '.join(model.repeated_factors) or 'None'}`")
            st.markdown(f"**Design Tags:** `{', '.join(model.design_tags)}`")
            st.code(f"Model ID: {model.model_id}", language="text")
else:
    st.warning("⚠️ No model matched the current trial design. Try adjusting design assumptions or expand your model registry.")

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
