# --- Step 1: Select Trial Type ---
st.header("1️⃣ Define Trial Context")
trial_type = st.radio(
    "How would you like to proceed?",
    ["Use a known trial example", "Enter trial details manually"],
    key="trial_type_selection"
)

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
        st.markdown(f"**Description:** {selected_example['summary']}")
else:
    st.session_state["use_example"] = False
    st.session_state["example_model_id"] = None
    st.session_state["example_key"] = None