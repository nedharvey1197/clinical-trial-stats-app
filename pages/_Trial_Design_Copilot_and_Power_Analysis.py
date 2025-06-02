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
    get_valid_endpoints,
    get_default_moa,
    get_default_endpoint
)
from lib.model_registry import MODEL_REGISTRY, get_model_by_id
from lib.statistical_dispatcher import run_statistical_model
from lib.test_recommender import recommend_statistical_test

# Future helper modules to be added
# from lib.helpers.optimizer import generate_design_suggestions
# from lib.helpers.model_comparator import compare_models
total_steps = 5
# App Config
st.set_page_config(page_title="Trial Design Copilot & Power Analysis", page_icon="🤖", layout="wide")
st.title("🤖 Trial Design Copilot & Power Analysis")
st.markdown("<h3 p class='subtitle'>Step-by-step workflow for designing, validating, and analyzing clinical trials. Includes statistical guidance, power/sample size calculations, and robust data validation for confident, explainable results.</h3>", unsafe_allow_html=True)

# Add reset functionality to sidebar
with st.sidebar:
    st.markdown("### Session Control")
    if st.button("🔄 Reset All Progress", type="primary"):
        # Clear all session state variables
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        # Reset progress step
        st.session_state["progress_step"] = 1
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Current Progress")
    if "progress_step" in st.session_state:
        st.markdown(f"**Current Step:** {st.session_state['progress_step']}/5")
    else:
        st.markdown("**Current Step:** 1/5")
    
    # Glossary & Learn More
    with st.expander("📚 Glossary & Learn More", expanded=False):
        st.markdown("""
**Outcome Type:** The kind of data measured for your primary endpoint (e.g., continuous, binary, count, time-to-event). [Learn more](https://en.wikipedia.org/wiki/Dependent_and_independent_variables)

**Covariate:** A variable that may influence the outcome but is not the main focus of the study (e.g., age, sex). [Learn more](https://en.wikipedia.org/wiki/Covariate)

**Normality:** The assumption that data are distributed in a bell-shaped (normal) curve. [Learn more](https://en.wikipedia.org/wiki/Normal_distribution)

**Sphericity:** The assumption that variances of the differences between all combinations of related groups are equal (important for repeated measures). [Learn more](https://en.wikipedia.org/wiki/Sphericity)

**Power:** The probability of detecting a true effect if it exists. [Learn more](https://en.wikipedia.org/wiki/Statistical_power)

**Sample Size:** The number of subjects in each group or arm of the trial. [Learn more](https://en.wikipedia.org/wiki/Sample_size_determination)

**Randomization:** The process of assigning subjects to groups by chance. [Learn more](https://en.wikipedia.org/wiki/Randomization)

**Repeated Measures:** When the same subjects are measured more than once (e.g., before and after treatment). [Learn more](https://en.wikipedia.org/wiki/Repeated_measures_design)
        """)

# At the top of your file, set default progress state
if "progress_step" not in st.session_state:
    st.session_state["progress_step"] = 1

def reset_steps_from(step):
    # Remove all session state keys for steps >= step and reset progress
    for k in [
        "moa", "endpoint", "outcome_type", "n_groups", "repeated_measures", "covariates",
        "selected_model_id", "required_columns", "data"
    ]:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state["progress_step"] = step

# Mapping from endpoint to recommended outcome variable type
endpoint_to_outcome_type = {
    "Event-Free Survival": "time-to-event",
    "Major Adverse Cardiac Events": "binary",
    "Blood Pressure Change": "continuous",
    "Restenosis Rate": "binary",
    "Stent Thrombosis": "binary",
    "Target Lesion Revascularization": "binary",
    "LDL Reduction": "continuous",
    "HbA1c Change": "continuous",
    "Depression Score": "continuous",
    "PASI Score": "continuous",
    "Seizure Frequency": "count",
    # ...add more as needed...
}

# Utility: Canonical test key mapping for explain_test

def get_canonical_test_key(label: str) -> str:
    label_norm = label.lower().replace('-', ' ').replace('_', ' ').replace('  ', ' ').strip()
    mapping = {
        "independent samples t test": "T-test",
        "independent samples t-test": "T-test",
        "t test": "T-test",
        "t-test": "T-test",
        "independent t test": "T-test",
        "independent t-test": "T-test",
        "paired t test": "Paired T-Test",
        "paired t-test": "Paired T-Test",
        "mann whitney u": "Mann-Whitney U Test",
        "mann-whitney u": "Mann-Whitney U Test",
        "bootstrap t test": "Bootstrap T-Test",
        "chi square test": "Chi-Square Test",
        "chi-square test": "Chi-Square Test",
        "fisher's exact test": "Fisher's Exact Test",
        "log rank test": "Log-Rank Test",
        "wilcoxon signed rank test": "Wilcoxon Signed-Rank Test",
        # Add more as needed
    }
    return mapping.get(label_norm, label)

# --- Step 1: Main Trial Setup ---
st.header("📝 Step 1: Main Trial Setup")
with st.expander("Step 1: Main Trial Setup", expanded=st.session_state.get("progress_step", 1) == 1):
    if st.session_state.get("progress_step", 1) > 1:
        # Show summary
        st.write("**Therapeutic Area:**", st.session_state["therapeutic_area"])
        st.write("**Trial Phase:**", st.session_state["phase"])
        st.write("**Trial Design Type:**", st.session_state["design_type"])
        st.write("**Intervention Type:**", st.session_state["intervention_type"])
        st.write("**Control Type:**", st.session_state["control_type"])
        st.write("**Sample Size Per Arm:**", st.session_state["n_per_arm"])
        st.write("**Randomization:**", st.session_state["randomization"])
        if st.button("Edit Step 1"):
            reset_steps_from(1)
    elif st.session_state.get("progress_step", 1) == 1:
        with st.form("main_trial_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                therapeutic_area = st.selectbox("Therapeutic Area", sorted(df_clinical_context["therapeutic_area"].unique()), key="step1_therapeutic_area_select", help="The medical field or disease area being studied (e.g., Cardiology, Oncology).")
                phase = st.selectbox("Trial Phase", ["Phase I", "Phase II", "Phase III", "Phase IV", "Not Applicable"], key="step1_phase_select", help="The stage of clinical development (I-IV) or 'Not Applicable' for observational studies.")
                design_type = st.selectbox("Trial Design Type", ["Parallel Group", "Crossover", "Single Arm", "Factorial", "Longitudinal"], key="step1_design_type_select", help="How subjects are assigned to groups and how interventions are compared. [Learn more](https://en.wikipedia.org/wiki/Randomized_controlled_trial)")
            with col2:
                intervention_type = st.selectbox("Intervention Type", get_valid_intervention_types(therapeutic_area), key=f"step1_intervention_type_select_{therapeutic_area}", help="The type of treatment or intervention being tested (e.g., Drug, Device, Procedure).")
                control_type = st.selectbox("Control Type", ["Placebo", "Standard of Care", "Active Comparator", "No Control"], key="step1_control_type_select", help="The comparison group for the intervention (e.g., Placebo, Standard of Care, Active Comparator, or No Control).")
            with col3:
                n_per_arm = st.selectbox("Sample Size Per Arm", ["<20", "20–50", "51–100", ">100"], key="step1_n_per_arm_select", help="The number of subjects in each group or arm. This will be reviewed later for adequacy. [Learn more](https://en.wikipedia.org/wiki/Sample_size_determination)")
                st.caption("Note: Your sample size selection will be reviewed in later steps to ensure it is sufficient for valid statistical analysis and your desired power.")
                randomization = st.selectbox("Randomization", ["Randomized", "Non-Randomized", "Stratified Randomization"], key="step1_randomization_select", help="How subjects are assigned to groups. Randomization helps prevent bias. [Learn more](https://en.wikipedia.org/wiki/Randomization)")
            submitted = st.form_submit_button("Continue to Step 2")
        if submitted:
            st.session_state["therapeutic_area"] = therapeutic_area
            st.session_state["phase"] = phase
            st.session_state["design_type"] = design_type
            st.session_state["intervention_type"] = intervention_type
            st.session_state["control_type"] = control_type
            st.session_state["n_per_arm"] = n_per_arm
            st.session_state["randomization"] = randomization
            st.session_state["progress_step"] = 2

# --- Step 2: Key Modeling Details ---
st.header("🧬 Step 2: Key Modeling Details")
with st.expander("Step 2: Key Modeling Details", expanded=st.session_state.get("progress_step", 1) == 2):
    if st.session_state.get("progress_step", 1) > 2:
        st.write("**MoA:**", st.session_state["moa"])
        st.write("**Primary Endpoint:**", st.session_state["endpoint"])
        st.write("**Outcome Variable Type:**", st.session_state["outcome_type"])
        st.write("**Number of Groups:**", st.session_state["n_groups"])
        st.write("**Repeated Measures:**", st.session_state["repeated_measures"])
        st.write("**Covariates:**", st.session_state["covariates"])
        if st.button("Edit Step 2"):
            reset_steps_from(2)
    elif st.session_state.get("progress_step", 1) == 2:
        moa_options = get_valid_moas(st.session_state["therapeutic_area"], st.session_state["intervention_type"])
        default_moa = get_default_moa(st.session_state["therapeutic_area"], st.session_state["intervention_type"])
        moa = st.selectbox(
            "Mechanism of Action (MoA)",
            moa_options,
            index=moa_options.index(default_moa) if default_moa in moa_options else 0,
            key=f"step2_moa_select_{st.session_state['therapeutic_area']}_{st.session_state['intervention_type']}",
            help="The specific biological or technological mechanism by which the intervention works (e.g., 'Stent' for a device, 'Beta Blocker' for a drug). [Learn more](https://en.wikipedia.org/wiki/Mechanism_of_action)"
        )
        endpoint_options = get_valid_endpoints(st.session_state["therapeutic_area"], st.session_state["intervention_type"], moa, st.session_state["phase"])
        default_endpoint = get_default_endpoint(st.session_state["therapeutic_area"], st.session_state["intervention_type"], st.session_state["phase"])
        endpoint = st.selectbox(
            "Primary Endpoint",
            endpoint_options,
            index=endpoint_options.index(default_endpoint) if default_endpoint in endpoint_options else 0,
            key=f"step2_endpoint_select_{st.session_state['therapeutic_area']}_{st.session_state['intervention_type']}_{moa}_{st.session_state['phase']}",
            help="The main clinical outcome used to judge the effectiveness of the intervention (e.g., 'Major Adverse Cardiac Events', 'Event-Free Survival'). [Learn more](https://en.wikipedia.org/wiki/Endpoint_(clinical_research))"
        )
        # --- Use metaMapping for all suggestions and prepopulation ---
        meta = map_clinical_context_to_meta(
            endpoint=endpoint,
            design_type=st.session_state["design_type"],
            phase=st.session_state["phase"],
            n_per_arm=st.session_state["n_per_arm"]
        )
        st.success(f"Recommended outcome variable type for '{endpoint}': **{meta['outcome_type']}**")
        st.caption(f"Sample type: {meta['sample_type']} | Paired: {meta['paired']} | Normality likely: {meta['normal']} | Small-N: {meta['small_n']}")
        outcome_type = st.selectbox(
            "Outcome Variable Type",
            ["continuous", "binary", "count", "time-to-event"],
            index=["continuous", "binary", "count", "time-to-event"].index(meta["outcome_type"]),
            key="step2_outcome_type_select",
            help="The type of data measured for the primary endpoint.\n- 'continuous': Numeric values (e.g., blood pressure, cholesterol)\n- 'binary': Yes/No or Success/Failure (e.g., event occurred)\n- 'count': Number of events (e.g., number of exacerbations)\n- 'time-to-event': Time until an event occurs (e.g., survival time) [Learn more](https://en.wikipedia.org/wiki/Dependent_and_independent_variables)"
        )
        n_groups = st.number_input(
            "Number of Groups",
            min_value=1, max_value=20, value=2, step=1,
            key="step2_n_groups_input",
            help="The number of independent groups or arms in your trial (e.g., 2 for 'treatment' vs 'control'). [Learn more](https://en.wikipedia.org/wiki/Randomized_controlled_trial)"
        )
        # --- Repeated Measures Logic with Advanced Override ---
        repeated_measures_allowed = "Time" in meta["repeated_factors"]
        repeated_measures_disabled = not repeated_measures_allowed
        # Show recommendation or not relevant message first
        if repeated_measures_allowed:
            st.info("Repeated measures are recommended for this design and endpoint.")
        elif repeated_measures_disabled:
            st.warning("Repeated measures are not relevant for this design and endpoint.")
        else:
            st.info("Repeated measures are optional for this design and endpoint.")

        # Only one repeated measures checkbox is visible at a time
        repeated_measures = False
        advanced_override = False
        if repeated_measures_allowed:
            # Show enabled main checkbox
            repeated_measures = st.checkbox(
                "Repeated Measures?",
                value=True,
                key="step2_repeated_measures_checkbox",
                help="Check this box if the same subjects are measured more than once (e.g., before and after treatment, or at multiple time points). Example: Measuring blood pressure at baseline, 1 month, and 3 months in the same patients."
            )
        elif repeated_measures_disabled:
            st.markdown(
                "<div style='margin-left:1em'><span style='color:#b71c1c; font-weight:bold;'>(Advanced: Override)</span></div>",
                unsafe_allow_html=True
            )
            advanced_override = st.checkbox(
                "Enable advanced override (for expert/statistical users only)",
                value=False,
                key="step2_advanced_override_checkbox",
                help="Enable this to manually override the repeated measures setting. For expert/statistical users only."
            )
            if advanced_override:
                repeated_measures = st.checkbox(
                    "Repeated Measures? (Override)",
                    value=False,
                    key="step2_repeated_measures_checkbox_override",
                    help="Override: Check this box if you want to force repeated measures for this design.",
                    disabled=False
                )
            else:
                repeated_measures = st.checkbox(
                    "Repeated Measures?",
                    value=False,
                    key="step2_repeated_measures_checkbox",
                    help="Check this box if the same subjects are measured more than once (e.g., before and after treatment, or at multiple time points). Example: Measuring blood pressure at baseline, 1 month, and 3 months in the same patients.",
                    disabled=True
                )
        else:
            # Ambiguous case: show enabled main checkbox
            repeated_measures = st.checkbox(
                "Repeated Measures?",
                value=False,
                key="step2_repeated_measures_checkbox",
                help="Check this box if the same subjects are measured more than once (e.g., before and after treatment, or at multiple time points). Example: Measuring blood pressure at baseline, 1 month, and 3 months in the same patients."
            )
        covariates = st.text_input(
            "Covariates (comma-separated, optional)",
            key="step2_covariates_input",
            help="Other variables you want to adjust for in the analysis (e.g., age, sex, baseline disease severity). Example: 'age, sex, baseline LDL'. [Learn more](https://en.wikipedia.org/wiki/Covariate)"
        )
        st.markdown("**When ready, continue to model selection and data gathering.**")
        if st.button("Continue to Model Selection", key="step2_continue_model_inference"):
            st.session_state["moa"] = moa
            st.session_state["endpoint"] = endpoint
            st.session_state["outcome_type"] = outcome_type
            st.session_state["n_groups"] = n_groups
            st.session_state["repeated_measures"] = repeated_measures
            st.session_state["covariates"] = covariates
            st.session_state["progress_step"] = 3

# --- Step 3: Model Inference & Template Generation ---
st.header("📊 Step 3: Model Inference & Data Template")
with st.expander("Step 3: Model Inference & Data Template", expanded=st.session_state.get("progress_step", 1) == 3):
    if st.session_state.get("progress_step", 1) > 3:
        st.write("**Recommended Model:**", st.session_state.get("selected_model_id", "N/A"))
        st.write("**Required Data Columns:**", st.session_state.get("required_columns", []))
        if st.button("Edit Step 3"):
            reset_steps_from(3)
    elif st.session_state.get("progress_step", 1) == 3:
        from lib.test_recommender import recommend_statistical_test
        from lib.model_registry import get_model_by_id
        
        # Get all necessary parameters from session state
        design_type = st.session_state.get("design_type")
        n_groups = st.session_state.get("n_groups", 2)
        outcome_type = st.session_state.get("outcome_type")
        randomization = st.session_state.get("randomization")
        sample_size = st.session_state.get("n_per_arm")
        
        # Determine between/within factors based on design type
        if design_type == "Parallel Group":
            n_between = 1
            n_within = 0
            n_within_levels = []
        elif design_type == "Crossover":
            n_between = 1
            n_within = 1
            n_within_levels = [2]  # Typically two periods in crossover
        elif design_type == "Single Arm":
            n_between = 0
            n_within = 1
            n_within_levels = [2]  # Before/after comparison
        elif design_type == "Factorial":
            n_between = 2  # Assuming 2x2 factorial
            n_within = 0
            n_within_levels = []
        elif design_type == "Longitudinal":
            n_between = 1
            n_within = 1
            n_within_levels = [3]  # Assuming at least 3 time points
        else:
            n_between = 1
            n_within = 0
            n_within_levels = []
        
        # Get repeated measures and covariates status
        has_repeated_measures = st.session_state.get("repeated_measures", False)
        has_covariates = bool(st.session_state.get("covariates"))
        
        # Get model recommendations
        recommended_model_id, alternative_model_ids = recommend_statistical_test(
            n_between=n_between,
            n_within=n_within,
            n_groups=n_groups,
            n_within_levels=n_within_levels,
            outcome_type=outcome_type,
            design_type=design_type,
            randomization=randomization,
            has_covariates=has_covariates,
            has_repeated_measures=has_repeated_measures,
            sample_size=sample_size
        )
        
        if recommended_model_id:
            best_model = get_model_by_id(recommended_model_id)
            st.session_state["selected_model_id"] = best_model.model_id
            
            # Display primary recommendation
            st.success(f"Recommended Model: **{best_model.label}**")
            st.markdown(f"**Description:** {best_model.description}")

            # --- Model Assumptions & Details ---
            test_key = get_canonical_test_key(best_model.label)
            test_expl = explain_test(test_key)
            st.info(
                f"**Test:** {test_expl.get('title', best_model.label)}\n\n"
                f"**Summary:** {test_expl.get('summary', 'N/A')}\n\n"
                f"**When to Use:** {test_expl.get('when_to_use', 'N/A')}\n\n"
                f"**Assumptions:** {test_expl.get('assumptions', 'N/A')}\n\n"
                f"**Limitations:** {test_expl.get('limitations', 'N/A')}\n\n"
                f"{'**Reference:** [Link](' + test_expl.get('reference', '#') + ')' if test_expl.get('reference') else ''}\n\n"
                f"**Software/Method:** {test_expl.get('software', 'N/A')}"
            )
            if test_key == best_model.label:
                st.warning(f"No canonical mapping for model label '{best_model.label}'. Please add it to get_canonical_test_key to ensure explanations are shown.")

            # Design Tags with label and explanation
            st.markdown("**Design Tags (model requirements/features):**")
            tag_explanations = {
                "parallel": "Parallel group design",
                "two-groups": "Exactly two groups",
                "multi-group": "Multiple groups",
                "continuous": "Continuous outcome",
                "independent": "Independent samples",
                "paired": "Paired/related samples",
                "longitudinal": "Longitudinal/repeated measures",
                "multi-timepoint": "Multiple time points",
                "repeated": "Repeated measures",
                "survival": "Survival/time-to-event analysis",
                "time-to-event": "Time-to-event outcome",
                "binary": "Binary outcome",
                "categorical": "Categorical outcome",
                "small-n": "Small sample size",
            }
            tag_cols = st.columns(4)
            for i, tag in enumerate(best_model.design_tags):
                expl = tag_explanations.get(tag, "")
                if expl:
                    html = f"- `{tag}`<br><span style='font-size:0.9em;color:#888;'>{expl}</span>"
                else:
                    html = f"- `{tag}`"
                tag_cols[i % 4].markdown(html, unsafe_allow_html=True)
            st.caption("Design tags describe the key requirements and features of the recommended model.")

            # Sample Size Requirements with label, explanation, and check/warning
            st.markdown("**Sample Size Requirements for this Model:**")
            st.caption("These are the minimum and recommended sample sizes for valid statistical inference with this model. Your selection in Step 1 is checked below.")
            user_n_per_arm = st.session_state.get("n_per_arm", "")
            user_n_val = 0
            if user_n_per_arm == "<20":
                user_n_val = 10
            elif user_n_per_arm == "20-50":
                user_n_val = 35
            elif user_n_per_arm == "51-100":
                user_n_val = 75
            elif user_n_per_arm == ">100":
                user_n_val = 150
            reqs = best_model.sample_size_requirements
            for req, value in reqs.items():
                # Check if user's sample size meets requirement
                meets = False
                if "min" in req and user_n_val >= value:
                    meets = True
                if "max" in req and user_n_val <= value:
                    meets = True
                if "recommended" in req and user_n_val >= value:
                    meets = True
                check = "✅" if meets else "⚠️"
                st.markdown(f"- {req.replace('_', ' ').title()}: {value} &nbsp; {check}")
            if not any(("min" in req and user_n_val >= value) or ("max" in req and user_val <= value) or ("recommended" in req and user_n_val >= value) for req, value in reqs.items()):
                st.warning("Your selected sample size may not meet the requirements for this model. Consider increasing your sample size for valid inference.")

            # --- Power Analysis Section for Continuous Outcomes ---
            if best_model.model_id in ["t_test", "one_way_anova"]:
                st.markdown("---")
                st.markdown("### 🧮 Power & Sample Size Analysis")
                st.caption("Estimate the minimum sample size needed for your desired power and effect size. These values will be used for both power analysis and, if you choose, for auto-generating example data in Step 4.")
                from statsmodels.stats.power import TTestIndPower, FTestAnovaPower
                # Get or set effect size and stddev in session state
                effect_size = st.session_state.get("power_effect_size", 0.5)
                stddev = st.session_state.get("power_stddev", 1.0)
                power = st.session_state.get("power_power", 0.8)
                alpha = st.session_state.get("power_alpha", 0.05)
                effect_size = st.number_input("Effect size (Cohen's d, standardized difference)", value=effect_size, step=0.1, key="power_effect_size")
                stddev = st.number_input("Standard deviation (for continuous outcome)", value=stddev, step=0.1, key="power_stddev")
                power = st.number_input("Desired power (e.g., 0.8 for 80%)", value=power, min_value=0.01, max_value=0.99, step=0.01, key="power_power")
                alpha = st.number_input("Significance level (alpha)", value=alpha, min_value=0.001, max_value=0.2, step=0.001, key="power_alpha")
                n_groups = st.session_state.get("n_groups", 2)
                if best_model.model_id == "t_test":
                    analysis = TTestIndPower()
                    required_n = analysis.solve_power(
                        effect_size=effect_size,
                        alpha=alpha,
                        power=power,
                        alternative='two-sided'
                    )
                    st.info(f"Required sample size per group for t-test: **{int(required_n)+1}**")
                elif best_model.model_id == "one_way_anova":
                    analysis = FTestAnovaPower()
                    required_n = analysis.solve_power(
                        effect_size=effect_size,
                        alpha=alpha,
                        power=power,
                        k_groups=n_groups
                    )
                    st.info(f"Required sample size per group for ANOVA: **{int(required_n)+1}**")
                # Store only the calculated required_n for Step 4
                st.session_state["power_required_n"] = int(required_n)+1
            else:
                st.info("Power/sample size analysis is not yet supported for this model type.\n\nFor binary, count, survival, or repeated measures models, power analysis requires additional assumptions (e.g., expected proportions, event rates, follow-up time, or variance components). If you need power analysis for these models, please consult a statistician or use specialized tools such as statsmodels, lifelines, or longpower (R). If you would like this feature added, let us know!")

            # Required data structure
            st.markdown("#### Required Data Structure (Columns)")
            columns = ["subject", "group", "outcome"]
            if n_within:
                columns.append("time")
            if st.session_state.get("covariates"):
                columns += [c.strip() for c in st.session_state["covariates"].split(",") if c.strip()]
            st.session_state["required_columns"] = columns
            st.write(columns)
            
            st.markdown("**Download the data template below, then continue to data collection.**")
            import pandas as pd
            template_df = pd.DataFrame(columns=columns)
            csv = template_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Data Template (CSV)",
                data=csv,
                file_name="trial_data_template.csv",
                mime="text/csv",
                key="step3_download_template"
            )
            
            if st.button("Continue to Data Collection", key="step3_continue_data_collection"):
                st.session_state["progress_step"] = 4
        else:
            st.error("No suitable model found for the selected design. Please revise your inputs.")
            st.markdown("""
            Common reasons for no model match:
            - Sample size too small for the selected design
            - Incompatible combination of design type and number of groups
            - Outcome type not supported by available models
            - Required randomization not specified
            """)

# --- Step 4: Data Collection ---
st.header("📁 Step 4: Data Collection")
with st.expander("Step 4: Data Collection", expanded=st.session_state.get("progress_step", 1) == 4):
    required_columns = st.session_state.get("required_columns", ["subject", "group", "outcome"])
    model_id = st.session_state.get("selected_model_id", "")
    simple_models = ["t_test", "one_way_anova"]
    data = None
    form_data_valid = False
    upload_data_valid = False
    sample_size_override = False

    # Ensure all needed variables are defined from session state
    design_type = st.session_state.get("design_type")
    outcome_type = st.session_state.get("outcome_type")
    n_groups = st.session_state.get("n_groups", 2)

    if st.session_state.get("progress_step", 1) > 4:
        st.write("**Data uploaded/entered.**")
        if st.button("Edit Step 4"):
            reset_steps_from(4)
    elif st.session_state.get("progress_step", 1) == 4:
        # Get sample size requirements from model
        model = get_model_by_id(model_id)
        min_per_group = model.sample_size_requirements.get("min_per_group", 0) if model else 0
        min_total = model.sample_size_requirements.get("min_total", 0) if model else 0
        sample_size_override = False

        # Step 4: Data Collection - Offer three mutually exclusive options
        st.markdown("#### Choose a Data Collection Method")
        data_entry_options = [
            "Auto-generate Example Data",
            "Fill Out a Form",
            "Upload Data Using the Template"
        ]
        can_autogen = (design_type == "Parallel Group" and outcome_type in ["continuous", "binary"])  # Simple case
        data_entry_labels = [
            "Auto-generate Example Data" + (" (coming soon for this design)" if not can_autogen else ""),
            "Fill Out a Form",
            "Upload Data Using the Template"
        ]
        data_entry_method = st.radio(
            "Select how you want to provide your data:",
            data_entry_labels,
            index=0 if can_autogen else 1,
            key="step4_data_entry_method"
        )

        if data_entry_method.startswith("Auto-generate") and not can_autogen:
            st.info("Auto-generation is not yet available for this design or outcome type. Please use manual entry or upload.")
        elif data_entry_method.startswith("Auto-generate"):
            # Prepopulate effect size and stddev from power analysis if available
            effect_size = st.session_state.get("power_effect_size", 1.0)
            stddev = st.session_state.get("power_stddev", 1.0)
            effect_size = st.number_input("Effect size (difference between group means or proportions)", value=effect_size, step=0.1, key="autogen_effect_size")
            stddev = st.number_input("Standard deviation (for continuous)", value=stddev, step=0.1, key="autogen_stddev")
            if (st.session_state.get("power_effect_size") is not None or st.session_state.get("power_stddev") is not None):
                st.info("Using effect size and standard deviation from power analysis above. Change if you want to generate data with different parameters.")
            n_per_group = st.session_state.get("n_per_arm", "<20>")
            n_per_group_val_from_step1 = 20 if n_per_group == "<20" else 35 if n_per_group == "20-50" else 75 if n_per_group == "51-100" else 150
            power_required_n = st.session_state.get("power_required_n", n_per_group_val_from_step1)
            # Use the power analysis value as default, but allow override
            n_per_group_val = st.number_input(
                "Number of subjects per group",
                min_value=2,
                value=power_required_n,
                step=1,
                key="autogen_n_per_group"
            )
            # Warn if user value is less than power analysis requirement
            if n_per_group_val < power_required_n:
                st.warning(f"This is less than the sample size required for your desired power (N={power_required_n}). You may proceed, but your study may be underpowered.")
            # Check sample size requirement for model
            sample_size_ok = (min_per_group and n_per_group_val >= min_per_group) or (min_total and n_groups * n_per_group_val >= min_total)
            if not sample_size_ok:
                st.warning(f"Your selected sample size per group ({n_per_group_val}) is below the minimum required for this model ({min_per_group if min_per_group else min_total//n_groups}). Consider increasing your sample size.")
                sample_size_override = st.checkbox("Override sample size requirement (not recommended)", value=False, key="step4_autogen_sample_size_override")
            else:
                sample_size_override = False
            if (sample_size_ok or sample_size_override) and st.button("Auto-generate Example Data", key="step4_autogen_data"):
                import numpy as np
                import pandas as pd
                np.random.seed(42)
                rows = []
                for group in range(1, n_groups+1):
                    if outcome_type == "continuous":
                        mean = (group-1) * effect_size
                        outcomes = np.random.normal(loc=mean, scale=stddev, size=n_per_group_val)
                    elif outcome_type == "binary":
                        p = 0.5 + (group-1)*effect_size/2  # crude mapping
                        outcomes = np.random.binomial(1, min(max(p,0),1), size=n_per_group_val)
                    else:
                        outcomes = np.zeros(n_per_group_val)
                    for i, val in enumerate(outcomes):
                        rows.append({"subject": f"S{group}_{i+1}", "group": f"Group{group}", "outcome": val})
                auto_df = pd.DataFrame(rows)
                st.session_state["data"] = auto_df
                st.success("Auto-generated example data:")
                st.dataframe(auto_df)

        # Option 2: Fill Out a Form
        elif data_entry_method.startswith("Fill Out a Form"):
            if model_id in simple_models:
                st.markdown("#### Enter Data Manually (for small/simple datasets)")
                n_rows = st.number_input("Number of rows (subjects)", min_value=2, max_value=100, value=6, step=1, key="step4_n_rows_input")
                manual_df = pd.DataFrame({col: ["" for _ in range(n_rows)] for col in required_columns})
                for i in range(n_rows):
                    cols = st.columns(len(required_columns))
                    for j, col in enumerate(required_columns):
                        manual_df.at[i, col] = cols[j].text_input(f"{col} (row {i+1})", key=f"step4_{col}_{i}")
                # Check sample size requirement for manual entry
                sample_size_ok = (min_per_group and n_rows >= min_per_group) or (min_total and n_rows >= min_total)
                if not sample_size_ok:
                    st.warning(f"Your entered number of rows ({n_rows}) is below the minimum required for this model ({min_per_group if min_per_group else min_total}). Consider increasing your sample size.")
                    sample_size_override = st.checkbox("Override sample size requirement (not recommended)", value=False, key="step4_manual_sample_size_override")
                else:
                    sample_size_override = False
                if (sample_size_ok or sample_size_override) and st.button("Use Entered Data", key="step4_use_entered_data"):
                    try:
                        manual_df["outcome"] = pd.to_numeric(manual_df["outcome"])
                        data = manual_df
                        form_data_valid = True
                        st.session_state["data"] = data
                        st.success("Manual data entry accepted.")
                        st.dataframe(data)
                    except Exception as e:
                        st.error(f"Error in manual data: {e}")
            else:
                st.info("Manual entry is only available for simple models (t-test, one-way ANOVA). For other models, please upload your data using the template.")

        # Option 3: Upload Data Using the Template
        elif data_entry_method.startswith("Upload Data"):
            st.markdown("#### Upload Data Using the Template")
            import pandas as pd
            template_df = pd.DataFrame(columns=required_columns)
            csv = template_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Data Template (CSV)",
                data=csv,
                file_name="trial_data_template.csv",
                mime="text/csv",
                key="step4_download_template"
            )
            uploaded_file = st.file_uploader("Upload trial dataset (CSV)", type=["csv"], key="step4_data_upload")
            upload_data_valid = False
            if uploaded_file:
                try:
                    upload_df = pd.read_csv(uploaded_file)
                    missing_cols = [col for col in required_columns if col not in upload_df.columns]
                    if missing_cols:
                        st.error(f"Missing required columns: {missing_cols}")
                    else:
                        # Check sample size requirement for upload
                        group_col = "group" if "group" in upload_df.columns else required_columns[1]
                        group_counts = upload_df[group_col].value_counts()
                        sample_size_ok = (min_per_group and all(group_counts >= min_per_group)) or (min_total and len(upload_df) >= min_total)
                        if not sample_size_ok:
                            st.warning(f"One or more groups in your uploaded data have fewer rows than the minimum required for this model ({min_per_group if min_per_group else min_total}). Consider increasing your sample size.")
                            sample_size_override = st.checkbox("Override sample size requirement (not recommended)", value=False, key="step4_upload_sample_size_override")
                        else:
                            sample_size_override = False
                        if sample_size_ok or sample_size_override:
                            st.success("Data uploaded and validated.")
                            st.dataframe(upload_df.head())
                            data = upload_df
                            upload_data_valid = True
                            st.session_state["data"] = data
                except Exception as e:
                    st.error(f"Error reading uploaded file: {e}")

        # Only show the continue button if a data entry method has been selected
        if (form_data_valid or upload_data_valid or ("data" in st.session_state and st.session_state["data"] is not None and not st.session_state["data"].empty)) and st.button("Continue to Step 5: Review & Confirm", key="step4_continue_review_confirm"):
            st.session_state["progress_step"] = 5

# --- Step 5: Run Statistical Analysis ---
st.header("✅ Step 5: Review & Confirm & Run Statistical Analysis")
if st.session_state.get("progress_step", 1) == 5:
    with st.expander("Step 5: Review & Confirm & Run Statistical Analysis", expanded=True):
        st.header("Run Statistical Analysis")
        st.success("Ready to run analysis with the confirmed configuration.")

        # --- Model Assumptions, Checks & Details ---
        from lib.explainer.explain_test_selection import explain_test
        model_id = st.session_state.get("selected_model_id")
        best_model = get_model_by_id(model_id) if model_id else None
        test_key = get_canonical_test_key(best_model.label if best_model else model_id)
        test_expl = explain_test(test_key)
        results = st.session_state.get("analysis_results", {})
        with st.expander("ℹ️ Model Assumptions, Checks & Details", expanded=True):
            st.markdown(f"**Test:** {test_expl.get('title', model_id)}")
            st.markdown(f"**Summary:** {test_expl.get('summary', 'N/A')}")
            st.markdown(f"**When to Use:** {test_expl.get('when_to_use', 'N/A')}")
            st.markdown(f"**Assumptions:** {test_expl.get('assumptions', 'N/A')}")
            st.markdown(f"**Limitations:** {test_expl.get('limitations', 'N/A')}")
            st.markdown(f"**Reference:** [Link]({test_expl.get('reference', '#')})" if test_expl.get('reference') else "")
            st.markdown(f"**Software/Method:** {test_expl.get('software', 'N/A')}")
            if test_key == (best_model.label if best_model else model_id):
                st.warning(f"No canonical mapping for model label '{best_model.label if best_model else model_id}'. Please add it to get_canonical_test_key to ensure explanations are shown.")
            # Show actual assumption check results if available
            if 'Assumptions' in results:
                st.markdown("---")
                st.markdown("**Assumption Check Results:**")
                assumptions = results['Assumptions']
                if 'Shapiro-Wilk' in assumptions:
                    p = assumptions['Shapiro-Wilk'][1]
                    st.markdown(f"- Shapiro-Wilk test for normality: p = {p:.4g} ({'PASS' if p >= 0.05 else 'FAIL'})")
                if 'Levene' in assumptions:
                    p = assumptions['Levene'][1]
                    st.markdown(f"- Levene's test for equal variances: p = {p:.4g} ({'PASS' if p >= 0.05 else 'FAIL'})")
                if 'Mauchly-Sphericity' in assumptions:
                    p = assumptions['Mauchly-Sphericity'].get('p_value', None)
                    if p is not None:
                        st.markdown(f"- Mauchly's test for sphericity: p = {p:.4g} ({'PASS' if p >= 0.05 else 'FAIL'})")
                if any((('Shapiro-Wilk' in assumptions and assumptions['Shapiro-Wilk'][1] < 0.05),
                        ('Levene' in assumptions and assumptions['Levene'][1] < 0.05),
                        ('Mauchly-Sphericity' in assumptions and assumptions['Mauchly-Sphericity'].get('p_value', 1) < 0.05))):
                    st.warning("One or more assumptions failed. An alternative test or correction was used where appropriate. See results below.")

        # --- Methods & Software ---
        import sys
        import statsmodels, scipy, pandas, numpy
        try:
            import lifelines
            lifelines_version = lifelines.__version__
        except ImportError:
            lifelines_version = None
        with st.expander("🛠️ Methods & Software", expanded=False):
            st.markdown(f"**Main Statistical Test:** {test_expl.get('title', model_id)} ([reference]({test_expl.get('reference', '#')}))")
            st.markdown(f"**Power/Sample Size Calculation:** " + ("Closed-form formula for t-test/ANOVA ([statsmodels documentation](https://www.statsmodels.org/stable/power.html))" if best_model and best_model.model_id in ['t_test', 'one_way_anova'] else "N/A or not available for this model."))
            st.markdown("**Python Packages Used:**")
            st.markdown(f"- statsmodels {statsmodels.__version__}")
            st.markdown(f"- scipy {scipy.__version__}")
            st.markdown(f"- pandas {pandas.__version__}")
            st.markdown(f"- numpy {numpy.__version__}")
            if lifelines_version:
                st.markdown(f"- lifelines {lifelines_version}")
            st.markdown(f"- Python {sys.version.split()[0]}")

        # --- Downloadable Analysis Report ---
        import io
        if st.button("Download Analysis Report (Markdown)"):
            report = io.StringIO()
            report.write(f"# Clinical Trial Analysis Report\n\n")
            report.write(f"## Trial Setup\n")
            for k in ["therapeutic_area", "phase", "design_type", "intervention_type", "control_type", "n_per_arm", "randomization"]:
                v = st.session_state.get(k, "N/A")
                report.write(f"- **{k.replace('_', ' ').title()}:** {v}\n")
            report.write(f"\n## Modeling Details\n")
            for k in ["moa", "endpoint", "outcome_type", "n_groups", "repeated_measures", "covariates"]:
                v = st.session_state.get(k, "N/A")
                report.write(f"- **{k.replace('_', ' ').title()}:** {v}\n")
            report.write(f"\n## Recommended Model/Test\n")
            report.write(f"- **Test:** {test_expl.get('title', model_id)}\n")
            report.write(f"- **Summary:** {test_expl.get('summary', 'N/A')}\n")
            report.write(f"- **When to Use:** {test_expl.get('when_to_use', 'N/A')}\n")
            report.write(f"- **Assumptions:** {test_expl.get('assumptions', 'N/A')}\n")
            report.write(f"- **Limitations:** {test_expl.get('limitations', 'N/A')}\n")
            report.write(f"- **Reference:** {test_expl.get('reference', 'N/A')}\n")
            report.write(f"- **Software/Method:** {test_expl.get('software', 'N/A')}\n")
            report.write(f"\n## Assumption Check Results\n")
            if 'Assumptions' in results:
                assumptions = results['Assumptions']
                if 'Shapiro-Wilk' in assumptions:
                    p = assumptions['Shapiro-Wilk'][1]
                    report.write(f"- Shapiro-Wilk test for normality: p = {p:.4g} ({'PASS' if p >= 0.05 else 'FAIL'})\n")
                if 'Levene' in assumptions:
                    p = assumptions['Levene'][1]
                    report.write(f"- Levene's test for equal variances: p = {p:.4g} ({'PASS' if p >= 0.05 else 'FAIL'})\n")
                if 'Mauchly-Sphericity' in assumptions:
                    p = assumptions['Mauchly-Sphericity'].get('p_value', None)
                    if p is not None:
                        report.write(f"- Mauchly's test for sphericity: p = {p:.4g} ({'PASS' if p >= 0.05 else 'FAIL'})\n")
                if any((('Shapiro-Wilk' in assumptions and assumptions['Shapiro-Wilk'][1] < 0.05),
                        ('Levene' in assumptions and assumptions['Levene'][1] < 0.05),
                        ('Mauchly-Sphericity' in assumptions and assumptions['Mauchly-Sphericity'].get('p_value', 1) < 0.05))):
                    report.write("- One or more assumptions failed. An alternative test or correction was used where appropriate.\n")
            else:
                report.write("- Assumption checks not available.\n")
            report.write(f"\n## Methods & Software\n")
            report.write(f"- Main Statistical Test: {test_expl.get('title', model_id)} ({test_expl.get('reference', 'N/A')})\n")
            report.write(f"- Power/Sample Size Calculation: " + ("Closed-form formula for t-test/ANOVA (https://www.statsmodels.org/stable/power.html)" if best_model and best_model.model_id in ['t_test', 'one_way_anova'] else "N/A or not available for this model.") + "\n")
            report.write(f"- Python Packages Used:\n")
            report.write(f"    - statsmodels {statsmodels.__version__}\n")
            report.write(f"    - scipy {scipy.__version__}\n")
            report.write(f"    - pandas {pandas.__version__}\n")
            report.write(f"    - numpy {numpy.__version__}\n")
            if lifelines_version:
                report.write(f"    - lifelines {lifelines_version}\n")
            report.write(f"    - Python {sys.version.split()[0]}\n")
            report.write(f"\n## Main Results\n")
            # Add main results if available
            if 'T-test' in results:
                report.write(f"### T-test Results\n{results['T-test']}\n")
            if 'ANOVA' in results:
                report.write(f"### ANOVA Table\n{results['ANOVA']}\n")
            if 'Effect Sizes' in results:
                report.write(f"### Effect Sizes\n{results['Effect Sizes']}\n")
            if 'Alternative Test' in results:
                report.write(f"### Alternative Test (Assumptions Failed)\n{results['Alternative Test']}\n")
            if 'Quality Report' in results:
                report.write(f"### Data Quality Report\n{results['Quality Report']}\n")
            # Add any warnings
            if 'Warnings' in results:
                report.write(f"### Warnings\n{results['Warnings']}\n")
            st.download_button(
                label="Download Markdown Report",
                data=report.getvalue(),
                file_name="clinical_trial_analysis_report.md",
                mime="text/markdown",
                key="step5_download_report"
            )

        # Get data and model info
        data = st.session_state.get("data")
        model_id = st.session_state.get("selected_model_id")
        required_columns = st.session_state.get("required_columns", [])
        if data is None or data.empty:
            st.warning("No data available. Please upload or generate data in Step 4.")
            st.stop()

        # Inline explanation for analysis step
        st.markdown("""
        <div style='background:#f5f7fa; border-radius:10px; padding:16px; margin-bottom:12px; border:1px solid #e0e4ea;'>
        <strong>What happens in this step?</strong><br>
        In this step, you will run the recommended statistical analysis on your uploaded or generated data. You can select which variables to use, choose how to handle missing values, and customize which results to display. The analysis will check statistical assumptions, compute descriptive statistics, run the main test, and provide effect sizes and visualizations. Downloadable results are available for further reporting.
        </div>
        """, unsafe_allow_html=True)

        # Pre-select outcome and factors if possible
        outcome = st.selectbox(
            "Select outcome variable",
            required_columns,
            index=required_columns.index("outcome") if "outcome" in required_columns else 0,
            help="Choose the main outcome variable to analyze. This should match your primary endpoint."
        )
        between_factors = st.multiselect(
            "Between-subject factors",
            required_columns,
            default=[col for col in required_columns if col == "group"],
            help="Select columns that define independent groups (e.g., treatment vs control)."
        )
        within_factors = st.multiselect(
            "Within-subject (repeated) factors",
            required_columns,
            default=[col for col in required_columns if col == "time"],
            help="Select columns that represent repeated measures (e.g., time points, conditions). Leave blank if not applicable."
        )

        # Imputation method
        imputation_method = st.selectbox(
            "Imputation Method for Missing Values",
            ["mean", "median", "none"],
            index=0,
            help="Choose how to handle missing values in your outcome variable. 'mean' and 'median' will fill missing values; 'none' will skip imputation."
        )

        # Toggles for which results to display
        st.markdown("#### Customize Results Display")
        show_descriptive = st.checkbox(
            "Show Descriptive Statistics",
            value=True,
            help="Display summary statistics (mean, median, SD, etc.) for each group or condition."
        )
        show_ls_means = st.checkbox(
            "Show LS Means Table & Plot",
            value=True,
            help="Show least squares means (adjusted group means) and a plot for visual comparison."
        )
        show_assumptions = st.checkbox(
            "Show Assumption Checks",
            value=True,
            help="Display results of normality, variance, and other statistical assumption checks."
        )
        show_plot = st.checkbox(
            "Show Main Plot",
            value=True,
            help="Show a main plot (boxplot or interaction plot) to visualize group differences."
        )

        st.markdown("""
        <div style='background:#e8f5e9; border-radius:8px; padding:12px; margin-bottom:10px; border:1px solid #c8e6c9;'>
        <strong>Tip:</strong> You can download tables and plots after running the analysis for use in your reports or presentations.
        </div>
        """, unsafe_allow_html=True)

        if st.button("Run Analysis", key="run_analysis_step5"):
            from lib.analysis import AnalysisOrchestrator
            import io
            import warnings
            try:
                orchestrator = AnalysisOrchestrator(model_id, outcome, between_factors, within_factors)
                warning_buffer = io.StringIO()
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    results, quality_report = orchestrator.run_pipeline(data, imputation_method=imputation_method)
                    for warn in w:
                        if not issubclass(warn.category, (DeprecationWarning, PendingDeprecationWarning)):
                            warning_buffer.write(f"\n{warn.category.__name__}: {warn.message}\n\n")
                st.success("✅ Model run complete.")

                # Inline explanation for results
                st.markdown("""
                <div style='background:#f5f7fa; border-radius:10px; padding:14px; margin-bottom:10px; border:1px solid #e0e4ea;'>
                <strong>How to interpret your results:</strong><br>
                - <b>Assumption Checks</b>: Ensure your data meets the requirements for the selected test. If assumptions fail, consider alternative tests.<br>
                - <b>Descriptive Statistics</b>: Review group means, medians, and variability.<br>
                - <b>LS Means</b>: Adjusted means for each group, useful for unbalanced designs.<br>
                - <b>Main Plot</b>: Visualize group differences or trends.<br>
                - <b>Effect Sizes</b>: Quantify the magnitude of differences.<br>
                - <b>Test Results</b>: Main statistical test results (t-test, ANOVA, etc.).<br>
                - <b>Post-Hoc</b>: Pairwise comparisons if multiple groups.<br>
                - <b>Alternative Test</b>: Used if assumptions are violated.<br>
                - <b>Quality Report</b>: Notes on data quality and imputation.<br>
                </div>
                """, unsafe_allow_html=True)

                # Display results
                if show_assumptions and "Assumptions" in results:
                    st.markdown("### Assumption Checks")
                    st.write(results["Assumptions"])

                if show_descriptive and "Descriptive Stats" in results:
                    st.markdown("### Descriptive Statistics")
                    st.dataframe(results["Descriptive Stats"])
                    if "Exports" in results and "Descriptive Stats CSV" in results["Exports"]:
                        st.download_button(
                            label="Download Descriptive Stats (CSV)",
                            data=results["Exports"]["Descriptive Stats CSV"],
                            file_name="descriptive_stats.csv",
                            mime="text/csv"
                        )

                if show_ls_means and "LS Means" in results:
                    st.markdown("### LS Means Table")
                    st.dataframe(results["LS Means"])
                    if "Exports" in results and "LS Means CSV" in results["Exports"]:
                        st.download_button(
                            label="Download LS Means (CSV)",
                            data=results["Exports"]["LS Means CSV"],
                            file_name="ls_means.csv",
                            mime="text/csv"
                        )
                    if "LS Means Plot" in results:
                        st.markdown("### LS Means Plot")
                        st.pyplot(results["LS Means Plot"])
                        if "Exports" in results and "LS Means Plot PNG" in results["Exports"]:
                            st.download_button(
                                label="Download LS Means Plot (PNG)",
                                data=results["Exports"]["LS Means Plot PNG"],
                                file_name="ls_means_plot.png",
                                mime="image/png"
                            )

                if show_plot and "Plot" in results:
                    st.markdown("### Main Plot")
                    st.pyplot(results["Plot"])
                    if "Exports" in results and "Plot PNG" in results["Exports"]:
                        st.download_button(
                            label="Download Main Plot (PNG)",
                            data=results["Exports"]["Plot PNG"],
                            file_name="main_plot.png",
                            mime="image/png"
                        )

                # Effect Sizes
                if "Effect Sizes" in results:
                    st.markdown("### Effect Sizes")
                    st.write(results["Effect Sizes"])

                # Statistical Test Results
                if "T-test" in results:
                    st.markdown("### T-test Results")
                    st.write(results["T-test"])
                if "ANOVA" in results:
                    st.markdown("### ANOVA Table")
                    st.dataframe(results["ANOVA"])
                if "Post-Hoc" in results:
                    st.markdown("### Post-Hoc (Tukey HSD)")
                    posthoc = results["Post-Hoc"]
                    if hasattr(posthoc, 'summary'):
                        st.text(posthoc.summary())
                    elif hasattr(posthoc, '_results_table'):
                        import pandas as pd
                        st.dataframe(pd.DataFrame(posthoc._results_table.data[1:], columns=posthoc._results_table.data[0]))
                    else:
                        st.write(posthoc)
                if "Alternative Test" in results:
                    st.markdown("### Alternative Test (Assumptions Failed)")
                    st.write(results["Alternative Test"])

                # Quality Report
                if quality_report:
                    st.markdown("### Data Quality Report")
                    for key, value in quality_report.items():
                        st.write(f"{key}: {value}")

                # Warnings
                warnings_text = warning_buffer.getvalue()
                if warnings_text:
                    st.markdown(
                        f'<div style="background:#fff3cd; color:#856404; border:1px solid #ffeeba; border-radius:8px; padding:1em; margin:1em 0;">'
                        f'<strong>Statistical Warnings:</strong><br><pre style="white-space:pre-wrap;">{warnings_text}\n</pre>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"❌ Model execution failed: {str(e)}")