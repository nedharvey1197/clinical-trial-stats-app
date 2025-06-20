import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

canned_examples_with_desc = {
    "T-test": {
        "data": pd.DataFrame({
            'Subject': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Drug': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
            'Outcome': [120, 115, 110, 105, 100, 130, 125, 120, 115, 110]
        }),
        "description": (
            "A Phase II randomized controlled trial (RCT) comparing the efficacy of a new "
            "antihypertensive drug (Drug A) versus placebo (Drug B) in patients with mild "
            "hypertension. The outcome is the reduction in systolic blood pressure (mmHg) "
            "after 8 weeks of treatment, with higher reductions indicating better efficacy. "
            "The trial aims to determine if Drug A significantly lowers blood pressure "
            "compared to placebo.\n"
            "This is an example of a simple two-arm parallel group design comparing two independent treatments."
        ),
        "metadata": {
            "outcome_type": "continuous",
            "sample_type": "independent",
            "paired": False,
            "normal": True,
            "small_n": True
        },
        "selector_metadata": {
            "model_id": "T_test",
            "label": "Simple two-group comparison (T-test)",
            "summary": "A trial comparing the efficacy of a new drug vs placebo in patients with mild hypertension.",
            "therapeutic_area": "Cardiovascular",
            "phase": "Phase II",
            "design_type": "Parallel Group",
            "intervention_type": "Small Molecule",
            "moa": "Vasodilation",
            "control_type": "Placebo",
            "endpoint": "Blood Pressure Reduction",
            "n_per_arm": "20–50",
            "randomization": "Randomized"
        }
    },
    "One-way ANOVA": {
        "data": pd.DataFrame({
            'Subject': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            'Drug': ['Placebo', 'Placebo', 'Placebo', 'Placebo', 'Placebo',
                     'Low Dose', 'Low Dose', 'Low Dose', 'Low Dose', 'Low Dose',
                     'High Dose', 'High Dose', 'High Dose', 'High Dose', 'High Dose'],
            'Outcome': [120, 115, 110, 105, 100, 130, 125, 120, 115, 110, 140, 135, 130, 125, 120]
        }),
        "description": (
            "A Phase II dose-ranging trial evaluating a new cholesterol-lowering drug in "
            "patients with hyperlipidemia. Patients are randomized to receive placebo, low "
            "dose, or high dose of the drug for 12 weeks. The outcome is the percentage "
            "reduction in LDL cholesterol levels (mg/dL), with larger reductions indicating "
            "better efficacy. The trial tests whether different doses lead to varying "
            "cholesterol reductions.\n"
            "This is an example of a between-subjects single-factor design evaluating dose-response."
        ),
        "metadata": {
            "outcome_type": "continuous",
            "sample_type": "independent",
            "paired": False,
            "normal": True,
            "small_n": False
        },
        "selector_metadata": {
            "model_id": "One_way_anova",
            "label": "Dose-ranging trial (One-way ANOVA)",
            "summary": "A trial comparing three dosage levels (placebo, low, high) of a cholesterol-lowering drug.",
            "therapeutic_area": "Cardiovascular",
            "phase": "Phase II",
            "design_type": "Parallel Group",
            "intervention_type": "Small Molecule",
            "moa": "Cholesterol Synthesis Inhibition",
            "control_type": "Placebo",
            "endpoint": "LDL Reduction",
            "n_per_arm": "20–50",
            "randomization": "Randomized"
        }
    },
    "Two-way ANOVA": {
        "data": pd.DataFrame({
            'Subject': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'Drug': ['A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'B'],
            'Age_Group': ['Young', 'Young', 'Young', 'Young', 'Young', 'Young',
                          'Old', 'Old', 'Old', 'Old', 'Old', 'Old'],
            'Outcome': [120, 115, 110, 130, 125, 120, 125, 120, 115, 135, 130, 125]
        }),
        "description": (
            "A Phase III trial investigating a new anti-diabetic drug (Drug A vs. Drug B) "
            "in patients with type 2 diabetes, stratified by age (Young: <50 years, Old: "
            "≥50 years). The outcome is the change in HbA1c levels (%) after 6 months, "
            "with lower values indicating better glycemic control. The trial assesses the "
            "main effects of drug type and age, as well as their interaction, on HbA1c "
            "reduction.\n"
            "This is an example of a two-factor between-subjects factorial design."
        ),
        "metadata": {
            "outcome_type": "continuous",
            "sample_type": "independent",
            "paired": False,
            "normal": True,
            "small_n": False
        },
        "selector_metadata": {
            "model_id": "Two_way_anova",
            "label": "Stratified comparison by age (Two-way ANOVA)",
            "summary": "A trial comparing Drug A vs B for diabetes, stratified by patient age (Young vs Old).",
            "therapeutic_area": "Endocrinology",
            "phase": "Phase III",
            "design_type": "Parallel Group",
            "intervention_type": "Small Molecule",
            "moa": "Insulin Sensitization",
            "control_type": "Active Comparator",
            "endpoint": "HbA1c Reduction",
            "n_per_arm": "51–100",
            "randomization": "Stratified Randomization"
        }
    },
    "Three-way ANOVA": {
        "data": pd.DataFrame({
            'Subject': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            'Drug': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
            'Age_Group': ['Young', 'Young', 'Old', 'Old', 'Young', 'Young', 'Old', 'Old', 'Young', 'Young', 'Old', 'Old', 'Young', 'Young', 'Old', 'Old'],
            'Site': ['1', '1', '1', '1', '2', '2', '2', '2', '1', '1', '1', '1', '2', '2', '2', '2'],
            'Outcome': [120, 122, 130, 128, 115, 117, 125, 123, 135, 133, 140, 138, 130, 128, 135, 133]
        }),
        "description": (
            "A multicenter Phase III trial evaluating a new pain relief medication (Drug A "
            "vs. Drug B) in patients with osteoarthritis, across two age groups (Young: "
            "<50 years, Old: ≥50 years) and two clinical sites (Site 1, Site 2). The "
            "outcome is the reduction in pain score (VAS, 0-100 scale) after 4 weeks, with "
            "larger reductions indicating better pain relief. The trial examines the effects "
            "of drug, age, site, and their interactions on pain reduction.\n"
            "This is an example of a three-factor between-subjects factorial design with multiple interacting conditions."
        ),
        "metadata": {
            "outcome_type": "continuous",
            "sample_type": "independent",
            "paired": False,
            "normal": True,
            "small_n": False
        },
        "selector_metadata": {
            "model_id": "three_way_anova",
            "label": "Multicenter stratified trial (Three-way ANOVA)",
            "summary": "A trial evaluating pain relief across drug, age group, and trial site.",
            "therapeutic_area": "Rheumatology",
            "phase": "Phase III",
            "design_type": "Factorial",
            "intervention_type": "Small Molecule",
            "moa": "Pain Modulation",
            "control_type": "Active Comparator",
            "endpoint": "Pain Score Reduction",
            "n_per_arm": "51–100",
            "randomization": "Stratified Randomization"
        }
    },
    "One-way Repeated Measures ANOVA": {
        "data": pd.DataFrame({
            'Subject': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'Time': [1, 2, 3, 1, 2, 3, 1, 2, 3],
            'Outcome': [120, 115, 110, 130, 125, 120, 140, 135, 130]
        }),
        "description": (
            "A Phase II single-arm trial assessing the effect of a new anti-inflammatory "
            "drug on C-reactive protein (CRP) levels (mg/L) in patients with rheumatoid "
            "arthritis. Measurements are taken at baseline (Time 1), 4 weeks (Time 2), and "
            "8 weeks (Time 3), with lower CRP levels indicating reduced inflammation. The "
            "trial evaluates the change in CRP over time within the same patients.\n"
            "This is an example of a within-subject repeated-measures design over time."
        ),
        "metadata": {
            "outcome_type": "continuous",
            "sample_type": "related",
            "paired": True,
            "normal": True,
            "small_n": False
        },
        "selector_metadata": {
            "model_id": "one_way_rm_anova",
            "label": "Single-arm longitudinal trial (One-way RM ANOVA)",
            "summary": "A within-subject trial measuring inflammation reduction over three timepoints.",
            "therapeutic_area": "Rheumatology",
            "phase": "Phase II",
            "design_type": "Single Arm",
            "intervention_type": "Biologic",
            "moa": "Inflammation Modulation",
            "control_type": "No Control",
            "endpoint": "CRP Reduction",
            "n_per_arm": "20–50",
            "randomization": "Non-Randomized"
        }
    },
    "Two-way Repeated Measures ANOVA": {
        "data": pd.DataFrame({
            'Subject': [1, 1, 1, 1, 2, 2, 2, 2],
            'Time': [1, 1, 2, 2, 1, 1, 2, 2],
            'Condition': ['Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard'],
            'Outcome': [120, 115, 110, 105, 130, 125, 120, 115]
        }),
        "description": (
            "A Phase II crossover trial evaluating a cognitive enhancer in healthy "
            "volunteers under two task conditions (Easy, Hard). Cognitive performance "
            "(reaction time in milliseconds, lower is better) is measured at two time "
            "points (Time 1: before treatment, Time 2: after treatment) for each condition. "
            "The trial assesses the effects of time and task difficulty on cognitive "
            "performance within subjects.\n"
            "This is an example of a repeated-measures within-subject design with multiple time and condition effects."
        ),
        "metadata": {
            "outcome_type": "continuous",
            "sample_type": "mixed",
            "paired": True,
            "normal": True,
            "small_n": True
        },
        "selector_metadata": {
            "model_id": "two_way_rm_anova",
            "label": "Crossover trial with repeated tasks (Two-way RM ANOVA)",
            "summary": "A study evaluating cognitive performance under two task conditions across time.",
            "therapeutic_area": "Neurology",
            "phase": "Phase II",
            "design_type": "Crossover",
            "intervention_type": "Small Molecule",
            "moa": "Cognitive Enhancement",
            "control_type": "Placebo",
            "endpoint": "Reaction Time",
            "n_per_arm": "<20",
            "randomization": "Randomized"
        }
    },
    "Three-way Repeated Measures ANOVA": {
        "data": pd.DataFrame({
            'Subject': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            'Drug': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A'],
            'Time': [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
            'Condition': ['Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard'],
            'Outcome': [120, 115, 110, 105, 130, 125, 120, 115, 125, 120, 115, 110]
        }),
        "description": (
            "A Phase II trial comparing two anxiolytic drugs (Drug A, Drug B) in patients "
            "with generalized anxiety disorder (GAD). Anxiety levels (HAM-A score, lower is "
            "better) are measured under two stress conditions (Easy, Hard) at two time "
            "points (Time 1: baseline, Time 2: 6 weeks post-treatment). The trial examines "
            "the effects of drug, stress condition, and time on anxiety reduction.\n"
            "This is an example of a three-way fully repeated-measures design with multiple within-subject factors."
        ),
        "metadata": {
            "outcome_type": "continuous",
            "sample_type": "related",
            "paired": True,
            "normal": True,
            "small_n": True
        },
        "selector_metadata": {
            "model_id": "three_way_rm_anova",
            "label": "Stress response over time (Three-way RM ANOVA)",
            "summary": "A repeated measures study examining anxiety under drug, time, and stress condition.",
            "therapeutic_area": "Psychiatry",
            "phase": "Phase II",
            "design_type": "Crossover",
            "intervention_type": "Small Molecule",
            "moa": "Anxiolytic",
            "control_type": "Active Comparator",
            "endpoint": "HAM-A Score",
            "n_per_arm": "<20",
            "randomization": "Randomized"
        }
    },
    "Mixed ANOVA (One Between, One Repeated)": {
        "data": pd.DataFrame({
            'Subject': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            'Drug': ['A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'B'],
            'Time': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            'Outcome': [120, 115, 110, 130, 125, 120, 125, 120, 115, 135, 130, 125]
        }),
        "description": (
            "A Phase III trial comparing a new antidepressant (Drug A) versus placebo "
            "(Drug B) in patients with major depressive disorder. Depression scores (MADRS, "
            "lower is better) are measured at baseline (Time 1), 4 weeks (Time 2), and 8 "
            "weeks (Time 3). The trial assesses the effect of the drug over time on "
            "depression severity.\n"
            "This is an example of a mixed factorial design with one between- and one within-subject factor."
        ),
        "metadata": {
            "outcome_type": "continuous",
            "sample_type": "mixed",
            "paired": True,
            "normal": True,
            "small_n": True
        },
        "selector_metadata": {
            "model_id": "mixed_anova_1b1r",
            "label": "Treatment effect over time (Mixed ANOVA: 1 Between, 1 Repeated)",
            "summary": "A trial comparing antidepressant vs placebo, tracking changes in symptoms over time.",
            "therapeutic_area": "Psychiatry",
            "phase": "Phase III",
            "design_type": "Parallel Group",
            "intervention_type": "Small Molecule",
            "moa": "Antidepressant",
            "control_type": "Placebo",
            "endpoint": "MADRS Score",
            "n_per_arm": "20–50",
            "randomization": "Randomized"
        }
    },
    "Mixed ANOVA (Two Between, One Repeated)": {
        "data": pd.DataFrame({
            'Subject': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            'Drug': ['A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'B'],
            'Age_Group': ['Young', 'Young', 'Young', 'Young', 'Young', 'Young', 'Old', 'Old', 'Old', 'Old', 'Old', 'Old'],
            'Time': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            'Outcome': [120, 115, 110, 130, 125, 120, 125, 120, 115, 135, 130, 125]
        }),
        "description": (
            "A Phase III trial evaluating a new migraine treatment (Drug A vs. Drug B) in "
            "patients, stratified by age (Young: <50 years, Old: ≥50 years). Migraine "
            "frequency (number of attacks per month, lower is better) is measured at "
            "baseline (Time 1), 3 months (Time 2), and 6 months (Time 3). The trial "
            "investigates the effects of drug, age, and time on migraine frequency.\n"
            "This is an example of a mixed factorial design with multiple interacting between- and within-subject effects."
        ),
        "metadata": {
            "outcome_type": "continuous",
            "sample_type": "mixed",
            "paired": True,
            "normal": True,
            "small_n": False
        },
        "selector_metadata": {
            "model_id": "mixed_anova_2b1r",
            "label": "Stratified repeated measures trial (Mixed ANOVA: 2 Between, 1 Repeated)",
            "summary": "A migraine study comparing drugs across age groups over multiple visits.",
            "therapeutic_area": "Neurology",
            "phase": "Phase III",
            "design_type": "Parallel Group",
            "intervention_type": "Small Molecule",
            "moa": "Migraine Prevention",
            "control_type": "Active Comparator",
            "endpoint": "Migraine Frequency",
            "n_per_arm": "51–100",
            "randomization": "Stratified Randomization"
        }
    },
    "Mixed ANOVA (One Between, Two Repeated)": {
        "data": pd.DataFrame({
            'Subject': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            'Drug': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'A', 'A', 'A', 'A'],
            'Time': [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
            'Condition': ['Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard', 'Easy', 'Hard'],
            'Outcome': [120, 115, 110, 105, 130, 125, 120, 115, 125, 120, 115, 110]
        }),
        "description": (
            "A Phase II trial comparing a new asthma medication (Drug A vs. Drug B) in "
            "patients with asthma. Lung function (FEV1, higher is better) is measured under "
            "two exercise conditions (Easy, Hard) at two time points (Time 1: baseline, "
            "Time 2: 4 weeks post-treatment). The trial assesses the effects of drug, "
            "exercise condition, and time on lung function.\n"
            "This is an example of a mixed repeated-measures design with one between- and two within-subject factors."
        ),
        "metadata": {
            "outcome_type": "continuous",
            "sample_type": "mixed",
            "paired": True,
            "normal": True,
            "small_n": True
        },
        "selector_metadata": {
            "model_id": "mixed_anova_1b2r",
            "label": "Dual-condition repeated trial (Mixed ANOVA: 1 Between, 2 Repeated)",
            "summary": "A crossover trial comparing drug effects on lung function across time and exercise intensity.",
            "therapeutic_area": "Pulmonology",
            "phase": "Phase II",
            "design_type": "Crossover",
            "intervention_type": "Small Molecule",
            "moa": "Bronchodilation",
            "control_type": "Active Comparator",
            "endpoint": "FEV1",
            "n_per_arm": "20–50",
            "randomization": "Randomized"
        }
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
        "description": (
            "A Phase III trial evaluating a new antipsychotic drug (Drug A vs. Drug B) in "
            "patients with schizophrenia, stratified by age (Young: <50 years, Old: ≥50 "
            "years). Symptom severity (PANSS score, lower is better) is measured under two "
            "social stress conditions (Easy, Hard) at two time points (Time 1: baseline, "
            "Time 2: 12 weeks post-treatment). The trial examines the effects of drug, age, "
            "stress condition, and time on symptom severity.\n"
            "This is an example of a full factorial mixed model with multiple interacting conditions."     
        ),
        "metadata": {
            "outcome_type": "continuous",
            "sample_type": "mixed",
            "paired": True,
            "normal": True,
            "small_n": False
        },
        "selector_metadata": {
            "model_id": "complex_mixed_anova",
            "label": "Multifactorial psychiatric trial (Complex Mixed ANOVA)",
            "summary": "A schizophrenia trial analyzing drug, age, stress, and time on symptom severity.",
            "therapeutic_area": "Psychiatry",
            "phase": "Phase III",
            "design_type": "Factorial",
            "intervention_type": "Small Molecule",
            "moa": "Antipsychotic",
            "control_type": "Active Comparator",
            "endpoint": "PANSS Score",
            "n_per_arm": "51–100",
            "randomization": "Stratified Randomization"
        }
    }
}

# Factor mappings
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

def validate_canned_examples(examples: dict) -> None:
    """
    Validate canned example datasets.

    Args:
        examples (dict): Dictionary of canned examples.

    Raises:
        ValueError: If a dataset is missing required columns or contains missing values.
    """
    required_cols = ['Subject', 'Outcome']
    for model, example in examples.items():
        data = example['data']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Dataset for {model} missing required columns: {missing_cols}")
        if data['Outcome'].isnull().any():
            raise ValueError(f"Dataset for {model} contains missing values in 'Outcome'.")

def get_example_metadata(model: str) -> Dict[str, Any]:
    meta = canned_examples_with_desc.get(model, {}).get("metadata", {})
    if not meta:
        logger.warning(f"No metadata found for example '{model}'")
    return meta

def get_example_description(model: str) -> str:
    desc = canned_examples_with_desc.get(model, {}).get("description", "")
    if not desc:
        logger.warning(f"No description found for example '{model}'")
    return desc

def get_canned_example(model: str) -> Dict[str, Any]:
    """
    Retrieve a canned example dataset.

    Args:
        model (str): Name of the statistical model (e.g., "T-test").

    Returns:
        Dict[str, Any]: Dictionary containing the dataset and its description.

    Raises:
        ValueError: If the model is not found in the canned examples.
    """
    # Normalize known model_id aliases to match dictionary keys
    model_key_map = {
        "t_test": "T-test",
        "one_way_anova": "One-way ANOVA",
        "two_way_anova": "Two-way ANOVA",
        "three_way_anova": "Three-way ANOVA",
        "one_way_rm_anova": "One-way Repeated Measures ANOVA",
        "two_way_rm_anova": "Two-way Repeated Measures ANOVA",
        "three_way_rm_anova": "Three-way Repeated Measures ANOVA",
        "mixed_anova_1b1r": "Mixed ANOVA (One Between, One Repeated)",
        "mixed_anova_2b1r": "Mixed ANOVA (Two Between, One Repeated)",
        "mixed_anova_1b2r": "Mixed ANOVA (One Between, Two Repeated)",
        "complex_mixed_anova": "Complex Mixed ANOVA",
    }

    normalized_model = model_key_map.get(model, model)
    logger.info(f"Accessing canned example for model: {normalized_model}")

    if normalized_model not in canned_examples_with_desc:
        raise ValueError(f"No canned example found for model: {model}")
    return canned_examples_with_desc[normalized_model]

def get_selector_metadata(model: str) -> Dict[str, Any]:
    """
    Retrieve the selector metadata for a given model.

    Args:
        model (str): Name of the statistical model (e.g., "T-test").

    Returns:
        Dict[str, Any]: Dictionary containing the selector metadata.

    Raises:
        ValueError: If the model is not found in the canned examples.
    """
    if model not in canned_examples_with_desc:
        raise ValueError(f"No selector metadata found for model: {model}")
    return canned_examples_with_desc[model].get("selector_metadata", {})

# Validate datasets on module load
validate_canned_examples(canned_examples_with_desc)