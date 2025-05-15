import pandas as pd
import numpy as np
import logging

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
    logger.info(f"Accessing canned example for model: {model}")
    if model not in canned_examples_with_desc:
        raise ValueError(f"No canned example found for model: {model}")
    return canned_examples_with_desc[model]

# Validate datasets on module load
validate_canned_examples(canned_examples_with_desc)