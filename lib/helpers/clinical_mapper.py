import pandas as pd

# Canonical mapping table
clinical_context_rows = [
    ["Oncology", "Drug", "PD-1 Inhibitor", "Overall Response Rate", True],
    ["Oncology", "Drug", "Kinase Inhibitor", "Progression-Free Survival", True],
    ["Oncology", "Biologic", "CAR-T", "Progression-Free Survival", True],
    ["Oncology", "Radiation", "External Beam", "Local Tumor Control", True],
    ["Oncology", "Behavioral", "Psychosocial Support", "QoL Score", False],
    ["Cardiology", "Drug", "Beta Blocker", "Blood Pressure Change", True],
    ["Cardiology", "Device", "Stent", "Event-Free Survival", True],
    ["Cardiology", "Procedure", "Bypass Surgery", "Major Adverse Cardiac Events", True],
    ["Neurology", "Drug", "SSRI", "PHQ-9 Score", True],
    ["Neurology", "Device", "Deep Brain Stimulator", "Seizure Frequency", True],
    ["Neurology", "Behavioral", "Cognitive Therapy", "Cognitive Score", True],
    ["Immunology", "Biologic", "TNF Inhibitor", "ACR20 Response", True],
    ["Immunology", "Drug", "JAK Inhibitor", "Symptom Score", True],
    ["Infectious Disease", "Vaccine", "mRNA", "Seroconversion Rate", True],
    ["Infectious Disease", "Drug", "Protease Inhibitor", "Viral Load Reduction", True],
    ["Endocrinology", "Drug", "GLP-1 Agonist", "HbA1c Change", True],
    ["Endocrinology", "Drug", "SGLT2 Inhibitor", "Fasting Glucose", True],
    ["Psychiatry", "Drug", "SSRI", "Depression Score", True],
    ["Psychiatry", "Behavioral", "CBT", "Anxiety Score", True]
]
columns = ["therapeutic_area", "intervention_type", "moa", "endpoint", "default"]
df_clinical_context = pd.DataFrame(clinical_context_rows, columns=columns)

# Filtering logic
def get_valid_intervention_types(area):
    options = df_clinical_context[df_clinical_context["therapeutic_area"] == area]["intervention_type"].unique().tolist()
    return sorted(set(options + ["Other..."]))

def get_valid_moas(area, intervention):
    options = df_clinical_context[
        (df_clinical_context["therapeutic_area"] == area) &
        (df_clinical_context["intervention_type"] == intervention)
    ]["moa"].unique().tolist()
    return sorted(set(options + ["Other..."]))

def get_valid_endpoints(area, intervention, moa):
    options = df_clinical_context[
        (df_clinical_context["therapeutic_area"] == area) &
        (df_clinical_context["intervention_type"] == intervention) &
        (df_clinical_context["moa"] == moa)
    ]["endpoint"].unique().tolist()
    return sorted(set(options + ["Other..."]))