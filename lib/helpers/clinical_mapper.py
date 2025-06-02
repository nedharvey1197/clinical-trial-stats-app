import pandas as pd

# Canonical mapping table
clinical_context_rows = [
    # Oncology
    ["Oncology", "Drug", "PD-1 Inhibitor", "Overall Response Rate", True],
    ["Oncology", "Drug", "Kinase Inhibitor", "Progression-Free Survival", True],
    ["Oncology", "Drug", "Chemotherapy", "Overall Survival", True],
    ["Oncology", "Biologic", "CAR-T", "Progression-Free Survival", True],
    ["Oncology", "Biologic", "Monoclonal Antibody", "Overall Response Rate", True],
    ["Oncology", "Radiation", "External Beam", "Local Tumor Control", True],
    ["Oncology", "Radiation", "Brachytherapy", "Local Control Rate", True],
    ["Oncology", "Behavioral", "Psychosocial Support", "QoL Score", False],
    ["Oncology", "Device", "Radiation Device", "Treatment Accuracy", True],
    
    # Cardiology
    ["Cardiology", "Drug", "Beta Blocker", "Blood Pressure Change", True],
    ["Cardiology", "Drug", "ACE Inhibitor", "Blood Pressure Change", True],
    ["Cardiology", "Drug", "Statin", "LDL Reduction", True],
    ["Cardiology", "Device", "Stent", "Event-Free Survival", True],
    ["Cardiology", "Device", "Pacemaker", "Device Function", True],
    ["Cardiology", "Device", "ICD", "Shock Frequency", True],
    ["Cardiology", "Procedure", "Bypass Surgery", "Major Adverse Cardiac Events", True],
    ["Cardiology", "Procedure", "Ablation", "Arrhythmia Recurrence", True],
    ["Cardiology", "Biologic", "Stem Cell Therapy", "Ejection Fraction", True],
    
    # Neurology
    ["Neurology", "Drug", "SSRI", "PHQ-9 Score", True],
    ["Neurology", "Drug", "Antiepileptic", "Seizure Frequency", True],
    ["Neurology", "Drug", "Dopamine Agonist", "Motor Score", True],
    ["Neurology", "Device", "Deep Brain Stimulator", "Seizure Frequency", True],
    ["Neurology", "Device", "VNS", "Seizure Frequency", True],
    ["Neurology", "Device", "Spinal Cord Stimulator", "Pain Score", True],
    ["Neurology", "Behavioral", "Cognitive Therapy", "Cognitive Score", True],
    ["Neurology", "Behavioral", "Physical Therapy", "Motor Function", True],
    ["Neurology", "Biologic", "Gene Therapy", "Disease Progression", True],
    
    # Immunology
    ["Immunology", "Biologic", "TNF Inhibitor", "ACR20 Response", True],
    ["Immunology", "Biologic", "IL-17 Inhibitor", "PASI Score", True],
    ["Immunology", "Drug", "JAK Inhibitor", "Symptom Score", True],
    ["Immunology", "Drug", "DMARD", "Disease Activity Score", True],
    ["Immunology", "Behavioral", "Dietary Intervention", "Inflammation Markers", True],
    ["Immunology", "Biologic", "B-cell Depletion", "Autoantibody Levels", True],
    
    # Infectious Disease
    ["Infectious Disease", "Vaccine", "mRNA", "Seroconversion Rate", True],
    ["Infectious Disease", "Vaccine", "Viral Vector", "Seroconversion Rate", True],
    ["Infectious Disease", "Drug", "Protease Inhibitor", "Viral Load Reduction", True],
    ["Infectious Disease", "Drug", "Antibiotic", "Bacterial Clearance", True],
    ["Infectious Disease", "Biologic", "Monoclonal Antibody", "Viral Load", True],
    
    # Endocrinology
    ["Endocrinology", "Drug", "GLP-1 Agonist", "HbA1c Change", True],
    ["Endocrinology", "Drug", "SGLT2 Inhibitor", "Fasting Glucose", True],
    ["Endocrinology", "Drug", "Insulin", "Blood Glucose", True],
    ["Endocrinology", "Device", "Insulin Pump", "HbA1c Change", True],
    ["Endocrinology", "Device", "CGM", "Time in Range", True],
    ["Endocrinology", "Behavioral", "Dietary Intervention", "Weight Change", True],
    
    # Psychiatry
    ["Psychiatry", "Drug", "SSRI", "Depression Score", True],
    ["Psychiatry", "Drug", "Antipsychotic", "PANSS Score", True],
    ["Psychiatry", "Drug", "Mood Stabilizer", "Mania Score", True],
    ["Psychiatry", "Behavioral", "CBT", "Anxiety Score", True],
    ["Psychiatry", "Behavioral", "DBT", "Borderline Symptoms", True],
    ["Psychiatry", "Device", "TMS", "Depression Score", True],
    
    # Respiratory
    ["Respiratory", "Drug", "Bronchodilator", "FEV1 Change", True],
    ["Respiratory", "Drug", "Corticosteroid", "Exacerbation Rate", True],
    ["Respiratory", "Device", "Inhaler", "FEV1 Change", True],
    ["Respiratory", "Device", "CPAP", "AHI Score", True],
    ["Respiratory", "Behavioral", "Pulmonary Rehabilitation", "Exercise Capacity", True],
    
    # Gastroenterology
    ["Gastroenterology", "Drug", "PPI", "Symptom Score", True],
    ["Gastroenterology", "Drug", "Biologic", "Endoscopic Score", True],
    ["Gastroenterology", "Device", "Endoscope", "Polyp Detection", True],
    ["Gastroenterology", "Behavioral", "Dietary Intervention", "Symptom Score", True],
    
    # Dermatology
    ["Dermatology", "Drug", "Topical Corticosteroid", "PASI Score", True],
    ["Dermatology", "Drug", "Biologic", "PASI Score", True],
    ["Dermatology", "Device", "Light Therapy", "Lesion Count", True],
    ["Dermatology", "Behavioral", "Photoprotection", "UV Exposure", True],
    
    # Ophthalmology
    ["Ophthalmology", "Drug", "Anti-VEGF", "Visual Acuity", True],
    ["Ophthalmology", "Drug", "Prostaglandin", "IOP Change", True],
    ["Ophthalmology", "Device", "IOL", "Visual Acuity", True],
    ["Ophthalmology", "Device", "Glaucoma Drainage", "IOP Change", True],
    
    # Rare Diseases
    ["Rare Diseases", "Drug", "Enzyme Replacement", "Biomarker Level", True],
    ["Rare Diseases", "Drug", "Substrate Reduction", "Disease Progression", True],
    ["Rare Diseases", "Biologic", "Gene Therapy", "Disease Progression", True],
    ["Rare Diseases", "Device", "Assistive Device", "Function Score", True]
]

# Default intervention types by therapeutic area
therapeutic_area_defaults = {
    "Oncology": ["Drug", "Biologic", "Radiation", "Device", "Behavioral"],
    "Cardiology": ["Drug", "Device", "Procedure", "Biologic"],
    "Neurology": ["Drug", "Device", "Behavioral", "Biologic"],
    "Immunology": ["Biologic", "Drug", "Behavioral"],
    "Infectious Disease": ["Vaccine", "Drug", "Biologic"],
    "Endocrinology": ["Drug", "Device", "Behavioral"],
    "Psychiatry": ["Drug", "Behavioral", "Device"],
    "Respiratory": ["Drug", "Device", "Behavioral"],
    "Gastroenterology": ["Drug", "Device", "Behavioral"],
    "Dermatology": ["Drug", "Device", "Behavioral"],
    "Ophthalmology": ["Drug", "Device"],
    "Rare Diseases": ["Drug", "Biologic", "Device"]
}

# Default MoAs by intervention type and therapeutic area
moa_defaults = {
    "Oncology": {
        "Drug": ["PD-1 Inhibitor", "Kinase Inhibitor", "Chemotherapy"],
        "Biologic": ["CAR-T", "Monoclonal Antibody"],
        "Device": ["Radiation Device"],
        "Behavioral": ["Psychosocial Support"]
    },
    "Cardiology": {
        "Drug": ["Beta Blocker", "ACE Inhibitor", "Statin"],
        "Device": ["Stent", "Pacemaker", "ICD"],
        "Procedure": ["Bypass Surgery", "Ablation"],
        "Biologic": ["Stem Cell Therapy"]
    }
    # ... (similar mappings for other therapeutic areas)
}

# Default endpoints by therapeutic area and intervention type
endpoint_defaults = {
    "Oncology": {
        "Drug": ["Overall Response Rate", "Progression-Free Survival", "Overall Survival"],
        "Biologic": ["Progression-Free Survival", "Overall Response Rate"],
        "Device": ["Local Control Rate", "Complication Rate", "Treatment Accuracy"],
        "Behavioral": ["QoL Score", "Symptom Burden"]
    },
    "Cardiology": {
        "Drug": ["Blood Pressure Change", "LDL Reduction", "Event-Free Survival"],
        "Device": [
            "Major Adverse Cardiac Events",  # MACE
            "Event-Free Survival",
            "Target Lesion Revascularization",
            "Stent Thrombosis",
            "Restenosis Rate",
            "Shock Frequency",  # for ICD
            "Device Function"   # for pacemaker
        ],
        "Procedure": ["Major Adverse Cardiac Events", "Arrhythmia Recurrence"],
        "Biologic": ["Ejection Fraction", "Cardiac Function"]
    },
    "Neurology": {
        "Drug": ["PHQ-9 Score", "Seizure Frequency", "Motor Score"],
        "Device": ["Seizure Frequency", "Pain Score"],
        "Behavioral": ["Cognitive Score", "Motor Function"],
        "Biologic": ["Disease Progression"]
    },
    "Immunology": {
        "Biologic": ["ACR20 Response", "PASI Score", "Autoantibody Levels"],
        "Drug": ["Symptom Score", "Disease Activity Score"],
        "Behavioral": ["Inflammation Markers", "Patient-Reported Outcomes"]
    },
    "Infectious Disease": {
        "Vaccine": ["Seroconversion Rate", "Immunogenicity"],
        "Drug": ["Viral Load Reduction", "Bacterial Clearance"],
        "Biologic": ["Viral Load"]
    },
    "Endocrinology": {
        "Drug": ["HbA1c Change", "Fasting Glucose", "Blood Glucose"],
        "Device": ["HbA1c Change", "Time in Range"],
        "Behavioral": ["Weight Change"]
    },
    "Psychiatry": {
        "Drug": ["Depression Score", "PANSS Score", "Mania Score"],
        "Behavioral": ["Anxiety Score", "Borderline Symptoms"],
        "Device": ["Depression Score"]
    },
    "Respiratory": {
        "Drug": ["FEV1 Change", "Exacerbation Rate"],
        "Device": ["FEV1 Change", "AHI Score"],
        "Behavioral": ["Exercise Capacity"]
    },
    "Gastroenterology": {
        "Drug": ["Symptom Score", "Endoscopic Score"],
        "Device": ["Polyp Detection"],
        "Behavioral": ["Symptom Score"]
    },
    "Dermatology": {
        "Drug": ["PASI Score"],
        "Biologic": ["PASI Score"],
        "Device": ["Lesion Count"],
        "Behavioral": ["UV Exposure"]
    },
    "Ophthalmology": {
        "Drug": ["Visual Acuity", "IOP Change"],
        "Device": ["Visual Acuity", "IOP Change"]
    },
    "Rare Diseases": {
        "Drug": ["Biomarker Level", "Disease Progression"],
        "Biologic": ["Disease Progression"],
        "Device": ["Function Score"]
    }
}

columns = ["therapeutic_area", "intervention_type", "moa", "endpoint", "default"]
df_clinical_context = pd.DataFrame(clinical_context_rows, columns=columns)

# Filtering logic
def get_valid_intervention_types(area):
    """Get valid intervention types for a therapeutic area with defaults."""
    if area in therapeutic_area_defaults:
        return therapeutic_area_defaults[area]
    # Fallback to database values if not in defaults
    options = df_clinical_context[df_clinical_context["therapeutic_area"] == area]["intervention_type"].unique().tolist()
    return sorted(set(options + ["Other..."]))

def get_valid_moas(area, intervention):
    """Get valid MoAs for a therapeutic area and intervention type with defaults."""
    if area in moa_defaults and intervention in moa_defaults[area]:
        return moa_defaults[area][intervention]
    # Fallback to database values if not in defaults
    options = df_clinical_context[
        (df_clinical_context["therapeutic_area"] == area) &
        (df_clinical_context["intervention_type"] == intervention)
    ]["moa"].unique().tolist()
    return sorted(set(options + ["Other..."]))

def get_valid_endpoints(area, intervention, moa, phase=None):
    """Get valid endpoints for a therapeutic area, intervention type, and MoA with phase-prioritized ordering."""
    endpoints = []
    if area in endpoint_defaults and intervention in endpoint_defaults[area]:
        endpoints = endpoint_defaults[area][intervention]
        # Optionally, reorder endpoints based on phase
        if phase == "Phase I":
            endpoints = sorted(endpoints, key=lambda x: "Safety" not in x)
        elif phase == "Phase II":
            endpoints = sorted(endpoints, key=lambda x: "Efficacy" not in x)
    # Fallback to database values if not in defaults
    if not endpoints:
        options = df_clinical_context[
            (df_clinical_context["therapeutic_area"] == area) &
            (df_clinical_context["intervention_type"] == intervention) &
            (df_clinical_context["moa"] == moa)
        ]["endpoint"].unique().tolist()
        endpoints = sorted(set(options + ["Other..."]))
    return endpoints

def get_default_endpoint(area, intervention, phase):
    """Get the most appropriate default endpoint based on therapeutic area, intervention type, and phase."""
    endpoints = get_valid_endpoints(area, intervention, moa=None, phase=phase)
    return endpoints[0] if endpoints else "Other..."

def get_default_moa(area, intervention):
    """Get the most appropriate default MoA based on therapeutic area and intervention type."""
    if area in moa_defaults and intervention in moa_defaults[area]:
        return moa_defaults[area][intervention][0]  # Return first default MoA
    
    # Fallback to database values
    options = df_clinical_context[
        (df_clinical_context["therapeutic_area"] == area) &
        (df_clinical_context["intervention_type"] == intervention)
    ]["moa"].unique().tolist()
    return options[0] if options else "Other..."