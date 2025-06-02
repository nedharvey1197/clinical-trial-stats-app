def map_clinical_context_to_meta(endpoint, design_type, phase, n_per_arm):
    """Infer statistical model metadata from clinical trial design."""
    endpoint_map = {
        "Overall Response Rate": "binary",
        "Progression-Free Survival": "time-to-event",
        "HbA1c Change": "continuous",
        "Blood Pressure Change": "continuous",
        "PHQ-9 Score": "continuous",
        "ACR20 Response": "binary",
        "Number of Exacerbations": "count",
        "Overall Survival": "time-to-event",
        "Disease-Free Survival": "time-to-event",
        "Event-Free Survival": "time-to-event",
        "Relapse-Free Survival": "time-to-event",
        "Distant Recurrence-Free Survival": "time-to-event",
        "Local Recurrence-Free Survival": "time-to-event",
        "Disease-Specific Survival": "time-to-event",
        "Progression-Free Survival Rate": "binary",
        "Overall Survival Rate": "binary",
        "Disease-Free Survival Rate": "binary",
        "Event-Free Survival Rate": "binary",
        "Relapse-Free Survival Rate": "binary",
        "Distant Recurrence-Free Survival Rate": "binary",
        "Local Recurrence-Free Survival Rate": "binary",
        "Disease-Specific Survival Rate": "binary"
    }
    outcome_type = endpoint_map.get(endpoint, "continuous")

    # Sample structure + pairing logic
    if design_type in ["Crossover", "Longitudinal"]:
        paired = True
        sample_type = "related"
    elif design_type == "Single Arm":
        paired = True
        sample_type = "related"
    else:
        paired = False
        sample_type = "independent"

    # Normality
    normal = outcome_type == "continuous" and n_per_arm in ["51-100", ">100"]

    # Small-N logic
    small_n = n_per_arm in ["<20", "20-50"] or phase == "Phase I"
    
    # Repeated measures logic for model matching
    repeated_factors = []
    if design_type in ["Crossover", "Longitudinal"]:
        repeated_factors.append("Time")
    if design_type == "Factorial":
        repeated_factors.append("Condition")

    # Return extended meta
    return {
        "outcome_type": outcome_type,
        "sample_type": sample_type,
        "paired": paired,
        "normal": normal,
        "small_n": small_n,
        "repeated_factors": repeated_factors
    }

