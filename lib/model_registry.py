from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class StatisticalModel:
    model_id: str
    label: str
    description: str
    design_tags: List[str]
    valid_designs: List[str]
    min_groups: int
    max_groups: int
    valid_outcome_types: List[str]
    requires_randomization: bool
    supports_covariates: bool
    supports_repeated_measures: bool
    sample_size_requirements: Dict[str, Any]

MODEL_REGISTRY = [
    StatisticalModel(
        model_id="t_test",
        label="Independent Samples t-test",
        description="Compares means between two independent groups",
        design_tags=["parallel", "two-groups", "continuous", "independent"],
        valid_designs=["Parallel Group"],
        min_groups=2,
        max_groups=2,
        valid_outcome_types=["continuous"],
        requires_randomization=True,
        supports_covariates=False,
        supports_repeated_measures=False,
        sample_size_requirements={"min_per_group": 20, "recommended_per_group": 30}
    ),
    StatisticalModel(
        model_id="paired_t_test",
        label="Paired Samples t-test",
        description="Compares means between two related groups",
        design_tags=["crossover", "two-groups", "continuous", "paired"],
        valid_designs=["Crossover", "Single Arm"],
        min_groups=1,
        max_groups=2,
        valid_outcome_types=["continuous"],
        requires_randomization=False,
        supports_covariates=False,
        supports_repeated_measures=True,
        sample_size_requirements={"min_per_group": 20, "recommended_per_group": 30}
    ),
    StatisticalModel(
        model_id="one_way_anova",
        label="One-way ANOVA",
        description="Compares means across multiple independent groups",
        design_tags=["parallel", "multi-group", "continuous", "independent"],
        valid_designs=["Parallel Group"],
        min_groups=3,
        max_groups=20,
        valid_outcome_types=["continuous"],
        requires_randomization=True,
        supports_covariates=False,
        supports_repeated_measures=False,
        sample_size_requirements={"min_per_group": 20, "recommended_per_group": 30}
    ),
    StatisticalModel(
        model_id="repeated_measures_anova",
        label="Repeated Measures ANOVA",
        description="Analyzes changes over time within the same subjects",
        design_tags=["longitudinal", "multi-timepoint", "continuous", "repeated"],
        valid_designs=["Longitudinal", "Crossover"],
        min_groups=1,
        max_groups=20,
        valid_outcome_types=["continuous"],
        requires_randomization=False,
        supports_covariates=True,
        supports_repeated_measures=True,
        sample_size_requirements={"min_total": 30, "recommended_total": 50}
    ),
    StatisticalModel(
        model_id="cox_regression",
        label="Cox Proportional Hazards Regression",
        description="Analyzes time-to-event data with covariates",
        design_tags=["survival", "time-to-event", "multi-group"],
        valid_designs=["Parallel Group", "Single Arm"],
        min_groups=1,
        max_groups=20,
        valid_outcome_types=["time-to-event"],
        requires_randomization=False,
        supports_covariates=True,
        supports_repeated_measures=False,
        sample_size_requirements={"min_events": 50, "recommended_events": 100}
    ),
    StatisticalModel(
        model_id="logistic_regression",
        label="Logistic Regression",
        description="Analyzes binary outcomes with covariates",
        design_tags=["binary", "multi-group", "categorical"],
        valid_designs=["Parallel Group", "Single Arm"],
        min_groups=1,
        max_groups=20,
        valid_outcome_types=["binary"],
        requires_randomization=False,
        supports_covariates=True,
        supports_repeated_measures=False,
        sample_size_requirements={"min_per_group": 30, "recommended_per_group": 50}
    ),
    StatisticalModel(
        model_id="mixed_effects_model",
        label="Mixed Effects Model",
        description="Analyzes repeated measures with random effects",
        design_tags=["longitudinal", "repeated", "multi-group"],
        valid_designs=["Longitudinal", "Crossover"],
        min_groups=1,
        max_groups=20,
        valid_outcome_types=["continuous", "binary", "count"],
        requires_randomization=False,
        supports_covariates=True,
        supports_repeated_measures=True,
        sample_size_requirements={"min_subjects": 20, "recommended_subjects": 40}
    ),
    StatisticalModel(
        model_id="fishers_exact",
        label="Fisher's Exact Test",
        description="Analyzes categorical data in small samples",
        design_tags=["binary", "small-n", "categorical"],
        valid_designs=["Parallel Group", "Single Arm"],
        min_groups=2,
        max_groups=2,
        valid_outcome_types=["binary"],
        requires_randomization=True,
        supports_covariates=False,
        supports_repeated_measures=False,
        sample_size_requirements={"max_per_group": 20}
    )
]

def get_model_by_id(model_id: str) -> StatisticalModel:
    """Get a model from the registry by its ID."""
    for model in MODEL_REGISTRY:
        if model.model_id == model_id:
            return model
    return None

def validate_model_compatibility(model: StatisticalModel, design_type: str, n_groups: int, 
                               outcome_type: str, randomization: str, has_covariates: bool,
                               has_repeated_measures: bool, sample_size: str) -> tuple[bool, str]:
    """Validate if a model is compatible with the given design parameters."""
    if design_type not in model.valid_designs:
        return False, f"Model {model.label} is not valid for {design_type} design"
    
    if n_groups < model.min_groups or n_groups > model.max_groups:
        return False, f"Model {model.label} requires between {model.min_groups} and {model.max_groups} groups"
    
    if outcome_type not in model.valid_outcome_types:
        return False, f"Model {model.label} does not support {outcome_type} outcomes"
    
    if model.requires_randomization and randomization == "Non-Randomized":
        return False, f"Model {model.label} requires randomization"
    
    if has_covariates and not model.supports_covariates:
        return False, f"Model {model.label} does not support covariates"
    
    if has_repeated_measures and not model.supports_repeated_measures:
        return False, f"Model {model.label} does not support repeated measures"
    
    # Check sample size requirements
    if sample_size == "<20" and "max_per_group" in model.sample_size_requirements:
        if model.sample_size_requirements["max_per_group"] < 20:
            return False, f"Sample size too large for {model.label}"
    
    return True, ""