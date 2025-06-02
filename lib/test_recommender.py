from typing import List, Tuple
from lib.model_registry import MODEL_REGISTRY, validate_model_compatibility, get_model_by_id

def recommend_statistical_test(
    n_between: int,
    n_within: int,
    n_groups: int,
    n_within_levels: List[int],
    outcome_type: str,
    design_type: str,
    randomization: str,
    has_covariates: bool,
    has_repeated_measures: bool,
    sample_size: str
) -> Tuple[str, List[str]]:
    """
    Recommend appropriate statistical tests based on design parameters.
    
    Args:
        n_between: Number of between-subject factors
        n_within: Number of within-subject factors
        n_groups: Number of groups/arms
        n_within_levels: Number of levels for each within-subject factor
        outcome_type: Type of outcome variable
        design_type: Type of trial design
        randomization: Randomization type
        has_covariates: Whether covariates are present
        has_repeated_measures: Whether repeated measures are present
        sample_size: Sample size category
        
    Returns:
        Tuple of (recommended_model_id, list_of_alternative_model_ids)
    """
    compatible_models = []
    
    for model in MODEL_REGISTRY:
        is_compatible, reason = validate_model_compatibility(
            model=model,
            design_type=design_type,
            n_groups=n_groups,
            outcome_type=outcome_type,
            randomization=randomization,
            has_covariates=has_covariates,
            has_repeated_measures=has_repeated_measures,
            sample_size=sample_size
        )
        
        if is_compatible:
            compatible_models.append(model)
    
    if not compatible_models:
        return None, []
    
    # Sort models by appropriateness score
    scored_models = []
    for model in compatible_models:
        score = calculate_model_appropriateness_score(
            model=model,
            n_between=n_between,
            n_within=n_within,
            n_groups=n_groups,
            n_within_levels=n_within_levels,
            outcome_type=outcome_type,
            design_type=design_type,
            has_covariates=has_covariates,
            has_repeated_measures=has_repeated_measures,
            sample_size=sample_size
        )
        scored_models.append((model, score))
    
    # Sort by score in descending order
    scored_models.sort(key=lambda x: x[1], reverse=True)
    
    # Return best model and alternatives
    best_model = scored_models[0][0]
    alternatives = [m[0].model_id for m in scored_models[1:]]
    
    return best_model.model_id, alternatives

def calculate_model_appropriateness_score(
    model,
    n_between: int,
    n_within: int,
    n_groups: int,
    n_within_levels: List[int],
    outcome_type: str,
    design_type: str,
    has_covariates: bool,
    has_repeated_measures: bool,
    sample_size: str
) -> float:
    """Calculate how appropriate a model is for the given design parameters."""
    score = 0.0
    
    # Base score for compatibility
    score += 10.0
    
    # Design type match
    if design_type in model.valid_designs:
        score += 5.0
    
    # Outcome type match
    if outcome_type in model.valid_outcome_types:
        score += 5.0
    
    # Sample size appropriateness
    if sample_size == "<20" and "max_per_group" in model.sample_size_requirements:
        if model.sample_size_requirements["max_per_group"] >= 20:
            score += 3.0
    elif sample_size in ["20-50", "51-100", ">100"]:
        if "min_per_group" in model.sample_size_requirements:
            min_req = model.sample_size_requirements["min_per_group"]
            if (sample_size == "20-50" and min_req <= 50) or \
               (sample_size == "51-100" and min_req <= 100) or \
               (sample_size == ">100" and min_req <= 100):
                score += 3.0
    
    # Covariate support
    if has_covariates and model.supports_covariates:
        score += 2.0
    
    # Repeated measures support
    if has_repeated_measures and model.supports_repeated_measures:
        score += 2.0
    
    # Group size appropriateness
    if model.min_groups <= n_groups <= model.max_groups:
        score += 2.0
    
    return score 