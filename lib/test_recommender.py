def recommend_statistical_test(
    n_between: int,
    n_within: int,
    n_groups: int,
    n_within_levels: list = None
) -> str:
    """
    Recommend the appropriate statistical test model_id based on design.
    Args:
        n_between (int): Number of between-subjects factors
        n_within (int): Number of within-subjects (repeated) factors
        n_groups (int): Number of groups in the main between factor (if any)
        n_within_levels (list): Number of levels for each within factor (optional)
    Returns:
        str: model_id for the recommended test
    """
    n_within_levels = n_within_levels or []
    # Pure repeated measures ANOVA (no between factors)
    if n_between == 0 and n_within == 1 and all(l > 1 for l in n_within_levels):
        return "one_way_rm_anova"
    if n_between == 0 and n_within == 2 and all(l > 1 for l in n_within_levels):
        return "two_way_rm_anova"
    if n_between == 0 and n_within == 3 and all(l > 1 for l in n_within_levels):
        return "three_way_rm_anova"
    # Between-subjects only
    if n_between == 1 and n_within == 0 and n_groups == 2:
        return "t_test"
    if n_between == 1 and n_within == 0 and n_groups > 2:
        return "one_way_anova"
    if n_between == 2 and n_within == 0:
        return "two_way_anova"
    if n_between == 3 and n_within == 0:
        return "three_way_anova"
    # Mixed designs
    if n_between == 1 and n_within == 1:
        return "mixed_anova_1b1r"
    if n_between == 2 and n_within == 1:
        return "mixed_anova_2b1r"
    if n_between == 1 and n_within == 2:
        return "mixed_anova_1b2r"
    if n_between >= 2 and n_within >= 2:
        return "complex_mixed_anova"
    return "No appropriate test found. Please check your design." 