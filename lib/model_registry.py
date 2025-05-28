class StatisticalModel:
    def __init__(
        self,
        label: str,
        model_id: str,
        description: str,
        outcome_types: list,
        repeated_factors: list = None,
        design_tags: list = None,
    ):
        self.label = label
        self.model_id = model_id
        self.description = description
        self.outcome_types = outcome_types
        self.repeated_factors = repeated_factors or []
        self.design_tags = design_tags or []

    def to_dict(self):
        return {
            "label": self.label,
            "model_id": self.model_id,
            "description": self.description,
            "outcome_types": self.outcome_types,
            "repeated_factors": self.repeated_factors,
            "design_tags": self.design_tags,
        }


MODEL_REGISTRY = [
    StatisticalModel("T-test", "t_test", "Compare means between two independent groups.", ["continuous"]),
    StatisticalModel("One-way ANOVA", "one_way_anova", "Compare means across multiple independent groups.", ["continuous"]),
    StatisticalModel("Two-way ANOVA", "two_way_anova", "Analyze main and interaction effects of two factors.", ["continuous"]),
    StatisticalModel("Three-way ANOVA", "three_way_anova", "Analyze effects of three independent variables.", ["continuous"]),
    StatisticalModel("One-way Repeated Measures ANOVA", "one_way_rm_anova", "Analyze repeated measurements for one within-subject factor (e.g., time).", ["continuous"], ["Time"]),
    StatisticalModel("Two-way Repeated Measures ANOVA", "two_way_rm_anova", "Analyze repeated measurements for two within-subject factors (e.g., time and condition).", ["continuous"], ["Time", "Condition"]),
    StatisticalModel("Three-way Repeated Measures ANOVA", "three_way_rm_anova", "Analyze repeated measurements across three within-subject factors.", ["continuous"], ["Time", "Condition"]),
    StatisticalModel("Mixed ANOVA (One Between, One Repeated)", "mixed_anova_1b1r", "Test between-group differences with repeated measures.", ["continuous"], ["Time"]),
    StatisticalModel("Mixed ANOVA (Two Between, One Repeated)", "mixed_anova_2b1r", "Evaluate two independent group effects and one repeated measure.", ["continuous"], ["Time"]),
    StatisticalModel("Mixed ANOVA (One Between, Two Repeated)", "mixed_anova_1b2r", "Analyze mixed model with one between-subject and two within-subject factors.", ["continuous"], ["Time", "Condition"]),
    StatisticalModel("Complex Mixed ANOVA", "complex_mixed_anova", "Handle complex repeated measures and factorial designs.", ["continuous"], ["Time", "Condition"]),
]