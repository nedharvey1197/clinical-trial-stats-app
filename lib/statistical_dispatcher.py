from .base_models import ClinicalTrialAnalysis

def run_statistical_model(model_id, data, outcome, between_factors=None, within_factors=None):
    model = ClinicalTrialAnalysis(data)

    if model_id == "t_test":
        return model.run_t_test(outcome, between_factors)
    elif model_id == "one_way_anova":
        return model.run_one_way_anova(outcome, between_factors)
    elif model_id == "two_way_anova":
        return model.run_two_way_anova(outcome, between_factors)
    elif model_id == "three_way_anova":
        return model.run_three_way_anova(outcome, between_factors)
    elif model_id == "one_way_rm_anova":
        return model.run_one_way_rm_anova(outcome, within_factors)
    elif model_id == "two_way_rm_anova":
        return model.run_two_way_rm_anova(outcome, within_factors)
    elif model_id == "three_way_rm_anova":
        return model.run_three_way_rm_anova(outcome, within_factors)
    elif model_id == "mixed_anova_1b1r":
        return model.run_mixed_anova_1b1r(outcome, between_factors, within_factors)
    elif model_id == "mixed_anova_2b1r":
        return model.run_mixed_anova_2b1r(outcome, between_factors, within_factors)
    elif model_id == "mixed_anova_1b2r":
        return model.run_mixed_anova_1b2r(outcome, between_factors, within_factors)
    elif model_id == "complex_mixed_anova":
        return model.run_complex_mixed_anova(outcome, between_factors, within_factors)
    elif model_id == "log_rank_test":
        raise NotImplementedError("Log-rank test not yet implemented. Please use a survival analysis package such as lifelines.")
    else:
        raise ValueError(f"Unsupported model_id: {model_id}")