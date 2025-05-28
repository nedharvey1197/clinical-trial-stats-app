# lib/explainer/explain_test_selection.py
from typing import Dict

def explain_test(test_name: str) -> Dict[str, str]:
    """
    Return structured explanation for a given statistical test.

    Args:
        test_name (str): Name of the test (e.g., 'T-test')

    Returns:
        Dict[str, str]: Explanation components: title, summary, when_to_use, assumptions, reference.
    """
    explanations = {
        "T-test": {
            "title": "Independent T-Test",
            "summary": "Compares the means of two independent groups.",
            "when_to_use": "Use when comparing two unpaired groups with normally distributed data and equal variances.",
            "assumptions": "Normality, equal variance, independent observations.",
            "reference": "https://en.wikipedia.org/wiki/Student%27s_t-test"
        },
        "Bootstrap T-Test": {
            "title": "Bootstrap T-Test",
            "summary": "Estimates confidence intervals and p-values by resampling.",
            "when_to_use": "Use when sample size is small or assumptions of parametric tests are questionable.",
            "assumptions": "Resampling from observed data; fewer assumptions.",
            "reference": "https://en.wikipedia.org/wiki/Bootstrapping_(statistics)"
        },
        "Mann-Whitney U Test": {
            "title": "Mann-Whitney U Test",
            "summary": "Nonparametric test comparing two independent samples.",
            "when_to_use": "Use when comparing ranks of unpaired groups without assuming normality.",
            "assumptions": "Independence, similar distribution shapes.",
            "reference": "https://en.wikipedia.org/wiki/Mann–Whitney_U_test"
        },
        "Fisher's Exact Test": {
            "title": "Fisher's Exact Test",
            "summary": "Tests association in a 2x2 contingency table.",
            "when_to_use": "Use for binary outcomes with small sample sizes.",
            "assumptions": "Fixed margins, independence.",
            "reference": "https://en.wikipedia.org/wiki/Fisher%27s_exact_test"
        },
        "Chi-Square Test": {
            "title": "Chi-Square Test",
            "summary": "Tests independence in a contingency table.",
            "when_to_use": "Use for categorical data with large enough expected counts.",
            "assumptions": "Expected frequencies > 5 in all cells, independence.",
            "reference": "https://en.wikipedia.org/wiki/Chi-squared_test"
        },
        "Log-Rank Test": {
            "title": "Log-Rank Test",
            "summary": "Compares survival distributions of two groups.",
            "when_to_use": "Use for time-to-event analysis comparing Kaplan-Meier curves.",
            "assumptions": "Proportional hazards, non-informative censoring.",
            "reference": "https://en.wikipedia.org/wiki/Log-rank_test"
        },
        "Paired T-Test": {
            "title": "Paired T-Test",
            "summary": "Compares means of two related samples (e.g., before-and-after).",
            "when_to_use": "Use when the same subjects are measured twice under two conditions.",
            "assumptions": "Normality of the difference scores.",
            "reference": "https://en.wikipedia.org/wiki/Student%27s_t-test#Paired_samples"
        },
        "Wilcoxon Signed-Rank Test": {
            "title": "Wilcoxon Signed-Rank Test",
            "summary": "Nonparametric test for comparing two related samples.",
            "when_to_use": "Use for paired samples when data is not normally distributed.",
            "assumptions": "Symmetry of differences, paired samples.",
            "reference": "https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test"
        }
    }

    return explanations.get(test_name, {
        "title": test_name,
        "summary": "Explanation not available.",
        "when_to_use": "N/A",
        "assumptions": "N/A",
        "reference": ""
    })
