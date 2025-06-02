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
            "assumptions": "1. Normality: The outcome is normally distributed in each group. 2. Equal variance: The two groups have similar spread. 3. Independence: Observations are independent.",
            "limitations": "Sensitive to outliers and non-normality. Not valid if variances are very different.",
            "reference": "https://en.wikipedia.org/wiki/Student%27s_t-test",
            "software": "statsmodels >=0.14.0, scipy >=1.11.0"
        },
        "Bootstrap T-Test": {
            "title": "Bootstrap T-Test",
            "summary": "Estimates confidence intervals and p-values by resampling.",
            "when_to_use": "Use when sample size is small or assumptions of parametric tests are questionable.",
            "assumptions": "1. Resampling: Assumes the sample is representative of the population. 2. Fewer distributional assumptions.",
            "limitations": "May be less accurate with very small samples or highly skewed data.",
            "reference": "https://en.wikipedia.org/wiki/Bootstrapping_(statistics)",
            "software": "numpy >=1.24.0, scipy >=1.11.0"
        },
        "Mann-Whitney U Test": {
            "title": "Mann-Whitney U Test",
            "summary": "Nonparametric test comparing two independent samples.",
            "when_to_use": "Use when comparing ranks of unpaired groups without assuming normality.",
            "assumptions": "1. Independence: Observations are independent. 2. Similar distribution shapes in both groups.",
            "limitations": "Less power than t-test if data are normal. Only tests for difference in distribution, not means.",
            "reference": "https://en.wikipedia.org/wiki/Mann–Whitney_U_test",
            "software": "scipy >=1.11.0"
        },
        "Fisher's Exact Test": {
            "title": "Fisher's Exact Test",
            "summary": "Tests association in a 2x2 contingency table.",
            "when_to_use": "Use for binary outcomes with small sample sizes.",
            "assumptions": "1. Fixed margins: Row and column totals are fixed. 2. Independence: Observations are independent.",
            "limitations": "Only for small samples and 2x2 tables. Computationally intensive for large tables.",
            "reference": "https://en.wikipedia.org/wiki/Fisher%27s_exact_test",
            "software": "scipy >=1.11.0"
        },
        "Chi-Square Test": {
            "title": "Chi-Square Test",
            "summary": "Tests independence in a contingency table.",
            "when_to_use": "Use for categorical data with large enough expected counts.",
            "assumptions": "1. Expected frequencies > 5 in all cells. 2. Independence: Observations are independent.",
            "limitations": "Not valid for small expected counts. Only tests for association, not causality.",
            "reference": "https://en.wikipedia.org/wiki/Chi-squared_test",
            "software": "scipy >=1.11.0"
        },
        "Log-Rank Test": {
            "title": "Log-Rank Test",
            "summary": "Compares survival distributions of two groups.",
            "when_to_use": "Use for time-to-event analysis comparing Kaplan-Meier curves.",
            "assumptions": "1. Proportional hazards: The ratio of hazard rates is constant over time. 2. Non-informative censoring: Censoring is unrelated to outcome.",
            "limitations": "Sensitive to non-proportional hazards. Only compares survival curves, not covariates.",
            "reference": "https://en.wikipedia.org/wiki/Log-rank_test",
            "software": "lifelines >=0.27.0, scipy >=1.11.0"
        },
        "Paired T-Test": {
            "title": "Paired T-Test",
            "summary": "Compares means of two related samples (e.g., before-and-after).",
            "when_to_use": "Use when the same subjects are measured twice under two conditions.",
            "assumptions": "1. Normality of the difference scores. 2. Pairs are independent.",
            "limitations": "Sensitive to outliers and non-normality in difference scores.",
            "reference": "https://en.wikipedia.org/wiki/Student%27s_t-test#Paired_samples",
            "software": "statsmodels >=0.14.0, scipy >=1.11.0"
        },
        "Wilcoxon Signed-Rank Test": {
            "title": "Wilcoxon Signed-Rank Test",
            "summary": "Nonparametric test for comparing two related samples.",
            "when_to_use": "Use for paired samples when data is not normally distributed.",
            "assumptions": "1. Symmetry of differences. 2. Pairs are independent.",
            "limitations": "Less power than paired t-test if data are normal. Only tests for median difference.",
            "reference": "https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test",
            "software": "scipy >=1.11.0"
        }
    }

    return explanations.get(test_name, {
        "title": test_name,
        "summary": "Explanation not available.",
        "when_to_use": "N/A",
        "assumptions": "N/A",
        "reference": ""
    })
