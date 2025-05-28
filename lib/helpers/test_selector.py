# lib/helpers/test_selector.py
from typing import Dict, List

def recommend_tests(outcome_type: str, sample_type: str, paired: bool = False, normal: bool = True, small_n: bool = False) -> List[Dict]:
    """
    Rule-based recommender for statistical tests based on trial design inputs.

    Args:
        outcome_type (str): 'continuous', 'binary', or 'time-to-event'
        sample_type (str): 'independent' or 'related'
        paired (bool): Are samples paired/repeated?
        normal (bool): Is distribution assumed normal?
        small_n (bool): Small sample size (e.g. < 30 per group)

    Returns:
        List[Dict]: Recommended tests with rationale and assumptions.
    """
    recommendations = []

    if outcome_type == 'binary':
        if small_n or not normal:
            recommendations.append({
                "test": "Fisher's Exact Test",
                "reason": "Binary outcome with small N. Exact test is robust.",
                "assumptions": "Independence, fixed margins."
            })
        else:
            recommendations.append({
                "test": "Chi-Square Test",
                "reason": "Binary outcome with adequate sample size.",
                "assumptions": "Expected counts > 5 per cell."
            })

    elif outcome_type == 'continuous':
        if paired:
            if normal:
                recommendations.append({
                    "test": "Paired T-Test",
                    "reason": "Continuous paired data with normal distribution.",
                    "assumptions": "Normality of difference scores."
                })
            else:
                recommendations.append({
                    "test": "Wilcoxon Signed-Rank Test",
                    "reason": "Nonparametric test for paired continuous data.",
                    "assumptions": "Symmetry of difference scores."
                })
        else:
            if normal:
                recommendations.append({
                    "test": "Independent T-Test",
                    "reason": "Continuous, unpaired data with normal distribution.",
                    "assumptions": "Normality, equal variance."
                })
                if small_n:
                    recommendations.append({
                        "test": "Bootstrap T-Test",
                        "reason": "Improves reliability for small samples.",
                        "assumptions": "Fewer assumptions; resampling-based."
                    })
            else:
                recommendations.append({
                    "test": "Mann-Whitney U Test",
                    "reason": "Nonparametric test for independent samples.",
                    "assumptions": "Distribution-free; assumes similar shape."
                })

    elif outcome_type == 'time-to-event':
        recommendations.append({
            "test": "Log-Rank Test",
            "reason": "Standard test for survival analysis.",
            "assumptions": "Non-informative censoring, proportional hazards."
        })

    return recommendations