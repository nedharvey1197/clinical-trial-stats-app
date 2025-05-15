def get_explanation(step: str, model_type: str) -> dict:
    """
    Generate explanation text for a given step.

    Args:
        step (str): Analysis step (e.g., "data_input", "assumption_checks").
        model_type (str): Model type (e.g., "T-test").

    Returns:
        dict: Explanation content.
    """
    explanations = {
        "data_input": {
            "What": f"Displaying a preloaded dataset for the selected model: {model_type}.",
            "Why": "This allows you to see the expected data format and explore the analysis without entering your own data. The dataset is tailored to the model's requirements.",
            "Source": "The data is predefined in the app, similar to how statisticians might use example datasets in SAS or SPSS to demonstrate analysis."
        },
        "assumption_checks": {
            "What": "Checking statistical assumptions (normality, equal variances, sphericity).",
            "Why": "These tests ensure the data meets the assumptions of the model. Shapiro-Wilk tests normality, Levene’s tests equal variances, and Mauchly’s tests sphericity for repeated measures. If assumptions fail, alternative tests are used (e.g., Kruskal-Wallis for non-normal data).",
            "Implications": "Violations (e.g., non-normality, sphericity violation) may inflate Type I error rates. Corrections like Greenhouse-Geisser reduce this risk.",
            "Source": "Using <code>scipy.stats</code> for Shapiro-Wilk and Levene’s tests, and a custom implementation of Mauchly’s Test (not built into Python libraries, unlike SAS PROC MIXED).",
            "Reference": "Shapiro-Wilk test: Shapiro, M. B., & Wilk, M. B. (1965). An analysis of variance test for normality. Biometrika, 52(3-4), 591-611."
        },
        "results": {
            "What": "Running the statistical test and displaying results.",
            "Why": "This tests the main hypotheses (e.g., differences between groups, effects over time). For ANOVA, an F-test is used; for T-test, a t-statistic. Mixed models provide detailed fit statistics (e.g., Log-Likelihood, AIC).",
            "Source": "Using <code>statsmodels</code> for ANOVA and mixed models, <code>scipy.stats</code> for alternative tests (e.g., Kruskal-Wallis). Equivalent to SAS PROC GLM or PROC MIXED."
        },
        "visualization": {
            "What": "Generating a box plot (non-repeated) or interaction plot (repeated measures).",
            "Why": "Visualizations help interpret the data and results. Box plots show group distributions, while interaction plots show trends over time or across factors.",
            "Source": "Using <code>seaborn</code> and <code>matplotlib</code>, similar to SAS ODS Graphics output."
        }
    }
    return explanations.get(step, {})