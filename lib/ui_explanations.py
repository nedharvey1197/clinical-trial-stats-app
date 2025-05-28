from .data import between_factors_dict, repeated_factors_dict
from typing import Dict, Any

def get_explanation(step: str, model_type: str, results: Dict[str, Any] = None, quality_report: Dict[str, Any] = None) -> Dict[str, str]:
    """
    Generate dynamic, model-specific explanation text for a given step, reflecting the program's logic and results in readable English.

    Args:
        step (str): Analysis step (e.g., "data_input", "assumption_checks").
        model_type (str): Model type (e.g., "T-test").
        results (Dict[str, Any], optional): Analysis results dictionary.
        quality_report (Dict[str, Any], optional): Quality report dictionary.

    Returns:
        Dict[str, str]: Explanation content with updated headers and no category labels.
    """
    # Determine model characteristics
    has_repeated_factors = len(repeated_factors_dict.get(model_type, [])) > 0
    has_between_factors = len(between_factors_dict.get(model_type, [])) > 0

    # Base explanations without category labels
    explanations = {
        "data_input": {
            "Here’s What I’m Doing": f"I’m showing you a sample dataset for the {model_type} test.",
            "How Does This Help You?": "This lets you see how the test works without needing your own data—it’s like a practice run, with everything set up just right.",
            "Expert Info": "This data is built into the app, similar to how researchers use example datasets in tools like SAS or SPSS to show how analyses work."
        },
        "assumption_checks": {
            "Here’s What I Did": "I made sure the data was ready for this test.",
            "Why Did I Do That?": (
                f"For the {model_type} test, I checked if the data looks normal (using a Shapiro-Wilk test)"
                f"{', if the groups have similar spreads (using a Levene’s test)' if has_between_factors else ''}"
                f"{', and if measurements over time are steady (using a Mauchly’s test)' if has_repeated_factors else ''}. "
                f"This makes sure the test gives us results we can trust, like "
                f"{'ensuring all the players in a game are starting fairly' if not has_repeated_factors else 'making sure a runner’s speed stays even lap after lap in a race' if not has_between_factors else 'ensuring all teams in a relay race are playing fair and running steadily'}."
            ),
            "Expert Info": (
                f"Using <code>scipy.stats</code> for Shapiro-Wilk{' and Levene’s' if has_between_factors else ''}"
                f"{', and a custom Mauchly’s Test implementation (not built into Python libraries, unlike SAS PROC MIXED)' if has_repeated_factors else ''}. "
                f"Equivalent to assumption checks in SAS PROC {'MIXED' if has_repeated_factors else 'GLM'}."
            ),
            "Reference": (
                "Shapiro-Wilk test: Shapiro, M. B., & Wilk, M. B. (1965). An analysis of variance test for normality. Biometrika, 52(3-4), 591-611."
                f"{' Mauchly’s Test: Mauchly, J. W. (1940). Significance test for sphericity of a normal n-variate distribution. Annals of Mathematical Statistics, 11(2), 204-209.' if has_repeated_factors else ''}"
            )
        },
        "results": {
            "Here’s What I Did": (
                f"I used a {model_type} to "
                f"{'compare groups' if not has_repeated_factors else 'see changes over time or conditions' if not has_between_factors else 'compare groups and changes over time'}."
            ),
            "Why Did I Do That?": (
                f"I wanted to see "
                f"{'if things like ' + ', '.join(between_factors_dict.get(model_type, [])) + ' (e.g., different drug doses) lead to different outcomes (e.g., cholesterol levels)' if not has_repeated_factors else ''}"
                f"{'how the same people’s results (e.g., pain levels) change over ' + ', '.join(repeated_factors_dict.get(model_type, [])) + ' (e.g., before and after treatment)' if not has_between_factors else ''}"
                f"{'both differences between groups (e.g., ' + ', '.join(between_factors_dict.get(model_type, [])) + ') and how their results change over ' + ', '.join(repeated_factors_dict.get(model_type, [])) + ' (e.g., before and after treatment)' if has_between_factors and has_repeated_factors else ''}. "
                f"{'Imagine you’re trying three different cookie recipes to see which one tastes the best—I’m helping figure out if one really stands out.' if not has_repeated_factors else 'It’s like tracking how your energy levels shift throughout the day—I’m checking if there’s a real pattern, like if you’re more tired after lunch.' if not has_between_factors else 'It’s like comparing two friends’ fitness progress over months—I’m seeing if one gets stronger faster.'}"
            )
        },
        "visualization": {
            "Here’s What I Did": "I created a picture to show the results.",
            "How Does This Help You?": (
                f"This chart shows how the results (e.g., cholesterol levels) "
                f"{'look for each group (e.g., drug doses)' if not has_repeated_factors else 'change over ' + ', '.join(repeated_factors_dict.get(model_type, [])) + ' (e.g., before and after treatment)' if not has_between_factors else 'change over ' + ', '.join(repeated_factors_dict.get(model_type, [])) + ' for different ' + ', '.join(between_factors_dict.get(model_type, []))}. "
                f"{'It’s like seeing how tall different plants grow with different fertilizers—if one plant shoots up higher, you’ll see it right away, making it easy to spot differences.' if not has_repeated_factors else 'It’s like a line graph of your daily mood—if you’re happier on weekends, the line will go up then, helping you see the trend clearly.' if not has_between_factors else 'It’s like comparing two friends’ running speeds over a month—if one friend gets faster sooner, their line will climb quicker, making it easy to see who’s improving more.'}"
            )
        }
    }

    # Get the base explanation for the step
    explanation = explanations.get(step, {}).copy()

    # Dynamically enhance explanations based on results and quality report
    if step == "assumption_checks" and results:
        assumptions = results.get('Assumptions', {})
        shapiro_p = assumptions.get('Shapiro-Wilk', (0.0, 0.0))[1]
        levene_p = assumptions.get('Levene', (0.0, 0.0))[1] if has_between_factors else None
        sphericity_p = assumptions.get('Mauchly-Sphericity', {'W': 0.0, 'p_value': 0.0})['p_value'] if has_repeated_factors else None

        # Add specific results
        results_str = f"Shapiro-Wilk check for normality: {shapiro_p:.4f} ({'looks unusual' if shapiro_p < 0.05 else 'looks normal'})"
        if levene_p is not None:
            results_str += f", Levene’s check for similar group spreads: {levene_p:.4f} ({'spreads differ' if levene_p < 0.05 else 'spreads are similar'})"
        if sphericity_p is not None:
            results_str += f", Mauchly’s check for consistency over time: {sphericity_p:.4f} ({'not consistent' if sphericity_p < 0.05 else 'consistent'})"
        explanation["Results"] = results_str

        # Add implications based on results
        implications = []
        if shapiro_p < 0.05:
            implications.append("The data looks a bit unusual, so I used a different method to avoid mistakes.")
        if levene_p is not None and levene_p < 0.05:
            implications.append("The groups have different spreads, so I adjusted the test to handle that.")
        if sphericity_p is not None and sphericity_p < 0.05:
            implications.append("The repeated measurements aren’t consistent, so I made a small tweak to fix it.")
        if implications:
            explanation["What We Did"] = " ".join(implications)
        else:
            explanation["What We Did"] = "Everything checked out, so I used the standard test."

    if step == "results" and results:
        # Initialize confidence to None to avoid UnboundLocalError
        confidence = None

        # Determine the test used and its outcome
        if 'Alternative Test' in results:
            test_used = "a backup test (since the data didn’t fit the usual rules)"
            test_result = f"Result: {results['Alternative Test']}"
            test_name = "Kruskal-Wallis" if "ANOVA" in model_type else "Mann-Whitney U"
            # Since there's no p-value here, confidence remains None
        elif model_type == "T-test":
            test_used = "a T-test"
            test_result = f"t = {results['T-test']['t_stat']:.2f}, p = {results['T-test']['p_value']:.4f}"
            test_name = "T-test"
            confidence = (1 - results['T-test']['p_value']) * 100
        elif "ANOVA" in results:
            test_used = "an ANOVA test"
            f_value = results['ANOVA']['F'][0] if 'F' in results['ANOVA'].columns else "N/A"
            p_value = results['ANOVA']['PR(>F)'][0] if 'PR(>F)' in results['ANOVA'].columns else "N/A"
            test_result = f"F = {f_value:.2f}, p = {p_value:.4f}"
            test_name = "ANOVA"
            confidence = (1 - p_value) * 100
        else:
            test_used = "a Mixed ANOVA test"
            test_result = "Check the detailed summary below for the results."
            test_name = "Mixed ANOVA"
            confidence = None  # Mixed models have multiple p-values, so we skip confidence here

        # Update the "Here’s What I Did" to include the test name
        explanation["Here’s What I Did"] = f"I used {test_used} to {'compare groups' if not has_repeated_factors else 'see changes over time or conditions' if not has_between_factors else 'compare groups and changes over time'}."

        # Add the test result in a conversational way
        if confidence is not None:
            result_summary = f"Here’s what I found: {test_result}. I’m {confidence:.1f}% sure there’s a real {'difference between groups' if not has_repeated_factors else 'change over time' if not has_between_factors else 'difference and change'}—it’s not just luck!"
        else:
            result_summary = f"Here’s what I found: {test_result}. Take a look at the summary below to see what’s different and what changed over time."
        explanation["Results"] = result_summary

        # Add effect size insights
        effect_sizes = results.get('Effect Sizes', {})
        effect_size_str = []
        for key, value in effect_sizes.items():
            if key != "Clinical Significance":
                # Check if value is numeric before formatting as float
                if isinstance(value, (int, float)):
                    effect_size_str.append(f"{key}: {value:.2f}")
                    # Add a simple interpretation of the effect size
                    if "Cohen" in key:
                        size_desc = "small" if value < 0.5 else "medium" if value < 0.8 else "large"
                        explanation["How Big"] = f"The difference is {size_desc}—{'barely noticeable' if size_desc == 'small' else 'pretty clear' if size_desc == 'medium' else 'really obvious'}!"
                    elif "Eta-squared" in key:
                        size_desc = "small" if value < 0.06 else "medium" if value < 0.14 else "large"
                        explanation["How Big"] = f"The difference is {size_desc}—{'not a big deal' if size_desc == 'small' else 'noticeable' if size_desc == 'medium' else 'a big deal'}!"
                else:
                    # If value is a string (e.g., error message), include it as-is
                    effect_size_str.append(f"{key}: {value}")
        if effect_size_str:
            explanation["Effect Sizes"] = f"For the experts: {'; '.join(effect_size_str)}. {effect_sizes.get('Clinical Significance', '')}"

        # Add sensitivity analysis insights
        if 'Sensitivity Test' in results:
            explanation["Double-Check"] = f"I double-checked with another test because the data was tricky: {results['Sensitivity Test']}"

        # Add technical details for experts
        if not has_repeated_factors:
            explanation["Expert Info"] = "Using <code>statsmodels</code> for ANOVA and <code>scipy.stats</code> for T-test and alternative tests (e.g., Kruskal-Wallis). Equivalent to SAS PROC GLM."
        else:
            explanation["Expert Info"] = "Using <code>statsmodels</code> MixedLM for mixed models. Equivalent to SAS PROC MIXED."

    if step == "visualization" and results:
        factors_str = f"across {', '.join(between_factors_dict.get(model_type, []))} {'and over ' + ', '.join(repeated_factors_dict.get(model_type, [])) if has_repeated_factors else ''}"
        explanation["How Does This Help You?"] = f"This chart helps you picture how the results spread out {factors_str}. It’s a quick way to spot {'differences between groups' if not has_repeated_factors else 'changes over time or conditions' if not has_between_factors else 'how groups change over time'}."

    # Add quality report insights for data_input or results
    if quality_report and (step == "data_input" or step == "results"):
        if 'imputation' in quality_report:
            explanation["Data Quality"] = f"Just so you know: {quality_report['imputation']}"
        if 'performance_warning' in quality_report and quality_report['performance_warning']:
            explanation["Data Quality"] = (explanation.get("Data Quality", "") + " " + quality_report['performance_warning']).strip()

    return explanation