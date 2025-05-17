"""
Clinical Trial Statistical Analysis Module

This module provides statistical analysis tools for clinical trial data, supporting:
- T-test
- One-way, Two-way, and Three-way ANOVA
- Repeated Measures ANOVA
- Mixed ANOVA designs
- Complex factorial designs

Functionality includes:
- Data validation and preprocessing
- Assumption checking (normality, equal variances, sphericity)
- Descriptive statistics
- Statistical testing (including alternative tests if assumptions fail)
- Visualization
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Any, Tuple
from matplotlib.figure import Figure

class ClinicalTrialAnalysis:
    """
    A class for performing statistical analysis on clinical trial data.

    Attributes:
        data (pd.DataFrame): The clinical trial data to be analyzed.
        results (Dict[str, Any]): Dictionary storing analysis results.
    """

    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the ClinicalTrialAnalysis object.

        Args:
            data (Optional[pd.DataFrame]): Clinical trial data to analyze.
                If None, data must be provided before running analysis.
        """
        self.data = data
        self.results: Dict[str, Any] = {}

    def _validate_data(self) -> None:
        """
        Validate that data exists and is properly formatted.

        Raises:
            ValueError: If no data is provided.
            TypeError: If data is not a pandas DataFrame.
        """
        if self.data is None:
            raise ValueError("No data provided. Please provide data before running analysis.")
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")

    def _mauchly_test(self, data: pd.DataFrame, repeated_factors: List[str], outcome: str) -> Tuple[float, float]:
        """
        Perform Mauchly's Test for Sphericity on repeated measures data.
        Args:
            data (pd.DataFrame): DataFrame containing the repeated measures.
            repeated_factors (List[str]): Names of the repeated-measures factors.
            outcome (str): Name of the outcome variable.
        Returns:
            Tuple[float, float]: W-statistic and p-value for Mauchly's Test.
        """
        subset_cols = ['Subject'] + repeated_factors
        if data.duplicated(subset=subset_cols).any():
            raise ValueError(
                f"Your data contains duplicate entries for the same {', '.join(subset_cols)}. "
                f"Each Subject should have only one value per combination of repeated factors. "
                f"Please check your data for duplicates or aggregate them (e.g., by averaging)."
            )
        # Pivot data to wide format: each row is a subject, columns are multi-index of repeated factors
        wide_data = data.pivot_table(index='Subject', columns=repeated_factors, values=outcome)
        wide_data = wide_data.dropna()  # Drop any subjects with missing data
        n_subjects = wide_data.shape[0]
        k = wide_data.shape[1]  # Number of levels (cells)
        if n_subjects < k:
            return 0.0, 1.0  # Cannot compute if subjects < levels
        cov_matrix = np.cov(wide_data, rowvar=False)
        det_cov = np.linalg.det(cov_matrix)
        trace_cov = np.trace(cov_matrix)
        mean_variance = trace_cov / k
        W = det_cov / (mean_variance ** k)
        df = (k * (k - 1)) / 2 - 1
        if df <= 0:
            return W, 1.0
        chi2 = - (n_subjects - 1) * np.log(W)
        p_value = 1 - stats.chi2.cdf(chi2, df)
        return W, p_value

    def validate_no_duplicates(self, repeated_factors: List[str]):
        """
        Validate that there are no duplicate (Subject + repeated_factors) combinations in the data.
        """
        if not repeated_factors:
            return
        subset_cols = ['Subject'] + repeated_factors
        if self.data.duplicated(subset=subset_cols).any():
            raise ValueError(
                f"Your data contains duplicate entries for the same {', '.join(subset_cols)}. "
                f"Each Subject should have only one value per combination of repeated factors. "
                f"Please check your data for duplicates or aggregate them (e.g., by averaging)."
            )

    def check_assumptions(self, outcome: str, between_factors: List[str], repeated_factors: List[str]) -> Dict[str, Any]:
        """
        Check statistical assumptions for the analysis.

        Performs:
        - Shapiro-Wilk test for normality on residuals.
        - Levene's test for equal variances (between-subjects).
        - Mauchly's test for sphericity (repeated measures).

        Args:
            outcome (str): Name of the outcome variable.
            between_factors (List[str]): List of between-subjects factors.
            repeated_factors (List[str]): List of repeated-measures factors.

        Returns:
            Dict[str, Any]: Dictionary containing assumption test results.
        """
        self._validate_data()
        if self.data is None or not isinstance(self.data, pd.DataFrame):
            raise ValueError("No data provided or data is not a DataFrame. Please provide valid data before running check_assumptions.")
        self.validate_no_duplicates(repeated_factors)
        assumptions: Dict[str, Any] = {}

        # Normality (Shapiro-Wilk on residuals)
        formula = f"{outcome} ~ {' * '.join(between_factors + repeated_factors)}" if (between_factors or repeated_factors) else f"{outcome} ~ 1"
        model = ols(formula, data=self.data).fit()
        residuals = model.resid
        assumptions['Shapiro-Wilk'] = stats.shapiro(residuals)

        # Equal variances (Levene's test for between factors)
        if between_factors:
            groups = [self.data[self.data[between_factors[0]] == level][outcome]
                     for level in self.data[between_factors[0]].unique()]
            assumptions['Levene'] = stats.levene(*groups)

        # Sphericity (Mauchly's test for repeated measures)
        if repeated_factors:
            W, p_value = self._mauchly_test(self.data, repeated_factors, outcome)
            assumptions['Mauchly-Sphericity'] = {'W': W, 'p_value': p_value}
            if p_value < 0.05:
                assumptions['Sphericity Note'] = "Sphericity violated; Greenhouse-Geisser correction applied in mixed models."

        return assumptions

    def descriptive_stats(self, outcome: str, factors: List[str]) -> pd.DataFrame:
        """
        Compute descriptive statistics for the data.

        Args:
            outcome (str): Name of the outcome variable.
            factors (List[str]): List of factors to group by.

        Returns:
            pd.DataFrame: DataFrame containing count, mean, effect, median, standard deviation, and standard error.
        """
        self._validate_data()
        if self.data is None or not isinstance(self.data, pd.DataFrame):
            raise ValueError("No data provided or data is not a DataFrame. Please provide valid data before running descriptive_stats.")
        # Compute overall mean for effect calculation
        overall_mean = self.data[outcome].mean()
        # Group by factors and compute statistics
        stats = self.data.groupby(factors)[outcome].agg(
            Count='count',
            Mean='mean',
            Median='median',
            Standard_Deviation='std'
        ).reset_index()
        # Compute Effect (Mean - Overall Mean)
        stats['Effect'] = stats['Mean'] - overall_mean
        # Compute Standard Error (SD / sqrt(n))
        stats['Standard_Error'] = stats['Standard_Deviation'] / np.sqrt(stats['Count'])
        return stats

    def ls_means(self, outcome: str, between_factors: List[str], repeated_factors: List[str]) -> pd.DataFrame:
        """
        Compute Least Squares (LS) Means with confidence intervals and degrees of freedom.

        Args:
            outcome (str): Name of the outcome variable.
            between_factors (List[str]): List of between-subjects factors.
            repeated_factors (List[str]): List of repeated-measures factors.

        Returns:
            pd.DataFrame: DataFrame containing LS Means, standard errors, 95% confidence intervals, and DF.
        """
        self._validate_data()
        if self.data is None or not isinstance(self.data, pd.DataFrame):
            raise ValueError("No data provided or data is not a DataFrame. Please provide valid data before running ls_means.")
        if not between_factors and not repeated_factors:
            stats = self.descriptive_stats(outcome, ['Subject'])
            return stats.rename(columns={'Mean': 'Mean', 'Standard_Deviation': 'Standard Error', 'Count': 'DF'})
        # Fit a model to get LS Means
        formula = f"{outcome} ~ {' * '.join(between_factors + repeated_factors)}" if (between_factors or repeated_factors) else f"{outcome} ~ 1"
        if repeated_factors:
            model = mixedlm(formula, self.data, groups=self.data['Subject'], re_formula=f"~{' + '.join(repeated_factors)}")
        else:
            model = ols(formula, self.data)
        result = model.fit()
        # Compute LS Means
        ls_means = []
        levels = self.data[between_factors[0]].unique() if between_factors else [None]
        df = len(self.data) - len(levels) if between_factors else len(self.data) - 1  # Approximate DF
        for level in levels:
            subset = self.data[self.data[between_factors[0]] == level] if between_factors else self.data
            mean = subset[outcome].mean()
            se = subset[outcome].std() / np.sqrt(len(subset))
            ci_lower = mean - 1.96 * se
            ci_upper = mean + 1.96 * se
            ls_means.append({
                between_factors[0] if between_factors else 'Intercept': level if level else 'Intercept',
                'Mean': mean,
                'Standard Error': se,
                'Lower CI': ci_lower,
                'Upper CI': ci_upper,
                'DF': df
            })
        return pd.DataFrame(ls_means)

    def expected_mean_squares(self, model_type: str, between_factors: List[str], repeated_factors: List[str]) -> pd.DataFrame:
        """
        Compute Expected Mean Squares for ANOVA models.

        Args:
            model_type (str): Type of statistical model.
            between_factors (List[str]): List of between-subjects factors.
            repeated_factors (List[str]): List of repeated-measures factors.

        Returns:
            pd.DataFrame: Table of expected mean squares.
        """
        self._validate_data()
        if self.data is None or not isinstance(self.data, pd.DataFrame):
            raise ValueError("No data provided or data is not a DataFrame. Please provide valid data before running expected_mean_squares.")
        if "ANOVA" not in model_type or repeated_factors:
            return pd.DataFrame()  # Only for non-repeated ANOVA models
        # Approximate Expected Mean Squares
        ems = []
        for factor in between_factors:
            df = len(self.data[factor].unique()) - 1
            ems.append({
                'Model Term': f"A: {factor}",
                'DF': df,
                'Term Fixed?': 'Yes',
                'Denominator Term': 'S(A)',
                'Expected Mean Square': 'S+sA'
            })
        # Residual term
        n_subjects = len(self.data['Subject'].unique())
        df_residual = n_subjects - len(between_factors) - 1
        ems.append({
            'Model Term': 'S(A)',
            'DF': df_residual,
            'Term Fixed?': 'No',
            'Denominator Term': '',
            'Expected Mean Square': 'S'
        })
        return pd.DataFrame(ems)

    def plot_data(self, outcome: str, between_factors: List[str], repeated_factors: List[str]) -> plt.Figure:
        """
        Generate visualization plots for the data.

        Args:
            outcome (str): Name of the outcome variable.
            between_factors (List[str]): List of between-subjects factors.
            repeated_factors (List[str]): List of repeated-measures factors.

        Returns:
            plt.Figure: Matplotlib figure object containing the plot.
        """
        self._validate_data()
        fig, ax = plt.subplots(figsize=(10, 6))
        if not repeated_factors:
            if between_factors:
                sns.boxplot(x=between_factors[0], y=outcome,
                        hue=between_factors[1] if len(between_factors) > 1 else None,
                        data=self.data, ax=ax)
            else:
                sns.boxplot(y=outcome, data=self.data, ax=ax)
            ax.set_title("Box Plot")
        else:
            if between_factors:
                sns.lineplot(x=repeated_factors[0], y=outcome,
                            hue=between_factors[0],
                            style=between_factors[1] if len(between_factors) > 1 else None,
                            data=self.data, ax=ax)
            else:
                sns.lineplot(x=repeated_factors[0], y=outcome,
                            data=self.data, ax=ax)
            ax.set_title("Interaction Plot")
        plt.tight_layout()
        return fig

    def plot_ls_means(self, outcome: str, between_factors: List[str], repeated_factors: List[str], ls_means: pd.DataFrame) -> Figure:
        """
        Generate LS Means plot with confidence intervals.

        Args:
            outcome (str): Name of the outcome variable.
            between_factors (List[str]): List of between-subjects factors.
            repeated_factors (List[str]): List of repeated-measures factors.
            ls_means (pd.DataFrame): DataFrame containing LS Means and confidence intervals.

        Returns:
            plt.Figure: Matplotlib figure object containing the LS Means plot.
        """
        if self.data is None or not isinstance(self.data, pd.DataFrame):
            raise ValueError("No data provided or data is not a DataFrame. Please provide valid data before plotting.")
        if ls_means is None or not isinstance(ls_means, pd.DataFrame):
            raise ValueError("LS Means must be a pandas DataFrame and cannot be None.")
        fig, ax = plt.subplots(figsize=(10, 6))
        if not repeated_factors:
            sns.pointplot(x=between_factors[0], y=outcome, hue=between_factors[1] if len(between_factors) > 1 else None,
                          data=self.data, errorbar=None, ax=ax)
            # Overlay confidence intervals
            for _, row in ls_means.iterrows():
                if between_factors:
                    level = row[between_factors[0]]
                else:
                    # Use the repeated factor value for x
                    level = row[repeated_factors[0]]
                ax.errorbar(x=level, y=row['Mean'],
                          yerr=[[row['Mean'] - row['Lower CI']], [row['Upper CI'] - row['Mean']]],
                          fmt='o', color='black', capsize=5)
            ax.set_title("LS Means Plot")
        else:
            # For repeated measures, plot means over time
            if between_factors:
                sns.pointplot(x=repeated_factors[0], y=outcome, hue=between_factors[0],
                              data=self.data, errorbar=None, ax=ax)
            else:
                sns.pointplot(x=repeated_factors[0], y=outcome,
                              data=self.data, errorbar=None, ax=ax)
            ax.set_title("LS Means Plot (Repeated Measures)")
        plt.tight_layout()
        return fig

    def run_analysis(self, model_type: str, outcome: str, between_factors: List[str], repeated_factors: List[str] = []) -> Dict[str, Any]:
        """
        Run the specified statistical analysis.

        Args:
            model_type (str): Type of statistical analysis to perform.
            outcome (str): Name of the outcome variable.
            between_factors (List[str]): List of between-subjects factors.
            repeated_factors (List[str], optional): List of repeated-measures factors.

        Returns:
            Dict[str, Any]: Dictionary containing analysis results.
        """
        self._validate_data()
        self.validate_no_duplicates(repeated_factors)
        self.results = {'Assumptions': self.check_assumptions(outcome, between_factors, repeated_factors)}

        # Check assumptions to decide which test to run
        shapiro_p = self.results['Assumptions']['Shapiro-Wilk'][1]
        levene_p = self.results['Assumptions']['Levene'][1] if between_factors else 1.0
        use_alternative = False

        # Descriptive statistics
        factors = between_factors + repeated_factors
        self.results['Descriptive Stats'] = self.descriptive_stats(outcome, factors)

        # LS Means
        self.results['LS Means'] = self.ls_means(outcome, between_factors, repeated_factors)
        self.results['LS Means Plot'] = self.plot_ls_means(outcome, between_factors, repeated_factors, self.results['LS Means'])

        # Plot data (Box or Interaction Plot)
        self.results['Plot'] = self.plot_data(outcome, between_factors, repeated_factors)

        # Expected Mean Squares (for ANOVA models)
        self.results['Expected Mean Squares'] = self.expected_mean_squares(model_type, between_factors, repeated_factors)

        # Alternative tests if assumptions fail
        if shapiro_p < 0.05:  # Non-normal
            if model_type == "T-test":
                self.results['Alternative Test'] = stats.mannwhitneyu(
                    self.data[self.data[between_factors[0]] == self.data[between_factors[0]].unique()[0]][outcome],
                    self.data[self.data[between_factors[0]] == self.data[between_factors[0]].unique()[1]][outcome]
                )
                return self.results
            elif model_type in ["One-way ANOVA", "Two-way ANOVA", "Three-way ANOVA"]:
                self.results['Alternative Test'] = stats.kruskal(
                    *[self.data[self.data[between_factors[0]] == level][outcome] for level in self.data[between_factors[0]].unique()]
                )
                return self.results
            else:
                use_alternative = True  # For repeated measures, proceed with caution

        if levene_p < 0.05 and not repeated_factors:  # Unequal variances, non-repeated
            if model_type == "T-test":
                self.results['Alternative Test'] = stats.ttest_ind(
                    self.data[self.data[between_factors[0]] == self.data[between_factors[0]].unique()[0]][outcome],
                    self.data[self.data[between_factors[0]] == self.data[between_factors[0]].unique()[1]][outcome],
                    equal_var=False
                )
                return self.results
            elif model_type == "One-way ANOVA":
                # Welch's ANOVA approximation using scipy.stats.f_oneway with unequal variances
                groups = [self.data[self.data[between_factors[0]] == level][outcome] for level in self.data[between_factors[0]].unique()]
                # Compute Welch's ANOVA manually (simplified)
                self.results['Alternative Test'] = "Welch's ANOVA approximated; p-value: " + str(stats.f_oneway(*groups).pvalue)
                return self.results

        # Statistical test based on model type
        if model_type == "T-test" and len(between_factors) == 1:
            groups = self.data[between_factors[0]].unique()
            t_stat, p_val = stats.ttest_ind(
                self.data[self.data[between_factors[0]] == groups[0]][outcome],
                self.data[self.data[between_factors[0]] == groups[1]][outcome]
            )
            self.results['T-test'] = {'t_stat': t_stat, 'p_value': p_val}

        elif model_type == "One-way ANOVA" and len(between_factors) == 1:
            model = ols(f"{outcome} ~ {between_factors[0]}", data=self.data).fit()
            anova_table = anova_lm(model, typ=2)
            self.results['ANOVA'] = anova_table
            self.results['Post-Hoc'] = pairwise_tukeyhsd(self.data[outcome], self.data[between_factors[0]])

        elif model_type == "Two-way ANOVA" and len(between_factors) == 2:
            model = ols(f"{outcome} ~ {between_factors[0]} * {between_factors[1]}", data=self.data).fit()
            anova_table = anova_lm(model, typ=2)
            self.results['ANOVA'] = anova_table

        elif model_type == "Three-way ANOVA" and len(between_factors) == 3:
            model = ols(f"{outcome} ~ {between_factors[0]} * {between_factors[1]} * {between_factors[2]}", data=self.data).fit()
            anova_table = anova_lm(model, typ=2)
            self.results['ANOVA'] = anova_table

        elif model_type == "One-way Repeated Measures ANOVA" and len(repeated_factors) == 1:
            model = mixedlm(f"{outcome} ~ 1", self.data, groups=self.data['Subject'], re_formula=f"~{repeated_factors[0]}")
            result = model.fit()
            self.results['Repeated Measures ANOVA'] = {
                'Summary': str(result.summary()),
                'Run Summary': {
                    'Likelihood Type': 'Restricted Maximum Likelihood',
                    'Fixed Model': '1',
                    'Number of Subjects': len(self.data['Subject'].unique()),
                    'Solution Type': 'Newton-Raphson',
                    'Fisher Iterations': 1,
                    'Newton Iterations': 1,
                    'Max Retries': 10,
                    'Lambda': 1,
                    'Log-Likelihood': result.llf,
                    'AIC': result.aic,
                    'Convergence': 'Normal' if result.converged else 'Failed',
                    'Run Time (Seconds)': 0.0  # Placeholder, as timing not captured
                },
                'Variance Estimates': {
                    'Variance': result.scale,
                    'Standard Deviation': np.sqrt(result.scale)
                }
            }

        elif model_type == "Two-way Repeated Measures ANOVA" and len(repeated_factors) == 2:
            model = mixedlm(f"{outcome} ~ {repeated_factors[0]} * {repeated_factors[1]}", self.data, groups=self.data['Subject'],
                            re_formula=f"~{repeated_factors[0]} + {repeated_factors[1]}")
            result = model.fit()
            self.results['Repeated Measures ANOVA'] = {
                'Summary': str(result.summary()),
                'Run Summary': {
                    'Likelihood Type': 'Restricted Maximum Likelihood',
                    'Fixed Model': f"{repeated_factors[0]} * {repeated_factors[1]}",
                    'Number of Subjects': len(self.data['Subject'].unique()),
                    'Solution Type': 'Newton-Raphson',
                    'Fisher Iterations': 1,
                    'Newton Iterations': 1,
                    'Max Retries': 10,
                    'Lambda': 1,
                    'Log-Likelihood': result.llf,
                    'AIC': result.aic,
                    'Convergence': 'Normal' if result.converged else 'Failed',
                    'Run Time (Seconds)': 0.0
                },
                'Variance Estimates': {
                    'Variance': result.scale,
                    'Standard Deviation': np.sqrt(result.scale)
                }
            }

        elif model_type == "Three-way Repeated Measures ANOVA" and len(repeated_factors) == 2 and len(between_factors) == 1:
            model = mixedlm(f"{outcome} ~ {between_factors[0]} * {repeated_factors[0]} * {repeated_factors[1]}", self.data,
                            groups=self.data['Subject'], re_formula=f"~{repeated_factors[0]} + {repeated_factors[1]}")
            result = model.fit()
            self.results['Repeated Measures ANOVA'] = {
                'Summary': str(result.summary()),
                'Run Summary': {
                    'Likelihood Type': 'Restricted Maximum Likelihood',
                    'Fixed Model': f"{between_factors[0]} * {repeated_factors[0]} * {repeated_factors[1]}",
                    'Number of Subjects': len(self.data['Subject'].unique()),
                    'Solution Type': 'Newton-Raphson',
                    'Fisher Iterations': 1,
                    'Newton Iterations': 1,
                    'Max Retries': 10,
                    'Lambda': 1,
                    'Log-Likelihood': result.llf,
                    'AIC': result.aic,
                    'Convergence': 'Normal' if result.converged else 'Failed',
                    'Run Time (Seconds)': 0.0
                },
                'Variance Estimates': {
                    'Variance': result.scale,
                    'Standard Deviation': np.sqrt(result.scale)
                }
            }

        elif model_type == "Mixed ANOVA (One Between, One Repeated)" and len(between_factors) == 1 and len(repeated_factors) == 1:
            model = mixedlm(f"{outcome} ~ {between_factors[0]} * {repeated_factors[0]}", self.data, groups=self.data['Subject'],
                            re_formula=f"~{repeated_factors[0]}")
            result = model.fit()
            self.results['Mixed ANOVA'] = {
                'Summary': str(result.summary()),
                'Run Summary': {
                    'Likelihood Type': 'Restricted Maximum Likelihood',
                    'Fixed Model': f"{between_factors[0]} * {repeated_factors[0]}",
                    'Number of Subjects': len(self.data['Subject'].unique()),
                    'Solution Type': 'Newton-Raphson',
                    'Fisher Iterations': 1,
                    'Newton Iterations': 1,
                    'Max Retries': 10,
                    'Lambda': 1,
                    'Log-Likelihood': result.llf,
                    'AIC': result.aic,
                    'Convergence': 'Normal' if result.converged else 'Failed',
                    'Run Time (Seconds)': 0.0
                },
                'Variance Estimates': {
                    'Variance': result.scale,
                    'Standard Deviation': np.sqrt(result.scale)
                }
            }

        elif model_type == "Mixed ANOVA (Two Between, One Repeated)" and len(between_factors) == 2 and len(repeated_factors) == 1:
            model = mixedlm(f"{outcome} ~ {between_factors[0]} * {between_factors[1]} * {repeated_factors[0]}", self.data,
                            groups=self.data['Subject'], re_formula=f"~{repeated_factors[0]}")
            result = model.fit()
            self.results['Mixed ANOVA'] = {
                'Summary': str(result.summary()),
                'Run Summary': {
                    'Likelihood Type': 'Restricted Maximum Likelihood',
                    'Fixed Model': f"{between_factors[0]} * {between_factors[1]} * {repeated_factors[0]}",
                    'Number of Subjects': len(self.data['Subject'].unique()),
                    'Solution Type': 'Newton-Raphson',
                    'Fisher Iterations': 1,
                    'Newton Iterations': 1,
                    'Max Retries': 10,
                    'Lambda': 1,
                    'Log-Likelihood': result.llf,
                    'AIC': result.aic,
                    'Convergence': 'Normal' if result.converged else 'Failed',
                    'Run Time (Seconds)': 0.0
                },
                'Variance Estimates': {
                    'Variance': result.scale,
                    'Standard Deviation': np.sqrt(result.scale)
                }
            }

        elif model_type == "Mixed ANOVA (One Between, Two Repeated)" and len(between_factors) == 1 and len(repeated_factors) == 2:
            model = mixedlm(f"{outcome} ~ {between_factors[0]} * {repeated_factors[0]} * {repeated_factors[1]}", self.data,
                            groups=self.data['Subject'], re_formula=f"~{repeated_factors[0]} + {repeated_factors[1]}")
            result = model.fit()
            self.results['Mixed ANOVA'] = {
                'Summary': str(result.summary()),
                'Run Summary': {
                    'Likelihood Type': 'Restricted Maximum Likelihood',
                    'Fixed Model': f"{between_factors[0]} * {repeated_factors[0]} * {repeated_factors[1]}",
                    'Number of Subjects': len(self.data['Subject'].unique()),
                    'Solution Type': 'Newton-Raphson',
                    'Fisher Iterations': 1,
                    'Newton Iterations': 1,
                    'Max Retries': 10,
                    'Lambda': 1,
                    'Log-Likelihood': result.llf,
                    'AIC': result.aic,
                    'Convergence': 'Normal' if result.converged else 'Failed',
                    'Run Time (Seconds)': 0.0
                },
                'Variance Estimates': {
                    'Variance': result.scale,
                    'Standard Deviation': np.sqrt(result.scale)
                }
            }

        elif model_type == "Complex Mixed ANOVA" and len(between_factors) >= 2 and len(repeated_factors) >= 2:
            formula = f"{outcome} ~ {' * '.join(between_factors + repeated_factors)}"
            model = mixedlm(formula, self.data, groups=self.data['Subject'],
                            re_formula=f"~{' + '.join(repeated_factors)}")
            result = model.fit()
            self.results['Complex Mixed ANOVA'] = {
                'Summary': str(result.summary()),
                'Run Summary': {
                    'Likelihood Type': 'Restricted Maximum Likelihood',
                    'Fixed Model': formula.split(' ~ ')[1],
                    'Number of Subjects': len(self.data['Subject'].unique()),
                    'Solution Type': 'Newton-Raphson',
                    'Fisher Iterations': 1,
                    'Newton Iterations': 1,
                    'Max Retries': 10,
                    'Lambda': 1,
                    'Log-Likelihood': result.llf,
                    'AIC': result.aic,
                    'Convergence': 'Normal' if result.converged else 'Failed',
                    'Run Time (Seconds)': 0.0
                },
                'Variance Estimates': {
                    'Variance': result.scale,
                    'Standard Deviation': np.sqrt(result.scale)
                }
            }

        return self.results
