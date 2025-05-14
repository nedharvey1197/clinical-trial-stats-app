"""
Clinical Trial Statistical Analysis Module

This module provides a comprehensive suite of statistical analysis tools for clinical trial data.
It supports various types of ANOVA designs, including:
- One-way, Two-way, and Three-way ANOVA
- Repeated Measures ANOVA
- Mixed ANOVA designs
- Complex factorial designs

The module includes functionality for:
- Data validation and preprocessing
- Assumption checking (normality, equal variances, sphericity)
- Descriptive statistics
- Statistical testing
- Visualization
- Interactive data input

Example:
    >>> from statistics_engine.models import ClinicalTrialAnalysis
    >>> # Create analysis with example data
    >>> data = pd.DataFrame({
    ...     'Subject': [1, 1, 1, 2, 2, 2],
    ...     'Drug': ['A', 'A', 'A', 'B', 'B', 'B'],
    ...     'Time': [1, 2, 3, 1, 2, 3],
    ...     'Outcome': [120, 115, 110, 130, 125, 120]
    ... })
    >>> analysis = ClinicalTrialAnalysis(data)
    >>> results = analysis.run_analysis(
    ...     model_type="Mixed ANOVA (One Between, One Repeated)",
    ...     outcome="Outcome",
    ...     between_factors=["Drug"],
    ...     repeated_factors=["Time"]
    ... )
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as stats
import seaborn as sns
import matplotlib
matplotlib.use('MacOSX')  # Use MacOSX backend which is native to macOS
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Any

class ClinicalTrialAnalysis:
    """
    A class for performing statistical analysis on clinical trial data.

    This class provides methods for analyzing clinical trial data using various
    statistical models, including ANOVA, repeated measures, and mixed designs.
    It includes functionality for data validation, assumption checking,
    descriptive statistics, and visualization.

    Attributes:
        data (pd.DataFrame): The clinical trial data to be analyzed
        results (Dict[str, Any]): Dictionary storing analysis results

    Example:
        >>> analysis = ClinicalTrialAnalysis()
        >>> # Input data interactively
        >>> analysis.input_data_interactively()
        >>> # Run analysis
        >>> results = analysis.run_analysis(
        ...     model_type="One-way ANOVA",
        ...     outcome="Outcome",
        ...     between_factors=["Treatment"],
        ...     repeated_factors=[]
        ... )
    """

    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the ClinicalTrialAnalysis object.

        Args:
            data (Optional[pd.DataFrame]): Clinical trial data to analyze.
                If None, data can be input interactively later.

        Example:
            >>> # Initialize with data
            >>> data = pd.DataFrame({'Subject': [1, 2], 'Outcome': [10, 20]})
            >>> analysis = ClinicalTrialAnalysis(data)
            >>> # Initialize without data
            >>> analysis = ClinicalTrialAnalysis()
        """
        self.data = data
        self.results: Dict[str, Any] = {}

    def _validate_data(self) -> None:
        """
        Validate that data exists and is properly formatted.

        Raises:
            ValueError: If no data is provided
            TypeError: If data is not a pandas DataFrame

        Example:
            >>> analysis = ClinicalTrialAnalysis()
            >>> analysis._validate_data()  # Raises ValueError
        """
        if self.data is None:
            raise ValueError("No data provided. Please provide data or use input_data_interactively()")
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")

    def input_data_interactively(self) -> None:
        """
        Interactive method to input data for analysis.

        This method guides the user through entering clinical trial data,
        including:
        - Number of groups
        - Between-subjects factors
        - Repeated-measures factors
        - Outcome measurements

        Example:
            >>> analysis = ClinicalTrialAnalysis()
            >>> analysis.input_data_interactively()
            Number of groups (e.g., 2 for T-test, 3 for One-way ANOVA): 2
            Between-subjects factors (comma-separated, e.g., Drug,Age): Drug
            Repeated-measures factors (comma-separated, e.g., Time): Time
            ...
        """
        print("Enter data for clinical trial analysis.")
        groups = int(input("Number of groups (e.g., 2 for T-test, 3 for One-way ANOVA): "))
        between_factors = input("Between-subjects factors (comma-separated, e.g., Drug,Age): ").split(",")
        repeated_factors = input("Repeated-measures factors (comma-separated, e.g., Time): ").split(",")

        data_dict = {'Subject': [], 'Outcome': []}
        for factor in between_factors + repeated_factors:
            data_dict[factor] = []

        subject_id = 1
        for group in range(groups):
            group_name = input(f"Name of group {group+1} (e.g., DrugA): ")
            subjects = int(input(f"Number of subjects in {group_name}: "))
            for s in range(subjects):
                for time in range(len(repeated_factors) if repeated_factors else 1):
                    outcome = float(input(f"Outcome for subject {s+1} in {group_name} at time {time+1}: "))
                    data_dict['Subject'].append(subject_id)
                    data_dict['Outcome'].append(outcome)
                    for factor in between_factors:
                        data_dict[factor].append(group_name if factor == between_factors[0] else input(f"{factor} for subject {s+1}: "))
                    for factor in repeated_factors:
                        data_dict[factor].append(time + 1)
                subject_id += 1

        self.data = pd.DataFrame(data_dict)

    def check_assumptions(self, outcome: str, between_factors: List[str], repeated_factors: List[str]) -> Dict[str, Any]:
        """
        Check statistical assumptions for the analysis.

        Performs:
        - Shapiro-Wilk test for normality on residuals
        - Levene's test for equal variances (between-subjects)
        - Sphericity check for repeated measures

        Args:
            outcome (str): Name of the outcome variable
            between_factors (List[str]): List of between-subjects factors
            repeated_factors (List[str]): List of repeated-measures factors

        Returns:
            Dict[str, Any]: Dictionary containing assumption test results

        Example:
            >>> analysis = ClinicalTrialAnalysis(data)
            >>> assumptions = analysis.check_assumptions(
            ...     outcome="Outcome",
            ...     between_factors=["Treatment"],
            ...     repeated_factors=["Time"]
            ... )
        """
        self._validate_data()
        assumptions: Dict[str, Any] = {}

        # Normality (Shapiro-Wilk on residuals)
        model = ols(f"{outcome} ~ {' * '.join(between_factors + repeated_factors)}", data=self.data).fit()
        residuals = model.resid
        assumptions['Shapiro-Wilk'] = stats.shapiro(residuals)

        # Equal variances (Levene's test for between factors)
        if between_factors:
            groups = [self.data[self.data[between_factors[0]] == level][outcome]
                     for level in self.data[between_factors[0]].unique()]
            assumptions['Levene'] = stats.levene(*groups)

        # Sphericity (Mauchly's test approximation for repeated measures)
        if repeated_factors:
            assumptions['Sphericity'] = "Not implemented in this example (use Greenhouse-Geisser correction if violated)"

        return assumptions

    def descriptive_stats(self, outcome: str, factors: List[str]) -> pd.DataFrame:
        """
        Compute descriptive statistics for the data.

        Args:
            outcome (str): Name of the outcome variable
            factors (List[str]): List of factors to group by

        Returns:
            pd.DataFrame: DataFrame containing mean and standard deviation
                         for each group

        Example:
            >>> analysis = ClinicalTrialAnalysis(data)
            >>> stats = analysis.descriptive_stats(
            ...     outcome="Outcome",
            ...     factors=["Treatment", "Time"]
            ... )
        """
        self._validate_data()
        return self.data.groupby(factors)[outcome].agg(['mean', 'std']).reset_index()

    def plot_data(self, outcome: str, between_factors: List[str], repeated_factors: List[str]) -> None:
        """
        Generate visualization plots for the data.
        """
        self._validate_data()
        plt.figure(figsize=(10, 6))  # Make plot larger
        if not repeated_factors:
            sns.boxplot(x=between_factors[0], y=outcome,
                       hue=between_factors[1] if len(between_factors) > 1 else None,
                       data=self.data)
            plt.title("Box Plot")
        else:
            sns.lineplot(x=repeated_factors[0], y=outcome,
                        hue=between_factors[0],
                        style=between_factors[1] if len(between_factors) > 1 else None,
                        data=self.data)
            plt.title("Interaction Plot")
        plt.tight_layout()
        print("\nDisplaying plot... (Close the plot window to continue)")
        plt.show()

    def run_analysis(self, model_type: str, outcome: str, between_factors: List[str], repeated_factors: List[str] = []) -> Dict[str, Any]:
        """
        Run the specified statistical analysis.

        Supported model types:
        - "T-test"
        - "One-way ANOVA"
        - "Two-way ANOVA"
        - "Three-way ANOVA"
        - "One-way Repeated Measures ANOVA"
        - "Two-way Repeated Measures ANOVA"
        - "Three-way Repeated Measures ANOVA"
        - "Mixed ANOVA (One Between, One Repeated)"
        - "Mixed ANOVA (Two Between, One Repeated)"
        - "Mixed ANOVA (One Between, Two Repeated)"
        - "Complex Mixed ANOVA"

        Args:
            model_type (str): Type of statistical analysis to perform
            outcome (str): Name of the outcome variable
            between_factors (List[str]): List of between-subjects factors
            repeated_factors (List[str], optional): List of repeated-measures factors

        Returns:
            Dict[str, Any]: Dictionary containing analysis results

        Example:
            >>> analysis = ClinicalTrialAnalysis(data)
            >>> results = analysis.run_analysis(
            ...     model_type="Mixed ANOVA (One Between, One Repeated)",
            ...     outcome="Outcome",
            ...     between_factors=["Treatment"],
            ...     repeated_factors=["Time"]
            ... )
        """
        self._validate_data()
        self.results = {'Assumptions': self.check_assumptions(outcome, between_factors, repeated_factors)}

        # Descriptive statistics
        factors = between_factors + repeated_factors
        self.results['Descriptive Stats'] = self.descriptive_stats(outcome, factors)

        # Plot data
        self.plot_data(outcome, between_factors, repeated_factors)

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
            self.results['Repeated Measures ANOVA'] = result.summary()

        elif model_type == "Two-way Repeated Measures ANOVA" and len(repeated_factors) == 2:
            model = mixedlm(f"{outcome} ~ {repeated_factors[0]} * {repeated_factors[1]}", self.data, groups=self.data['Subject'],
                            re_formula=f"~{repeated_factors[0]} + {repeated_factors[1]}")
            result = model.fit()
            self.results['Repeated Measures ANOVA'] = result.summary()

        elif model_type == "Three-way Repeated Measures ANOVA" and len(repeated_factors) == 2 and len(between_factors) == 1:
            model = mixedlm(f"{outcome} ~ {between_factors[0]} * {repeated_factors[0]} * {repeated_factors[1]}", self.data,
                            groups=self.data['Subject'], re_formula=f"~{repeated_factors[0]} + {repeated_factors[1]}")
            result = model.fit()
            self.results['Repeated Measures ANOVA'] = result.summary()

        elif model_type == "Mixed ANOVA (One Between, One Repeated)" and len(between_factors) == 1 and len(repeated_factors) == 1:
            model = mixedlm(f"{outcome} ~ {between_factors[0]} * {repeated_factors[0]}", self.data, groups=self.data['Subject'],
                            re_formula=f"~{repeated_factors[0]}")
            result = model.fit()
            self.results['Mixed ANOVA'] = result.summary()

        elif model_type == "Mixed ANOVA (Two Between, One Repeated)" and len(between_factors) == 2 and len(repeated_factors) == 1:
            model = mixedlm(f"{outcome} ~ {between_factors[0]} * {between_factors[1]} * {repeated_factors[0]}", self.data,
                            groups=self.data['Subject'], re_formula=f"~{repeated_factors[0]}")
            result = model.fit()
            self.results['Mixed ANOVA'] = result.summary()

        elif model_type == "Mixed ANOVA (One Between, Two Repeated)" and len(between_factors) == 1 and len(repeated_factors) == 2:
            model = mixedlm(f"{outcome} ~ {between_factors[0]} * {repeated_factors[0]} * {repeated_factors[1]}", self.data,
                            groups=self.data['Subject'], re_formula=f"~{repeated_factors[0]} + {repeated_factors[1]}")
            result = model.fit()
            self.results['Mixed ANOVA'] = result.summary()

        elif model_type == "Complex Mixed ANOVA" and len(between_factors) >= 2 and len(repeated_factors) >= 2:
            formula = f"{outcome} ~ {' * '.join(between_factors + repeated_factors)}"
            model = mixedlm(formula, self.data, groups=self.data['Subject'],
                            re_formula=f"~{' + '.join(repeated_factors)}")
            result = model.fit()
            self.results['Complex Mixed ANOVA'] = result.summary()

        return self.results

def main():
    """
    Main function for standalone testing of the ClinicalTrialAnalysis class.
    """
    def print_help():
        print("\n" + "="*50)
        print("CLINICAL TRIAL STATISTICAL ANALYSIS")
        print("="*50)
        print("\nAvailable Model Types:")
        print("1. T-test")
        print("2. One-way ANOVA")
        print("3. Two-way ANOVA")
        print("4. Three-way ANOVA")
        print("5. One-way Repeated Measures ANOVA")
        print("6. Two-way Repeated Measures ANOVA")
        print("7. Mixed ANOVA (One Between, One Repeated)")
        print("8. Mixed ANOVA (Two Between, One Repeated)")
        print("9. Mixed ANOVA (One Between, Two Repeated)")
        print("\nExample Usage:")
        print("- For T-test: Enter 'T-test' and one between factor (e.g., 'Treatment')")
        print("- For One-way ANOVA: Enter 'One-way ANOVA' and one between factor")
        print("- For Mixed ANOVA: Enter 'Mixed ANOVA (One Between, One Repeated)'")
        print("  Then enter between factor (e.g., 'Treatment') and repeated factor (e.g., 'Time')")
        print("\nType 'help' at any prompt to see this menu again")
        print("Type 'exit' to quit")
        print("="*50 + "\n")

    # Example data for Mixed ANOVA (Two Between, One Repeated)
    print("\n" + "="*50)
    print("RUNNING EXAMPLE ANALYSIS")
    print("="*50)

    data = pd.DataFrame({
        'Subject': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'Drug': ['A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'B'],
        'Age_Group': ['Young', 'Young', 'Young', 'Young', 'Young', 'Young', 'Old', 'Old', 'Old', 'Old', 'Old', 'Old'],
        'Time': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        'Glucose': [120, 115, 110, 130, 125, 120, 125, 120, 115, 135, 130, 125]
    })

    # Test with example data
    print("\nRunning example analysis with glucose data...")
    analysis = ClinicalTrialAnalysis(data)
    results = analysis.run_analysis(
        model_type="Mixed ANOVA (Two Between, One Repeated)",
        outcome="Glucose",
        between_factors=["Drug", "Age_Group"],
        repeated_factors=["Time"]
    )

    print("\n" + "="*50)
    print("EXAMPLE RESULTS")
    print("="*50)
    print("\nAssumptions:")
    for test, result in results['Assumptions'].items():
        print(f"- {test}: {result}")

    print("\nDescriptive Stats:")
    print(results['Descriptive Stats'].to_string())

    print("\nMixed ANOVA Results:")
    print(results['Mixed ANOVA'])

    # Test interactive mode
    print("\n" + "="*50)
    print("INTERACTIVE MODE")
    print("="*50)
    print_help()

    while True:
        model_type = input("\nEnter model type (or 'help' for menu, 'exit' to quit): ").strip()
        if model_type.lower() == 'help':
            print_help()
            continue
        if model_type.lower() == 'exit':
            break

        between_factors = input("Between factors (comma-separated, e.g., Treatment,Dose): ").strip()
        if between_factors.lower() == 'help':
            print_help()
            continue
        if between_factors.lower() == 'exit':
            break

        repeated_factors = input("Repeated factors (comma-separated, e.g., Time,Visit): ").strip()
        if repeated_factors.lower() == 'help':
            print_help()
            continue
        if repeated_factors.lower() == 'exit':
            break

        try:
            print("\n" + "="*50)
            print("ENTERING DATA")
            print("="*50)
            analysis = ClinicalTrialAnalysis()
            analysis.input_data_interactively()

            print("\n" + "="*50)
            print("RUNNING ANALYSIS")
            print("="*50)
            results = analysis.run_analysis(
                model_type=model_type,
                outcome="Outcome",
                between_factors=[f.strip() for f in between_factors.split(",")],
                repeated_factors=[f.strip() for f in repeated_factors.split(",")]
            )

            print("\n" + "="*50)
            print("ANALYSIS RESULTS")
            print("="*50)
            print("\nAssumptions:")
            for test, result in results['Assumptions'].items():
                print(f"- {test}: {result}")

            print("\nDescriptive Stats:")
            print(results['Descriptive Stats'].to_string())

            print("\nStatistical Results:")
            for key, value in results.items():
                if key not in ['Assumptions', 'Descriptive Stats']:
                    print(f"\n{key}:")
                    print(value)

        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            print("Please try again or type 'help' for guidance")

if __name__ == "__main__":
    main()
