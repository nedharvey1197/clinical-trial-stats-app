import scipy.stats as stats
import pandas as pd
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisLogic:
    """
    Handles decision logic for selecting statistical tests and running sensitivity analyses.
    """

    def __init__(self, model_type: str, outcome: str, between_factors: List[str], repeated_factors: List[str]):
        """
        Initialize the AnalysisLogic with model and factor details.

        Args:
            model_type (str): Type of statistical model (e.g., "T-test").
            outcome (str): Name of the outcome variable.
            between_factors (List[str]): List of between-subjects factors.
            repeated_factors (List[str]): List of repeated-measures factors.
        """
        self.model_type = model_type
        self.outcome = outcome
        self.between_factors = between_factors
        self.repeated_factors = repeated_factors
        logger.info(f"Initialized AnalysisLogic with model_type={model_type}, outcome={outcome}")
        logger.info(f"Between factors: {between_factors}, Repeated factors: {repeated_factors}")

    def decide_test(self, assumptions: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """
        Decide which tests to run based on assumption checks.

        Args:
            assumptions (Dict[str, Any]): Results of assumption tests.
            data (pd.DataFrame): Input data for analysis.

        Returns:
            Dict[str, Any]: Decision on primary and sensitivity tests.
        """
        logger.info(f"Deciding test for model: {self.model_type}")
        logger.info(f"Data shape: {data.shape} with columns: {data.columns.tolist()}")
        
        required_keys = ['Shapiro-Wilk']
        if self.between_factors:
            logger.info(f"Between-factors: {self.between_factors}")
            required_keys.append('Levene')
        if self.repeated_factors:
            logger.info(f"Repeated-factors: {self.repeated_factors}")
            required_keys.append('Mauchly-Sphericity')
        missing_keys = [key for key in required_keys if key not in assumptions]
        
        if missing_keys:
            logger.warning(f"Assumption check results missing keys: {missing_keys}")
            raise ValueError(f"Assumption check results missing keys: {missing_keys}")
        
        logger.info(f"Assumption check results: {assumptions}")
        shapiro_p = assumptions['Shapiro-Wilk'][1]
        logger.info(f"Shapiro-Wilk p-value: {shapiro_p}")
        
        levene_p = assumptions['Levene'][1] if self.between_factors else 1.0
        if self.between_factors:
            logger.info(f"Levene's test p-value: {levene_p}")
        
        sphericity_p = assumptions['Mauchly-Sphericity']['p_value'] if self.repeated_factors else 1.0
        if self.repeated_factors:
            logger.info(f"Mauchly's Sphericity test p-value: {sphericity_p}")
            
        decisions = {}

        # Primary test decision
        if shapiro_p < 0.05:
            decisions['primary_test'] = "non_parametric"
            logger.info(f"Primary test decision: non_parametric (Shapiro-Wilk p < 0.05)")
        elif levene_p < 0.05 and not self.repeated_factors:
            decisions['primary_test'] = "welch"
            logger.info(f"Primary test decision: welch (Levene p < 0.05)")
        else:
            decisions['primary_test'] = "parametric"
            logger.info(f"Primary test decision: parametric (assumptions met)")
            
        # Sensitivity test for marginal cases
        if 0.01 <= shapiro_p < 0.05 or 0.01 <= levene_p < 0.05 or 0.01 <= sphericity_p < 0.05:
            decisions['sensitivity_test'] = "both"  # Run both parametric and non-parametric
            logger.info(f"Sensitivity test decision: both (marginal p-values detected)")
            
            if 0.01 <= shapiro_p < 0.05:
                logger.info(f"Marginal normality (Shapiro-Wilk p={shapiro_p:.4f})")
            if 0.01 <= levene_p < 0.05:
                logger.info(f"Marginal homogeneity of variance (Levene p={levene_p:.4f})")
            if 0.01 <= sphericity_p < 0.05:
                logger.info(f"Marginal sphericity (Mauchly p={sphericity_p:.4f})")
        else:
            decisions['sensitivity_test'] = "none"
            logger.info(f"Sensitivity test decision: none (no marginal p-values)")
            
        return decisions

        """
        Run sensitivity tests for marginal cases.

        Args:
            data (pd.DataFrame): Input data.
            primary_test (str): Primary test type ('parametric', 'non_parametric', 'welch').

        Returns:
            Dict[str, Any]: Sensitivity test results.
        """ 
    def run_sensitivity_test(self, data: pd.DataFrame, primary_test: str) -> Dict[str, Any]:
        logger.info(f"Running sensitivity test for model: {self.model_type} with primary test: {primary_test}")
        logger.info(f"Data shape: {data.shape}")
        results = {}
        
        if self.model_type == "T-test":
            if not self.between_factors:
                logger.error("T-test requires at least one between-factor, none provided")
                results['non_parametric'] = "T-test requires at least one between-factor."
                return results
                
            groups = data[self.between_factors[0]].unique()
            logger.info(f"T-test groups found: {groups}")
            
            if len(groups) != 2:
                logger.warning(f"T-test requires exactly 2 groups, found {len(groups)}")
                results['non_parametric'] = "T-test requires exactly 2 groups."
                return results
                
            group1_data = data[data[self.between_factors[0]] == groups[0]][self.outcome]
            group2_data = data[data[self.between_factors[0]] == groups[1]][self.outcome]
            
            logger.info(f"Group '{groups[0]}' size: {len(group1_data)}, Group '{groups[1]}' size: {len(group2_data)}")
            
            if len(group1_data) < 2 or len(group2_data) < 2:
                logger.warning(f"Each group must have at least 2 observations for Mann-Whitney U test")
                results['non_parametric'] = "Each group must have at least 2 observations for Mann-Whitney U test."
                return results
                
            logger.info(f"Running Mann-Whitney U test between groups '{groups[0]}' and '{groups[1]}'")
            try:
                results['non_parametric'] = stats.mannwhitneyu(group1_data, group2_data)
                logger.info(f"Mann-Whitney U result: U={results['non_parametric'].statistic}, p={results['non_parametric'].pvalue}")
            except Exception as e:
                logger.error(f"Error running Mann-Whitney U test: {str(e)}")
                results['non_parametric'] = f"Error: {str(e)}"
                
        elif self.model_type in ["One-way ANOVA", "Two-way ANOVA", "Three-way ANOVA"]:
            if not self.between_factors:
                logger.error(f"{self.model_type} requires at least one between-factor, none provided")
                results['non_parametric'] = f"{self.model_type} requires at least one between-factor."
                return results
                
            groups = data[self.between_factors[0]].unique()
            logger.info(f"{self.model_type} groups in factor '{self.between_factors[0]}': {groups}")
            
            try:
                logger.info(f"Running Kruskal-Wallis test for {len(groups)} groups")
                group_data = [data[data[self.between_factors[0]] == level][self.outcome] for level in groups]
                for i, g in enumerate(group_data):
                    logger.info(f"Group {groups[i]} size: {len(g)}")
                
                results['non_parametric'] = stats.kruskal(*group_data)
                logger.info(f"Kruskal-Wallis result: H={results['non_parametric'].statistic}, p={results['non_parametric'].pvalue}")
            except Exception as e:
                logger.error(f"Error running Kruskal-Wallis test: {str(e)}")
                results['non_parametric'] = f"Error: {str(e)}"
                
        elif self.repeated_factors:
            logger.info(f"Running non-parametric test for repeated measures design")
            
            if len(self.repeated_factors) == 1:
                try:
                    logger.info(f"Pivoting data for Friedman test with factor '{self.repeated_factors[0]}'")
                    wide_data = data.pivot(index='Subject', columns=self.repeated_factors[0], values=self.outcome)
                    logger.info(f"Pivoted data shape: {wide_data.shape}, columns: {wide_data.columns.tolist()}")
                    
                    logger.info(f"Running Friedman test for {len(wide_data.columns)} time points")
                    results['non_parametric'] = stats.friedmanchisquare(*[wide_data[col] for col in wide_data.columns])
                    logger.info(f"Friedman test result: chi2={results['non_parametric'].statistic}, p={results['non_parametric'].pvalue}")
                except Exception as e:
                    logger.error(f"Error running Friedman test: {str(e)}")
                    results['non_parametric'] = f"Error: {str(e)}"
            else:
                logger.warning(f"Non-parametric test for {len(self.repeated_factors)} repeated factors not implemented")
                results['non_parametric'] = "Non-parametric test for multiple repeated factors not implemented."
        else:
            logger.warning(f"No sensitivity test implemented for model type: {self.model_type}")
            results['non_parametric'] = f"No sensitivity test implemented for model type: {self.model_type}"
            
        logger.info(f"Sensitivity test completed with results: {results}")
        return results
    
    def simulate_assumptions(self, shapiro_p: float, levene_p: float = 1.0, sphericity_p: float = 1.0) -> Dict[str, Any]:
        """Simulate assumption check results for testing."""
        assumptions = {
            'Shapiro-Wilk': (0.0, shapiro_p),
            'Levene': (0.0, levene_p) if self.between_factors else None,
            'Mauchly-Sphericity': {'W': 0.0, 'p_value': sphericity_p} if self.repeated_factors else None
        }
        return {k: v for k, v in assumptions.items() if v is not None}