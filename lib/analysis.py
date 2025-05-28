from .logic import AnalysisLogic
from .utils import check_data_quality, impute_missing, transform_data, export_table_to_csv, export_plot_to_png, check_performance, optimize_computation
from .enhanced_models import EnhancedClinicalTrialAnalysis
from typing import Dict, Any, List, Tuple
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisOrchestrator:
    """
    Orchestrates the clinical trial analysis pipeline.
    """

    def __init__(self, model_type: str, outcome: str, between_factors: List[str], repeated_factors: List[str], mcid: float = None):
        """
        Initialize the orchestrator with model and factor details.

        Args:
            model_type (str): Type of statistical model.
            outcome (str): Outcome variable name.
            between_factors (List[str]): Between-subjects factors.
            repeated_factors (List[str]): Repeated-measures factors.
            mcid (float, optional): Minimal Clinically Important Difference.
        """
        self.model_type = model_type
        self.outcome = outcome
        self.between_factors = between_factors
        self.repeated_factors = repeated_factors
        self.mcid = mcid
        self.logic = AnalysisLogic(model_type, outcome, between_factors, repeated_factors)

    def run_pipeline(self, data: pd.DataFrame, imputation_method: str = "mean") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run the full analysis pipeline.

        Args:
            data (pd.DataFrame): Input data.
            imputation_method (str): Method to handle missing values ('mean', 'median', 'none'). Default is 'mean'.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Analysis results and quality report.

        Raises:
            ValueError: If the imputation method is invalid or data quality checks fail.
        """
        logger.info(f"Starting analysis pipeline for model: {self.model_type}")
        # Validate imputation method
        valid_methods = ["mean", "median", "none"]
        if imputation_method not in valid_methods:
            raise ValueError(f"Invalid imputation method: {imputation_method}. Must be one of {valid_methods}.")

        # Step 1: Data Quality Check
        try:
            data, quality_report = check_data_quality(data, self.outcome, self.between_factors, self.repeated_factors)
        except Exception as e:
            logger.error(f"Data quality check failed: {str(e)}")
            raise ValueError(f"Data quality check failed: {str(e)}")

        # Step 2: Performance Check and Optimization
        quality_report['performance_warning'] = check_performance(data)
        data = optimize_computation(data, self.outcome)

        # Step 3: Data Preprocessing
        if data[self.outcome].isnull().any():
            logger.info(f"Missing values detected in '{self.outcome}'. Applying imputation method: {imputation_method}")
            if imputation_method != "none":
                try:
                    data = impute_missing(data, self.outcome, method=imputation_method)
                    quality_report['imputation'] = f"Missing values imputed using {imputation_method}."
                except Exception as e:
                    logger.error(f"Imputation failed: {str(e)}")
                    raise ValueError(f"Imputation failed: {str(e)}")
            else:
                quality_report['imputation'] = "Missing values detected but imputation skipped (method='none')."

        # Step 4: Run Analysis
        try:
            analysis = EnhancedClinicalTrialAnalysis(data, self.mcid)
            results = analysis.run_analysis(self.model_type, self.outcome, self.between_factors, self.repeated_factors)
        except Exception as e:
            logger.error(f"Statistical analysis failed: {str(e)}")
            raise ValueError(f"Statistical analysis failed: {str(e)}")

        # Step 5: Sensitivity Analysis
        try:
            decisions = self.logic.decide_test(results['Assumptions'], data)
            if decisions['sensitivity_test'] == "both":
                results['Sensitivity Test'] = self.logic.run_sensitivity_test(data, decisions['primary_test'])
        except Exception as e:
            logger.error(f"Sensitivity analysis failed: {str(e)}")
            raise ValueError(f"Sensitivity analysis failed: {str(e)}")

        # Step 6: Add Exports
        try:
            results['Exports'] = {
                'Descriptive Stats CSV': export_table_to_csv(results['Descriptive Stats'], "descriptive_stats"),
                'LS Means CSV': export_table_to_csv(results['LS Means'], "ls_means"),
                'Plot PNG': export_plot_to_png(results['Plot'], "plot"),
                'LS Means Plot PNG': export_plot_to_png(results['LS Means Plot'], "ls_means_plot")
            }
            if results['Expected Mean Squares'].size > 0:
                results['Exports']['Expected Mean Squares CSV'] = export_table_to_csv(results['Expected Mean Squares'], "expected_mean_squares")
        except Exception as e:
            logger.error(f"Export preparation failed: {str(e)}")
            raise ValueError(f"Export preparation failed: {str(e)}")

        logger.info("Pipeline completed successfully.")
        return results, quality_report

    def run_quality_check(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run only the quality check stage for testing.

        Args:
            data (pd.DataFrame): Input data.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Processed data and quality report.
        """
        return check_data_quality(data, self.outcome, self.between_factors, self.repeated_factors)