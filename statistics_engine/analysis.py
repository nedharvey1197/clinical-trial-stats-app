from statistics_engine.logic import AnalysisLogic
from statistics_engine.utils import check_data_quality, impute_missing, transform_data, export_table_to_csv, export_plot_to_png, check_performance, optimize_computation
from statistics_engine.enhanced_models import EnhancedClinicalTrialAnalysis
from typing import Dict, Any, List, Tuple

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

    
    def run_quality_check(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run only the quality check stage for testing."""
        return check_data_quality(data, self.outcome, self.between_factors, self.repeated_factors)
    
    def run_pipeline(self, data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run the full analysis pipeline.

        Args:
            data (pd.DataFrame): Input data.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Analysis results and quality report.
        """
        # Step 1: Data Quality Check
        data, quality_report = check_data_quality(data, self.outcome, self.between_factors, self.repeated_factors)

        # Step 2: Performance Check and Optimization
        quality_report['performance_warning'] = check_performance(data)
        data = optimize_computation(data, self.outcome)

        # Step 3: Data Preprocessing (optional imputation/transformation)
        if data[self.outcome].isnull().any():
            data = impute_missing(data, self.outcome, method="mean")
            quality_report['imputation'] = "Missing values imputed using mean."

        # Step 4: Run Analysis
        analysis = EnhancedClinicalTrialAnalysis(data, self.mcid)
        results = analysis.run_analysis(self.model_type, self.outcome, self.between_factors, self.repeated_factors)

        # Step 5: Sensitivity Analysis
        decisions = self.logic.decide_test(results['Assumptions'], data)
        if decisions['sensitivity_test'] == "both":
            results['Sensitivity Test'] = self.logic.run_sensitivity_test(data, decisions['primary_test'])

        # Step 6: Add Exports
        results['Exports'] = {
            'Descriptive Stats CSV': export_table_to_csv(results['Descriptive Stats'], "descriptive_stats"),
            'LS Means CSV': export_table_to_csv(results['LS Means'], "ls_means"),
            'Plot PNG': export_plot_to_png(results['Plot'], "plot"),
            'LS Means Plot PNG': export_plot_to_png(results['LS Means Plot'], "ls_means_plot")
        }
        if results['Expected Mean Squares'].size > 0:
            results['Exports']['Expected Mean Squares CSV'] = export_table_to_csv(results['Expected Mean Squares'], "expected_mean_squares")

        return results, quality_report