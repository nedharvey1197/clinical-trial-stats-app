# statistics_engine/modeling.py
from models.base_models import ClinicalTrialAnalysis
import pingouin as pg  # Fixed import for pingouin
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedClinicalTrialAnalysis(ClinicalTrialAnalysis):
    """
    Extends ClinicalTrialAnalysis with effect size calculations and clinical significance.
    """

    def __init__(self, data=None, mcid: float = None):
        """
        Initialize with optional MCID for clinical significance.

        Args:
            data (pd.DataFrame, optional): Input data.
            mcid (float, optional): Minimal Clinically Important Difference.
        """
        logger.info(f"Initializing EnhancedClinicalTrialAnalysis with mcid={mcid}")
        super().__init__(data)
        self.mcid = mcid  # Minimal Clinically Important Difference
        
        if data is not None:
            logger.info(f"Initialized with data shape: {data.shape}")
        if mcid is not None:
            logger.info(f"Minimal Clinically Important Difference (MCID) set to: {mcid}")

    def compute_effect_size(self, model_type: str, outcome: str, between_factors: List[str], results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute effect sizes for the analysis.

        Args:
            model_type (str): Model type.
            outcome (str): Outcome variable.
            between_factors (List[str]): Between-subjects factors.
            results (Dict[str, Any]): Analysis results.

        Returns:
            Dict[str, Any]: Effect size results.
        """
        logger.info(f"Computing effect size for model_type={model_type}, outcome={outcome}")
        logger.info(f"Between factors: {between_factors}")
        
        effect_sizes = {}
        
        if model_type == "T-test":
            logger.info("Computing Cohen's d effect size for T-test")
            
            if not between_factors:
                logger.error("No between-factors provided for T-test")
                effect_sizes['Cohen\'s d'] = "T-test requires at least one between-factor."
                return effect_sizes
                
            groups = self.data[between_factors[0]].unique()
            logger.info(f"T-test groups found: {groups}")
            
            if len(groups) != 2:
                logger.warning(f"T-test requires exactly 2 groups, found {len(groups)}")
                effect_sizes['Cohen\'s d'] = "T-test requires exactly 2 groups."
                return effect_sizes
                
            group1_data = self.data[self.data[between_factors[0]] == groups[0]][outcome]
            group2_data = self.data[self.data[between_factors[0]] == groups[1]][outcome]
            
            logger.info(f"Group '{groups[0]}' size: {len(group1_data)}, mean: {group1_data.mean():.2f}")
            logger.info(f"Group '{groups[1]}' size: {len(group2_data)}, mean: {group2_data.mean():.2f}")
            
            if len(group1_data) < 2 or len(group2_data) < 2:
                logger.warning("Each group must have at least 2 observations for Cohen's d")
                effect_sizes['Cohen\'s d'] = "Each group must have at least 2 observations for Cohen\'s d."
                return effect_sizes
                
            try:
                effect_sizes['Cohen\'s d'] = pg.compute_effsize(group1_data, group2_data, eftype='cohen')
                logger.info(f"Cohen's d effect size: {effect_sizes['Cohen\'s d']:.4f}")
                
                # Clinical significance assessment
                if self.mcid is not None:
                    logger.info(f"Assessing clinical significance with MCID: {self.mcid}")
                    mean_diff = abs(group1_data.mean() - group2_data.mean())
                    
                    if mean_diff >= self.mcid:
                        clinical_sig = "Clinically significant (difference exceeds MCID)"
                        logger.info(f"Mean difference {mean_diff:.2f} exceeds MCID {self.mcid}")
                    else:
                        clinical_sig = "Not clinically significant (difference below MCID)"
                        logger.info(f"Mean difference {mean_diff:.2f} below MCID {self.mcid}")
                        
                    effect_sizes['Clinical Significance'] = clinical_sig
            except Exception as e:
                logger.error(f"Error computing Cohen's d: {str(e)}")
                effect_sizes['Cohen\'s d'] = f"Error: {str(e)}"
                
        elif "ANOVA" in model_type:
            logger.info(f"Computing eta-squared effect size for {model_type}")
            
            anova_table = results.get('ANOVA', {})
            if not isinstance(anova_table, pd.DataFrame) or 'sum_sq' not in anova_table.columns:
                logger.error("ANOVA table not found or malformed")
                effect_sizes['Eta-squared'] = "ANOVA table not found or malformed."
                return effect_sizes
                
            logger.info(f"ANOVA table sum of squares: {anova_table['sum_sq'].values}")
            
            ss_between = anova_table['sum_sq'][0]
            ss_total = anova_table['sum_sq'].sum()
            
            logger.info(f"Sum of squares between: {ss_between}, Sum of squares total: {ss_total}")
            
            if ss_total == 0:
                logger.warning("Total sum of squares is zero; cannot compute eta-squared")
                effect_sizes['Eta-squared'] = "Total sum of squares is zero; cannot compute eta-squared."
                return effect_sizes
                
            effect_sizes['Eta-squared'] = ss_between / ss_total
            logger.info(f"Eta-squared effect size: {effect_sizes['Eta-squared']:.4f}")
            
            # Cohen's interpretation
            if effect_sizes['Eta-squared'] >= 0.14:
                interpretation = "Large effect"
            elif effect_sizes['Eta-squared'] >= 0.06:
                interpretation = "Medium effect"
            else:
                interpretation = "Small effect"
                
            logger.info(f"Effect size interpretation: {interpretation}")
            effect_sizes['Interpretation'] = interpretation
        else:
            logger.warning(f"Effect size calculation not implemented for model type: {model_type}")
            effect_sizes['Note'] = f"Effect size calculation not implemented for {model_type}"
            
        logger.info(f"Effect size computation completed: {effect_sizes}")
        return effect_sizes

    def run_analysis(self, model_type: str, outcome: str, between_factors: List[str], repeated_factors: List[str] = []) -> Dict[str, Any]:
        """
        Run the analysis and add effect sizes.

        Args:
            model_type (str): Type of statistical analysis.
            outcome (str): Name of the outcome variable.
            between_factors (List[str]): Between-subjects factors.
            repeated_factors (List[str]): Repeated-measures factors.

        Returns:
            Dict[str, Any]: Analysis results with effect sizes.
        """
        logger.info(f"Running enhanced analysis for model_type={model_type}, outcome={outcome}")
        logger.info(f"Between factors: {between_factors}, Repeated factors: {repeated_factors}")
        
        # Run base analysis
        logger.info("Executing parent class analysis method")
        results = super().run_analysis(model_type, outcome, between_factors, repeated_factors)
        logger.info("Base analysis completed")
        
        # Add effect sizes
        logger.info("Adding effect size calculations to results")
        results['Effect Sizes'] = self.compute_effect_size(model_type, outcome, between_factors, results)
        
        # Log key results
        if 'T-test' in results:
            logger.info(f"T-test results: t={results['T-test']['t_stat']:.2f}, p={results['T-test']['p_value']:.4f}")
        
        logger.info("Enhanced analysis completed")
        return results
    
    def mock_anova_table(self, ss_between: float, ss_total: float) -> pd.DataFrame:
        """Mock ANOVA table for testing."""
        logger.info(f"Mocking ANOVA table with ss_between={ss_between}, ss_total={ss_total}")
        return pd.DataFrame({
            'sum_sq': [ss_between, ss_total - ss_between],
            'df': [1, 10]
        })