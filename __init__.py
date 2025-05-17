"""
Clinical Trial Statistics Engine package.

This package provides statistical analysis tools and visualization features
for clinical trial data.
"""

__version__ = "0.2.0"

# Import key components for easy access
from .logic import AnalysisLogic
from .utils import check_data_quality, impute_missing, transform_data, export_table_to_csv, export_plot_to_png, check_performance, optimize_computation
from .enhanced_models import EnhancedClinicalTrialAnalysis
from .analysis import AnalysisOrchestrator
from .data import canned_examples_with_desc
from .visualization import AdvancedVisualization