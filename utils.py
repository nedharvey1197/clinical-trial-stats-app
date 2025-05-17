import pandas as pd
import numpy as np
import io
import logging
from datetime import datetime
from matplotlib.figure import Figure
from typing import Tuple, Dict, Any, List

# Configure logging
logger = logging.getLogger(__name__)

def check_data_quality(data: pd.DataFrame, outcome: str, between_factors: list, repeated_factors: list) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Check data quality and return cleaned data with a quality report.

    Args:
        data (pd.DataFrame): Input data.
        outcome (str): Outcome variable name.
        between_factors (list): Between-subjects factors.
        repeated_factors (list): Repeated-measures factors.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned data and quality report.
    """
    logger.info(f"Checking data quality for outcome: {outcome}")
    logger.info(f"Data shape: {data.shape}")
    
    quality_report = {}
    
    if outcome not in data.columns:
        logger.error(f"Outcome column '{outcome}' not found in data")
        raise ValueError(f"Outcome column '{outcome}' not found in data.")
        
    missing_outcome = data[outcome].isnull().sum()
    quality_report['missing_outcome'] = f"Missing values in Outcome: {missing_outcome} ({missing_outcome / len(data) * 100:.2f}%)"
    logger.info(quality_report['missing_outcome'])
    
    if missing_outcome > 0:
        quality_report['missing_action'] = "Consider imputing or removing missing values."
        logger.warning(quality_report['missing_action'])

    # Check data types
    if not np.issubdtype(data[outcome].dtype, np.number):
        logger.error(f"Outcome '{outcome}' must be numeric, found type: {data[outcome].dtype}")
        raise ValueError(f"Outcome '{outcome}' must be numeric.")

    # Check for duplicates in repeated measures
    if repeated_factors:
        subset_cols = ['Subject'] + repeated_factors
        logger.info(f"Checking for duplicates in columns: {subset_cols}")
        duplicates = data.duplicated(subset=subset_cols).sum()
        quality_report['duplicates'] = f"Duplicate entries for {', '.join(subset_cols)}: {duplicates}"
        logger.info(quality_report['duplicates'])
        
        if duplicates > 0:
            logger.error(quality_report['duplicates'])
            raise ValueError(quality_report['duplicates'])

    # Outlier detection (using IQR method)
    Q1 = data[outcome].quantile(0.25)
    Q3 = data[outcome].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data[outcome] < (Q1 - 1.5 * IQR)) | (data[outcome] > (Q3 + 1.5 * IQR))][outcome]
    quality_report['outliers'] = f"Outliers detected: {len(outliers)} values outside 1.5*IQR range."
    logger.info(quality_report['outliers'])
    
    if len(outliers) > 0:
        logger.warning(f"Found {len(outliers)} outliers in {outcome}: {outliers.values}")

    logger.info("Data quality check completed")
    return data, quality_report

def impute_missing(data: pd.DataFrame, outcome: str, method: str = "mean") -> pd.DataFrame:
    """
    Impute missing values in the outcome variable.

    Args:
        data (pd.DataFrame): Input data.
        outcome (str): Outcome variable name.
        method (str): Imputation method ('mean', 'median').

    Returns:
        pd.DataFrame: Data with imputed values.
    """
    logger.info(f"Imputing missing values in {outcome} using method: {method}")
    
    missing_count = data[outcome].isnull().sum()
    logger.info(f"Missing values before imputation: {missing_count}")
    
    if not np.issubdtype(data[outcome].dtype, np.number):
        logger.error(f"Cannot impute non-numeric outcome '{outcome}', type: {data[outcome].dtype}")
        raise ValueError(f"Cannot impute non-numeric outcome '{outcome}'.")
        
    if method == "mean":
        mean_value = data[outcome].mean()
        logger.info(f"Imputing with mean value: {mean_value}")
        data[outcome] = data[outcome].fillna(mean_value)
    elif method == "median":
        median_value = data[outcome].median()
        logger.info(f"Imputing with median value: {median_value}")
        data[outcome] = data[outcome].fillna(median_value)
    else:
        logger.warning(f"Unknown imputation method: {method}, defaulting to mean")
        data[outcome] = data[outcome].fillna(data[outcome].mean())
        
    remaining_missing = data[outcome].isnull().sum()
    logger.info(f"Missing values after imputation: {remaining_missing}")
    
    return data

def transform_data(data: pd.DataFrame, outcome: str, method: str = "log") -> pd.DataFrame:
    """
    Apply data transformation to the outcome variable.

    Args:
        data (pd.DataFrame): Input data.
        outcome (str): Outcome variable name.
        method (str): Transformation method ('log', 'sqrt').

    Returns:
        pd.DataFrame: Transformed data.
    """
    logger.info(f"Transforming data for {outcome} using method: {method}")
    
    if not np.issubdtype(data[outcome].dtype, np.number):
        logger.error(f"Cannot transform non-numeric outcome '{outcome}'")
        raise ValueError(f"Cannot transform non-numeric outcome '{outcome}'.")
        
    if method == "log":
        if (data[outcome] < 0).any():
            logger.error("Log transformation requires non-negative values")
            raise ValueError("Log transformation requires non-negative values.")
        logger.info(f"Applying log1p transformation to {outcome}")
        data[outcome] = np.log1p(data[outcome])
    elif method == "sqrt":
        if (data[outcome] < 0).any():
            logger.error("Square root transformation requires non-negative values")
            raise ValueError("Square root transformation requires non-negative values.")
        logger.info(f"Applying square root transformation to {outcome}")
        data[outcome] = np.sqrt(data[outcome])
    else:
        logger.warning(f"Unknown transformation method: {method}, no transformation applied")
        
    logger.info("Data transformation completed")
    return data

def check_performance(data: pd.DataFrame) -> str:
    """
    Check dataset size and issue performance warnings.

    Args:
        data (pd.DataFrame): Input data.

    Returns:
        str: Warning message if dataset is large.
    """
    logger.info(f"Checking performance for dataset of size: {data.shape}")
    
    if len(data) > 10000:
        warning_msg = "Large dataset detected (>10,000 rows). Analysis may be slow. Consider sampling or optimizing your data."
        logger.warning(warning_msg)
        return warning_msg
        
    logger.info("No performance issues detected")
    return ""

def optimize_computation(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """
    Optimize data for computation by converting to numpy where possible.

    Args:
        data (pd.DataFrame): Input data.
        outcome (str): Outcome variable.

    Returns:
        pd.DataFrame: Optimized data.
    """
    logger.info(f"Optimizing computation for {outcome}")
    
    original_type = data[outcome].dtype
    data[outcome] = data[outcome].to_numpy()
    
    logger.info(f"Converted {outcome} from {original_type} to numpy array")
    return data

def export_table_to_csv(table: pd.DataFrame, filename: str) -> bytes:
    """
    Export a DataFrame to CSV.

    Args:
        table (pd.DataFrame): Table to export.
        filename (str): Base filename.

    Returns:
        bytes: CSV data.
    """
    logger.info(f"Exporting table to CSV: {filename}")
    logger.info(f"Table shape: {table.shape}, columns: {table.columns.tolist()}")
    
    csv_bytes = table.to_csv(index=False).encode('utf-8')
    logger.info(f"CSV export completed, size: {len(csv_bytes)} bytes")
    
    return csv_bytes

def export_plot_to_png(plot: Figure, filename: str) -> bytes:
    """
    Export a Matplotlib figure to PNG.

    Args:
        plot (Figure): Plot to export.
        filename (str): Base filename.

    Returns:
        bytes: PNG data.
    """
    logger.info(f"Exporting plot to PNG: {filename}")
    
    buf = io.BytesIO()
    plot.savefig(buf, format="png", bbox_inches="tight")
    png_bytes = buf.getvalue()
    
    logger.info(f"PNG export completed, size: {len(png_bytes)} bytes")
    return png_bytes

def generate_mock_data(n_subjects: int, factors: List[str], outcome_mean: float = 100.0, outcome_std: float = 10.0) -> pd.DataFrame:
    """
    Generate mock data for testing.
    
    Args:
        n_subjects (int): Number of subjects to generate.
        factors (List[str]): List of factor names to include.
        outcome_mean (float): Mean value for the outcome variable.
        outcome_std (float): Standard deviation for the outcome variable.
        
    Returns:
        pd.DataFrame: Generated mock data for testing.
    """
    logger.info(f"Generating mock data for {n_subjects} subjects with factors: {factors}")
    logger.info(f"Outcome parameters: mean={outcome_mean}, std={outcome_std}")
    
    data = {'Subject': list(range(1, n_subjects + 1))}
    
    for factor in factors:
        data[factor] = ['A' if i % 2 == 0 else 'B' for i in range(n_subjects)]
        logger.info(f"Generated factor '{factor}' with levels: A, B")
        
    data['Outcome'] = np.random.normal(outcome_mean, outcome_std, n_subjects)
    logger.info(f"Generated outcome variable with actual mean: {np.mean(data['Outcome']):.2f}, std: {np.std(data['Outcome']):.2f}")
    
    df = pd.DataFrame(data)
    logger.info(f"Mock data generation complete, shape: {df.shape}")
    
    return df