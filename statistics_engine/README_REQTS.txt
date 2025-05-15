Requirements Document for Backend of Clinical Trial Statistical Analysis Application
Version: 1.0

Date: May 14, 2025

Prepared by: Expert Biostatistician (50 years of experience)

Project: Clinical Trial Statistical Analysis Tool

Scope: Backend functionality for statistical analysis of clinical trial data, designed to be reusable across Streamlit and React Copilot platforms.

1. Overview
1.1 Purpose
The backend (statistics_engine package) provides the core functionality for a clinical trial statistical analysis application. It supports a variety of statistical models (e.g., T-test, ANOVA, Mixed ANOVA) used in Phase II/III clinical trials, ensuring robust data preprocessing, assumption checking, test selection, analysis execution, and result generation. The backend is designed to be modular, reusable, and independent of the frontend framework, enabling integration with both Streamlit and React Copilot platforms.

1.2 Scope
The backend includes the following modules:

logic.py: Decision logic for test selection and sensitivity analysis.
utils.py: Utility functions for data preprocessing, validation, performance optimization, and result export.
modeling.py: Statistical modeling, extending the base ClinicalTrialAnalysis class with effect size and clinical significance features.
analysis.py: Orchestrates the analysis pipeline (data → assumptions → test → results).
data.py: Stores canned example datasets with contextual descriptions.
The backend builds on the existing base_models.py, which provides the core ClinicalTrialAnalysis class for statistical modeling.

1.3 Stakeholders
Clinical Trial Analysts: End-users who need robust statistical analysis for trial data.
Developers: Team responsible for integrating the backend into Streamlit and React Copilot platforms.
Regulatory Bodies: Require accurate, reproducible results for trial submissions (e.g., FDA, EMA).
Sponsor Organizations: Need reliable tools to support trial design and analysis.
2. Functional Requirements
2.1 General Requirements
FR-1: Modularity and Reusability
The backend must be structured as a modular Python package (statistics_engine) with distinct modules for logic, utilities, modeling, analysis, and data.
All modules must be independent of the frontend framework (e.g., Streamlit, React), allowing reuse in multiple applications.
Modules must be importable and usable via API endpoints (e.g., Flask/FastAPI) for React Copilot integration.
FR-2: Compatibility with Base Models
The backend must integrate with the existing base_models.py file, specifically the ClinicalTrialAnalysis class, without modifying its core functionality.
Enhancements (e.g., effect sizes) must be implemented via inheritance to preserve the original implementation.
FR-3: Support for Statistical Models
The backend must support the following statistical models:
T-test
One-way ANOVA
Two-way ANOVA
Three-way ANOVA
One-way Repeated Measures ANOVA
Two-way Repeated Measures ANOVA
Three-way Repeated Measures ANOVA (one between, two repeated)
Mixed ANOVA (One Between, One Repeated)
Mixed ANOVA (Two Between, One Repeated)
Mixed ANOVA (One Between, Two Repeated)
Complex Mixed ANOVA (≥2 Between, ≥2 Repeated)
2.2 Module-Specific Requirements
2.2.1 Logic Module (logic.py)
FR-4: Test Selection Logic

The module must implement a decision tree to select the appropriate statistical test based on assumption checks:
Normality (Shapiro-Wilk): If p < 0.05, use non-parametric test (e.g., Mann-Whitney U, Kruskal-Wallis).
Equal Variances (Levene’s): If p < 0.05 and no repeated factors, use Welch’s test.
Sphericity (Mauchly’s): If p < 0.05, note that Greenhouse-Geisser correction is applied in mixed models.
The decision tree must handle marginal cases (p-values between 0.01 and 0.05) by flagging them for sensitivity analysis.
FR-5: Sensitivity Analysis

The module must support running both parametric and non-parametric tests for marginal cases:
T-test: Mann-Whitney U.
ANOVA: Kruskal-Wallis.
Repeated Measures ANOVA (one factor): Friedman Test.
Repeated Measures ANOVA (multiple factors): Flag as not implemented (future enhancement: GLMM).
Results must be stored in a Sensitivity Test key in the output dictionary.
2.2.2 Utilities Module (utils.py)
FR-6: Data Quality Checks

The module must validate data for:
Missing values in the outcome variable (report percentage missing).
Non-numeric outcome values (raise an error).
Duplicates in repeated-measures data (raise an error if found).
Outliers using the IQR method (1.5*IQR rule, report count of outliers).
Output a quality report as a dictionary with keys: missing_outcome, missing_action, duplicates, outliers.
FR-7: Data Preprocessing

The module must support:
Imputation of missing values in the outcome variable (methods: mean, median).
Data transformations (log, square root) to stabilize variances or normalize data.
FR-8: Performance Optimization

The module must:
Issue a warning for large datasets (>10,000 rows).
Optimize computation by converting outcome data to NumPy arrays where applicable.
FR-9: Result Export

The module must support exporting:
Tables (e.g., descriptive stats, LS Means) as CSV.
Plots (e.g., box plots, interaction plots) as PNG.
Exported data must be returned as bytes for integration with frontend download functionality.
2.2.3 Modeling Module (modeling.py)
FR-10: Effect Size Calculation

The module must compute effect sizes for supported models:
T-test: Cohen’s d (using pingouin.compute_effsize).
ANOVA models: Eta-squared (SS_between / SS_total).
Results must be stored in the Effect Sizes key of the output dictionary.
FR-11: Clinical Significance

The module must compare effect sizes against a Minimal Clinically Important Difference (MCID) threshold.
MCID must be configurable via the class constructor (default: None).
Output a clinical significance statement (e.g., "Difference (5.2) exceeds MCID (5.0)") in the Effect Sizes dictionary.
FR-12: Inheritance from Base Models

The module must extend ClinicalTrialAnalysis from base_models.py using inheritance.
The enhanced class (EnhancedClinicalTrialAnalysis) must preserve all original functionality while adding effect size and clinical significance features.
2.2.4 Analysis Module (analysis.py)
FR-13: Pipeline Orchestration

The module must orchestrate the full analysis pipeline:
Data quality check (using utils.check_data_quality).
Performance check and optimization (using utils.check_performance, optimize_computation).
Data preprocessing (imputation, transformation if needed).
Statistical analysis (using EnhancedClinicalTrialAnalysis).
Sensitivity analysis (using logic.AnalysisLogic).
Result export preparation (using utils.export_table_to_csv, export_plot_to_png).
Output a tuple containing the analysis results (dictionary) and quality report (dictionary).
FR-14: Integration with Other Modules

The module must integrate with:
logic.AnalysisLogic for test selection and sensitivity analysis.
utils for data preprocessing and export.
modeling.EnhancedClinicalTrialAnalysis for statistical analysis.
2.2.5 Data Module (data.py)
FR-15: Canned Examples with Descriptions

The module must provide canned example datasets for all supported models (11 models).
Each dataset must include a contextual description (string) describing a realistic clinical trial scenario.
Datasets must be stored as a dictionary (canned_examples_with_desc) with keys: data (DataFrame), description (str).
FR-16: Factor Mappings

The module must provide mappings for between-subjects and repeated-measures factors for each model:
between_factors_dict: Dictionary mapping model types to between-subjects factors.
repeated_factors_dict: Dictionary mapping model types to repeated-measures factors.
2.3 Output Requirements
FR-17: Analysis Results

The backend must produce the following outputs for each analysis:
Assumption Checks: Shapiro-Wilk, Levene’s, Mauchly’s results (with p-values).
Descriptive Statistics: Table with count, mean, effect, median, standard deviation, standard error.
LS Means: Table with means, standard errors, 95% confidence intervals, degrees of freedom.
Expected Mean Squares: Table for ANOVA models (model term, DF, term fixed, denominator term, expected mean square).
Statistical Test Results: T-test (t-statistic, p-value), ANOVA (F-value, p-value, DF), Mixed ANOVA (summary, fit statistics).
Effect Sizes: Cohen’s d (T-test), eta-squared (ANOVA), with clinical significance statement.
Plots: Box plot (non-repeated), interaction plot (repeated measures), LS Means plot.
Sensitivity Test: Results of non-parametric tests for marginal cases.
Exports: CSV for tables, PNG for plots.
FR-18: Quality Report

The backend must produce a quality report with:
Missing data statistics.
Duplicate data detection.
Outlier detection.
Performance warnings.
Imputation/transformations applied (if any).
3. Non-Functional Requirements
3.1 Performance
NFR-1: The backend must process datasets with up to 10,000 rows within 5 seconds on a standard machine (e.g., 2.5 GHz CPU, 8 GB RAM).
NFR-2: For datasets larger than 10,000 rows, the backend must issue a performance warning and proceed with optimized computation (e.g., using NumPy).
3.2 Scalability
NFR-3: The backend must support high-concurrency usage (e.g., 100 simultaneous users) when integrated with a Flask/FastAPI server, with minimal latency (<1 second per request for datasets <1,000 rows).
NFR-4: The backend must be stateless, allowing horizontal scaling in a server environment.
3.3 Reliability
NFR-5: The backend must handle invalid inputs gracefully, raising appropriate errors (e.g., non-numeric outcome, duplicate data).
NFR-6: The backend must produce numerically accurate results, matching SAS outputs within rounding error (e.g., ±0.01 for p-values).
3.4 Maintainability
NFR-7: The codebase must be modular, with each module having a single responsibility (e.g., logic, utils, modeling).
NFR-8: All functions must include docstrings with purpose, arguments, returns, and examples.
NFR-9: The backend must use type hints for all function signatures to improve code readability and IDE support.
3.5 Compatibility
NFR-10: The backend must be compatible with Python 3.8+.
NFR-11: The backend must work with the following dependencies (versions as of May 14, 2025):
pandas>=2.0.0
numpy>=1.24.0
statsmodels>=0.14.0
scipy>=1.10.0
seaborn>=0.12.0
matplotlib>=3.7.0
pingouin>=0.5.0
3.6 Security
NFR-12: The backend must not store or log sensitive data (e.g., patient identifiers) unless explicitly required.
NFR-13: Input data must be validated to prevent injection attacks (e.g., ensure column names are safe strings).
4. Constraints
4.1 Dependencies
The backend relies on base_models.py for the core ClinicalTrialAnalysis class, which must remain unchanged.
The backend must use pingouin for effect size calculations, as statsmodels does not provide built-in support for Cohen’s d or eta-squared.
4.2 Limitations
Non-parametric tests for repeated measures with multiple factors (e.g., Two-way Repeated Measures ANOVA) are not fully implemented due to complexity; a placeholder message must be returned.
Welch’s ANOVA is approximated using scipy.stats.f_oneway; a more robust implementation (e.g., via pingouin.welch_anova) is recommended for future versions.
5. Assumptions
Input data is expected to be in a clean format (e.g., no special characters in column names, consistent data types).
Users will provide MCID values when clinical significance is required; otherwise, a default message will be displayed.
The backend assumes that the frontend will handle user input validation (e.g., ensuring numeric inputs for MCID).
6. Risks and Mitigations
Risk-1: Performance Bottlenecks with Large Datasets
Impact: Slow analysis for datasets >10,000 rows.
Mitigation: Implement performance warnings and optimizations (e.g., NumPy conversions). Future enhancements can include parallel processing with multiprocessing.
Risk-2: Numerical Discrepancies with SAS
Impact: Results may differ slightly due to algorithm differences (e.g., optimization in statsmodels vs. SAS).
Mitigation: Validate results against SAS for canned examples and document acceptable tolerances (±0.01 for p-values).
Risk-3: Missing Non-Parametric Alternatives
Impact: Lack of alternatives for repeated measures models with multiple factors.
Mitigation: Flag as not implemented and recommend GLMM in future versions.
7. Future Enhancements
FE-1: Implement robust Welch’s ANOVA using pingouin.welch_anova.
FE-2: Add non-parametric alternatives for repeated measures with multiple factors (e.g., GLMM via statsmodels.GLM).
FE-3: Support power analysis for trial design (e.g., using statsmodels.stats.power).
FE-4: Integrate with CDISC standards for direct data import from clinical trial databases.
8. Acceptance Criteria
The backend must pass unit tests for all supported models, ensuring results match SAS outputs within rounding error.
The backend must process a dataset of 10,000 rows in <5 seconds on a standard machine.
The backend must correctly handle assumption violations (e.g., non-normal data, unequal variances) by switching to appropriate alternative tests.
The backend must generate all required outputs (assumption checks, descriptive stats, LS Means, plots, etc.) as specified in FR-17.
The backend must be integrated into the Streamlit app and a Flask API, demonstrating reusability across platforms.
9. Glossary
MCID: Minimal Clinically Important Difference, a threshold for clinical significance.
Shapiro-Wilk Test: Tests for normality of residuals (p < 0.05 indicates non-normal data).
Levene’s Test: Tests for equal variances (p < 0.05 indicates unequal variances).
Mauchly’s Test: Tests for sphericity in repeated measures (p < 0.05 indicates violation).
Cohen’s d: Effect size for T-test, measuring standardized difference between means.