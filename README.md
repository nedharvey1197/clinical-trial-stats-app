# Clinical Trial Analysis App

A Streamlit application for analyzing clinical trial data with various statistical models.

## Setup

1. Create a virtual environment:

```bash
python3 -m venv venv
```

2. Activate the virtual environment:

- On macOS/Linux:

```bash
source venv/bin/activate
```

- On Windows:

```bash
.\venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

1. Make sure your virtual environment is activated
2. Run the Streamlit app:

```bash
streamlit run clinical_trial_analysis_app.py
```

## Features

- Multiple statistical models for clinical trial analysis:

  - One-Way ANOVA
  - Two-Way ANOVA
  - Three-Way ANOVA
  - ANCOVA
  - Mixed Model
  - Repeated Measures ANOVA
  - MANOVA
  - MANCOVA
  - Mixed MANOVA
  - Mixed MANCOVA

- Data input options:

  - CSV file upload
  - Manual data entry
  - Example data

- Comprehensive analysis outputs:
  - Assumption checks
  - Descriptive statistics
  - LS Means
  - Expected Mean Squares
  - Statistical test results
  - Visualizations

## Usage

1. Select a statistical model from the sidebar
2. Choose your data input method:
   - Upload a CSV file
   - Enter data manually
   - Use example data
3. Configure model parameters
4. Run the analysis
5. View and download results

## File Structure

- `clinical_trial_analysis_app.py`: Main Streamlit application
- `10_models.py`: Statistical model implementations
- `models/`: Directory containing model-specific code
- `Templates/`: Directory containing template files
- `requirements.txt`: Python package dependencies
