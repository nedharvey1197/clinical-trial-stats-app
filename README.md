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
- `models/`: Directory containing model-specific code
- `Templates/`: Directory containing template files
- `requirements.txt`: Python package dependencies

---

# Clinical Trial Analysis App – Tester Guide

This Streamlit app allows you to explore and analyze clinical trial data using a variety of statistical models. You can test the app using built-in example datasets or your own data, either by uploading a CSV file or entering data directly.

---

## Getting Started

1. **Clone the repository and set up the environment** (see the main README for setup instructions).
2. **Run the app:**
   ```bash
   streamlit run clinical_trial_analysis_app.py
   ```

---

## How to Use the App

### 1. Select a Statistical Model

- Use the sidebar to choose the statistical model you want to test (e.g., T-test, One-way ANOVA, Mixed ANOVA, etc.).
- When you select a model, the app will show a description and a sample dataset for that model.

---

### 2. Explore with Canned (Example) Data

- After selecting a model, you'll see a preloaded example dataset tailored for that model.
- You can:
  - View the dataset in the main panel.
  - See explanations and workflow steps in the right panel.
  - Run the analysis on the example data (it may run automatically, or you can click "Run Canned Example").
- The app will display:
  - Assumption checks
  - Descriptive statistics
  - Least Squares Means
  - Plots and visualizations
  - Statistical test results
  - Any relevant warnings or corrections

---

### 3. Analyze Your Own Data

You have two options:

#### a) Upload a CSV File

- In the sidebar or main panel, choose "Upload CSV" as your data input method.
- Click the file uploader and select your CSV file.
- The file should include:
  - A `Subject` column (for repeated measures)
  - An `Outcome` column (your dependent variable)
  - Any relevant factor columns (e.g., Drug, Time, Group, etc.)
- After uploading, select the outcome and factor columns as prompted.
- Click "Run Analysis with Custom Data" to see results.

#### b) Enter Data Manually

- Choose "Manual Entry" as your data input method.
- Specify the number of groups and subjects.
- Enter values for each subject and factor as prompted.
- Click "Submit Data" to create your dataset.
- Click "Run Analysis with Custom Data" to analyze.

---

### 4. Download Templates and Results

- Download a CSV template for the selected model to help format your own data.
- Download the full analysis and explanation as an HTML file for sharing or review.

---

### 5. Review Warnings and Explanations

- The app displays statistical warnings (e.g., convergence issues, assumption violations) in the explanations panel for transparency.
- General guidance is provided to help interpret these warnings.

---

## Tips for Testers

- Try each model with the canned example to see expected outputs.
- Upload your own data or use the template to test custom scenarios.
- Experiment with manual entry for small datasets.
- Review the explanations and warnings to ensure they are clear and helpful.
- Report any issues, confusing outputs, or unexpected warnings.

---

## Need Help?

If you have questions or encounter issues, please open an issue on GitHub or contact the development team.
