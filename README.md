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
2. Run the app launcher to choose which application to start:

```bash
streamlit run app_launcher.py
```

Alternatively, you can directly run either the main analysis app or the interactive dashboard:

```bash
# Run the main analysis app
streamlit run clinical_trial_analysis_app.py

# Run the interactive dashboard
streamlit run interactive_dashboard.py
```

## Features

### Main Analysis App
- Multiple statistical models for clinical trial analysis:

  - T-test
  - One-Way ANOVA
  - Two-Way ANOVA
  - Three-Way ANOVA
  - Repeated Measures ANOVA
  - Mixed ANOVA variants
  - and more

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

### Interactive Dashboard
- Advanced interactive visualizations:
  - Interactive boxplots
  - Distribution plots with kernel density estimates
  - Q-Q plots for normality assessment
  - Interactive interaction plots with confidence intervals
  - Residual diagnostic plots

- Real-time data exploration:
  - Dynamic filtering
  - Multiple visualization options
  - Customizable statistical parameters

- Enhanced statistical outputs:
  - Effect size calculations (Cohen's d, Eta-squared)
  - Clinical significance assessment
  - Detailed assumption testing
  - Interpretation guidance

## Usage

### Main Analysis App
1. Select a statistical model from the sidebar
2. Choose your data input method:
   - Upload a CSV file
   - Enter data manually
   - Use example data
3. Configure model parameters
4. Run the analysis
5. View and download results

### Interactive Dashboard
1. Select a data source (canned example or custom upload)
2. Choose variables and factors to visualize
3. Select visualization types to display
4. Configure statistical options (significance level, MCID)
5. Generate visualizations and explore the data
6. Download results and settings

## File Structure

- `app_launcher.py`: Entry point to choose which app to run
- `clinical_trial_analysis_app.py`: Main Streamlit application
- `interactive_dashboard.py`: Interactive data exploration dashboard
- `models/`: Directory containing model-specific code
  - `base_models.py`: Core statistical models
- `statistics_engine/`: Advanced statistical functionality
  - `enhanced_models.py`: Models with effect size calculations
  - `visualization.py`: Interactive visualization tools
  - `utils.py`: Utility functions for data processing
- `knowledge_dashboard/`: Sentinel knowledge integration
- `requirements.txt`: Python package dependencies

---

# Clinical Trial Analysis App – Tester Guide

This Streamlit app allows you to explore and analyze clinical trial data using a variety of statistical models. You can test the app using built-in example datasets or your own data, either by uploading a CSV file or entering data directly.

---

## Getting Started

1. **Clone the repository and set up the environment** (see the main README for setup instructions).
2. **Run the app launcher:**
   ```bash
   streamlit run app_launcher.py
   ```

---

## How to Use the Main Analysis App

### 1. Select a Statistical Model

- Use the sidebar to choose the statistical model you want to test (e.g., T-test, One-way ANOVA, Mixed ANOVA, etc.).
- When you select a model, the app will show a description and a sample dataset for that model.

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

## How to Use the Interactive Dashboard

### 1. Select a Data Source

- Choose between using a canned example dataset or uploading your own data.
- If uploading, provide a CSV file with similar structure to the examples (Subject, Outcome, factor columns).

### 2. Configure Visualization Options

- Select the outcome variable you want to analyze.
- Choose primary and secondary grouping factors.
- Select which types of visualizations to display.

### 3. Statistical Options

- Choose whether to show advanced statistics.
- Set custom significance levels and clinically important difference thresholds.

### 4. Explore Visualizations

- Generate visualizations and view them in the main panel.
- Hover over elements for detailed information.
- View statistical summaries and test results.
- Download data and settings for future use.

---

## Download Templates and Results

- Download a CSV template for the selected model to help format your own data.
- Download the full analysis and explanation as an HTML file for sharing or review.
- Export visualization settings as JSON for reproducibility.

---

## Tips for Testers

- Try each model with the canned example to see expected outputs.
- Upload your own data or use the template to test custom scenarios.
- Experiment with manual entry for small datasets.
- Review the explanations and warnings to ensure they are clear and helpful.
- Report any issues, confusing outputs, or unexpected warnings.
- Test both apps (main analysis and interactive dashboard) with the same data to compare results.

---

## Need Help?

If you have questions or encounter issues, please open an issue on GitHub or contact the development team.
