import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
import logging
import scipy.stats as stats

# Configure logging
logger = logging.getLogger(__name__)

class AdvancedVisualization:
    """
    Advanced visualization tools for clinical trial data analysis.
    Provides interactive and static visualizations for better data exploration.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the visualization module.
        
        Args:
            data (pd.DataFrame, optional): Clinical trial data to visualize.
        """
        logger.info("Initializing AdvancedVisualization module")
        self.data = data
        
    def set_data(self, data: pd.DataFrame) -> None:
        """
        Set or update the data for visualization.
        
        Args:
            data (pd.DataFrame): Data to visualize.
        """
        logger.info(f"Setting visualization data with shape: {data.shape}")
        self.data = data
        
    def create_interactive_boxplot(self, outcome: str, 
                                 factors: List[str], 
                                 color_by: Optional[str] = None) -> go.Figure:
        """
        Create an interactive boxplot for outcome by factors.
        
        Args:
            outcome (str): Outcome variable name.
            factors (List[str]): Grouping factors.
            color_by (str, optional): Factor to use for color coding.
            
        Returns:
            go.Figure: Plotly figure object.
        """
        logger.info(f"Creating interactive boxplot for {outcome} by {factors}")
        
        if self.data is None:
            logger.error("No data available for visualization")
            raise ValueError("No data set for visualization. Use set_data() first.")
            
        if len(factors) == 0:
            logger.warning("No factors provided for grouping")
            return go.Figure()
            
        # Handle single or multiple factors
        if len(factors) == 1:
            fig = px.box(self.data, x=factors[0], y=outcome, 
                       color=color_by if color_by else factors[0],
                       title=f"Distribution of {outcome} by {factors[0]}",
                       labels={outcome: f"{outcome} Value", factors[0]: f"{factors[0]}"},
                       points="all")
        else:
            # For multiple factors, create a combined grouping variable
            self.data['_combined_factors'] = self.data[factors].apply(
                lambda row: ' - '.join(row.values.astype(str)), axis=1
            )
            fig = px.box(self.data, x='_combined_factors', y=outcome,
                       color=color_by if color_by else factors[0],
                       title=f"Distribution of {outcome} by {' & '.join(factors)}",
                       labels={outcome: f"{outcome} Value", '_combined_factors': "Factor Combinations"},
                       points="all")
        
        fig.update_layout(
            xaxis_title=' & '.join(factors),
            yaxis_title=outcome,
            legend_title=color_by if color_by else factors[0],
            font=dict(family="Arial, sans-serif", size=12),
            hovermode="closest"
        )
        
        logger.info("Interactive boxplot created successfully")
        return fig
    
    def create_interactive_interaction_plot(self, outcome: str, 
                                          x_factor: str, 
                                          trace_factor: str) -> go.Figure:
        """
        Create an interactive interaction plot.
        
        Args:
            outcome (str): Outcome variable name.
            x_factor (str): Factor for x-axis.
            trace_factor (str): Factor for different traces.
            
        Returns:
            go.Figure: Plotly figure object.
        """
        logger.info(f"Creating interaction plot for {outcome}: {x_factor} × {trace_factor}")
        
        if self.data is None:
            logger.error("No data available for visualization")
            raise ValueError("No data set for visualization. Use set_data() first.")
            
        # Compute means for each combination of factors
        aggregated = self.data.groupby([x_factor, trace_factor])[outcome].agg(['mean', 'std', 'count']).reset_index()
        aggregated['se'] = aggregated['std'] / np.sqrt(aggregated['count'])
        aggregated['ci_low'] = aggregated['mean'] - 1.96 * aggregated['se']
        aggregated['ci_high'] = aggregated['mean'] + 1.96 * aggregated['se']
        
        # Create the interaction plot
        fig = go.Figure()
        
        for trace_val in sorted(aggregated[trace_factor].unique()):
            trace_data = aggregated[aggregated[trace_factor] == trace_val]
            
            fig.add_trace(go.Scatter(
                x=trace_data[x_factor],
                y=trace_data['mean'],
                error_y=dict(
                    type='data',
                    array=1.96 * trace_data['se'],
                    visible=True
                ),
                mode='lines+markers',
                name=f"{trace_factor}={trace_val}",
                hovertemplate=f"{trace_factor}={trace_val}<br>{x_factor}=%{{x}}<br>{outcome}=%{{y:.2f}}<br>95% CI: [%{{customdata[0]:.2f}}, %{{customdata[1]:.2f}}]",
                customdata=np.stack((trace_data['ci_low'], trace_data['ci_high']), axis=-1)
            ))
        
        fig.update_layout(
            title=f"Interaction Plot: {outcome} by {x_factor} and {trace_factor}",
            xaxis_title=x_factor,
            yaxis_title=f"Mean {outcome}",
            legend_title=trace_factor,
            hovermode="closest",
            font=dict(family="Arial, sans-serif", size=12)
        )
        
        logger.info("Interactive interaction plot created successfully")
        return fig
    
    def create_distribution_plot(self, outcome: str, 
                               group_by: Optional[str] = None) -> go.Figure:
        """
        Create an interactive distribution plot for the outcome.
        
        Args:
            outcome (str): Outcome variable.
            group_by (str, optional): Group distributions by this factor.
            
        Returns:
            go.Figure: Plotly figure object.
        """
        logger.info(f"Creating distribution plot for {outcome}, grouped by {group_by}")
        
        if self.data is None:
            logger.error("No data available for visualization")
            raise ValueError("No data set for visualization. Use set_data() first.")
            
        fig = go.Figure()
        
        if group_by is not None:
            for group_val in sorted(self.data[group_by].unique()):
                group_data = self.data[self.data[group_by] == group_val][outcome]
                
                fig.add_trace(go.Histogram(
                    x=group_data,
                    name=f"{group_by}={group_val}",
                    opacity=0.7,
                    histnorm='probability density',
                    nbinsx=20
                ))
                
                # Add KDE plot
                kde_x = np.linspace(min(group_data), max(group_data), 100)
                kde = sns.kdeplot(group_data, bw_adjust=0.5).get_lines()[0].get_data()
                fig.add_trace(go.Scatter(
                    x=kde[0],
                    y=kde[1],
                    mode='lines',
                    name=f"KDE {group_by}={group_val}",
                    line=dict(width=2)
                ))
                
                plt.close()  # Close the seaborn plot
        else:
            fig.add_trace(go.Histogram(
                x=self.data[outcome],
                opacity=0.7,
                histnorm='probability density',
                nbinsx=20
            ))
            
            # Add KDE plot
            kde_x = np.linspace(min(self.data[outcome]), max(self.data[outcome]), 100)
            kde = sns.kdeplot(self.data[outcome], bw_adjust=0.5).get_lines()[0].get_data()
            fig.add_trace(go.Scatter(
                x=kde[0],
                y=kde[1],
                mode='lines',
                name="KDE",
                line=dict(width=2)
            ))
            
            plt.close()  # Close the seaborn plot
            
        fig.update_layout(
            title=f"Distribution of {outcome}" + (f" by {group_by}" if group_by else ""),
            xaxis_title=outcome,
            yaxis_title="Density",
            barmode='overlay',
            hovermode="closest",
            font=dict(family="Arial, sans-serif", size=12)
        )
        
        logger.info("Distribution plot created successfully")
        return fig
    
    def create_qq_plot(self, outcome: str, 
                     group_by: Optional[str] = None) -> go.Figure:
        """
        Create Q-Q plot for checking normality.
        
        Args:
            outcome (str): Outcome variable to check.
            group_by (str, optional): Create separate Q-Q plots by group.
            
        Returns:
            go.Figure: Plotly figure object.
        """
        logger.info(f"Creating Q-Q plot for {outcome}, grouped by {group_by}")
        
        if self.data is None:
            logger.error("No data available for visualization")
            raise ValueError("No data set for visualization. Use set_data() first.")
            
        if group_by is not None:
            # Create subplots for each group
            groups = sorted(self.data[group_by].unique())
            fig = make_subplots(rows=1, cols=len(groups), 
                               subplot_titles=[f"{group_by}={val}" for val in groups],
                               shared_yaxes=True)
            
            for i, group_val in enumerate(groups):
                group_data = self.data[self.data[group_by] == group_val][outcome].dropna()
                
                if len(group_data) > 1:  # Need at least 2 points
                    # Calculate theoretical quantiles
                    quantiles = np.linspace(0, 1, len(group_data) + 1)[1:-1]
                    theoretical_quantiles = stats.norm.ppf(quantiles)
                    
                    # Sort the data
                    sorted_data = np.sort(group_data)
                    
                    # Add the Q-Q plot
                    fig.add_trace(
                        go.Scatter(
                            x=theoretical_quantiles,
                            y=sorted_data,
                            mode='markers',
                            name=f"{group_by}={group_val}"
                        ),
                        row=1, col=i+1
                    )
                    
                    # Add the reference line
                    z = np.polyfit(theoretical_quantiles, sorted_data, 1)
                    line_x = np.array([min(theoretical_quantiles), max(theoretical_quantiles)])
                    line_y = z[0] * line_x + z[1]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=line_x,
                            y=line_y,
                            mode='lines',
                            name='Reference Line',
                            line=dict(color='red', width=2)
                        ),
                        row=1, col=i+1
                    )
        else:
            # Single Q-Q plot
            data_values = self.data[outcome].dropna()
            
            # Calculate theoretical quantiles
            quantiles = np.linspace(0, 1, len(data_values) + 1)[1:-1]
            theoretical_quantiles = stats.norm.ppf(quantiles)
            
            # Sort the data
            sorted_data = np.sort(data_values)
            
            fig = go.Figure()
            
            # Add the Q-Q plot
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sorted_data,
                    mode='markers',
                    name='Data'
                )
            )
            
            # Add the reference line
            z = np.polyfit(theoretical_quantiles, sorted_data, 1)
            line_x = np.array([min(theoretical_quantiles), max(theoretical_quantiles)])
            line_y = z[0] * line_x + z[1]
            
            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode='lines',
                    name='Reference Line',
                    line=dict(color='red', width=2)
                )
            )
        
        fig.update_layout(
            title=f"Q-Q Plot for {outcome}" + (f" by {group_by}" if group_by else ""),
            xaxis_title="Theoretical Quantiles",
            yaxis_title=outcome,
            hovermode="closest",
            showlegend=True,
            font=dict(family="Arial, sans-serif", size=12)
        )
        
        logger.info("Q-Q plot created successfully")
        return fig

    def create_residual_plots(self, model, outcome: str) -> go.Figure:
        """
        Create residual diagnostic plots for a fitted model.
        
        Args:
            model: Fitted statistical model.
            outcome (str): Outcome variable name.
            
        Returns:
            go.Figure: Plotly figure with residual plots.
        """
        logger.info(f"Creating residual plots for {outcome} model")
        
        # Extract residuals and fitted values
        residuals = model.resid
        fitted = model.fittedvalues
        
        # Create subplots: 2x2 grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Residuals vs Fitted",
                "Residuals Q-Q Plot",
                "Scale-Location Plot",
                "Residuals vs Leverage"
            ]
        )
        
        # 1. Residuals vs Fitted
        fig.add_trace(
            go.Scatter(
                x=fitted,
                y=residuals,
                mode='markers',
                name='Residuals',
                hovertemplate='Fitted: %{x:.2f}<br>Residual: %{y:.2f}'
            ),
            row=1, col=1
        )
        
        # Add horizontal line at y=0
        fig.add_trace(
            go.Scatter(
                x=[min(fitted), max(fitted)],
                y=[0, 0],
                mode='lines',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Residuals Q-Q Plot
        quantiles = np.linspace(0, 1, len(residuals) + 1)[1:-1]
        theoretical_quantiles = stats.norm.ppf(quantiles)
        sorted_residuals = np.sort(residuals)
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sorted_residuals,
                mode='markers',
                name='Q-Q',
                hovertemplate='Theoretical: %{x:.2f}<br>Residual: %{y:.2f}'
            ),
            row=1, col=2
        )
        
        # Add reference line
        z = np.polyfit(theoretical_quantiles, sorted_residuals, 1)
        line_x = np.array([min(theoretical_quantiles), max(theoretical_quantiles)])
        line_y = z[0] * line_x + z[1]
        
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=line_y,
                mode='lines',
                line=dict(color='red'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Scale-Location Plot (standardized residuals vs fitted)
        std_resid = np.sqrt(np.abs(residuals / np.std(residuals)))
        
        fig.add_trace(
            go.Scatter(
                x=fitted,
                y=std_resid,
                mode='markers',
                name='Scale-Location',
                hovertemplate='Fitted: %{x:.2f}<br>√|Std. Resid.|: %{y:.2f}'
            ),
            row=2, col=1
        )
        
        # 4. Residuals vs Leverage
        # Simplified version since leverage calculation depends on model type
        leverage = np.ones_like(residuals) / len(residuals)  # Placeholder
        
        fig.add_trace(
            go.Scatter(
                x=leverage,
                y=residuals,
                mode='markers',
                name='Leverage',
                hovertemplate='Leverage: %{x:.4f}<br>Residual: %{y:.2f}'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Residual Diagnostics for {outcome} Model",
            showlegend=False,
            height=800,
            font=dict(family="Arial, sans-serif", size=12)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Fitted Values", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Residuals", row=1, col=2)
        
        fig.update_xaxes(title_text="Fitted Values", row=2, col=1)
        fig.update_yaxes(title_text="√|Standardized Residuals|", row=2, col=1)
        
        fig.update_xaxes(title_text="Leverage", row=2, col=2)
        fig.update_yaxes(title_text="Residuals", row=2, col=2)
        
        logger.info("Residual plots created successfully")
        return fig

    def test_normality(self, outcome: str, group_by: Optional[str] = None) -> pd.DataFrame:
        """
        Perform normality tests on the outcome variable.
        
        Args:
            outcome (str): Outcome variable to test.
            group_by (str, optional): Group to test by.
            
        Returns:
            pd.DataFrame: Results of normality tests.
        """
        logger.info(f"Testing normality for {outcome}, grouped by {group_by}")
        
        if self.data is None:
            logger.error("No data available for visualization")
            raise ValueError("No data set for visualization. Use set_data() first.")
            
        results = []
        
        if group_by is not None:
            for group_val in sorted(self.data[group_by].unique()):
                group_data = self.data[self.data[group_by] == group_val][outcome].dropna()
                
                if len(group_data) >= 3:  # Need at least 3 points for some tests
                    # Shapiro-Wilk test
                    shapiro_stat, shapiro_p = stats.shapiro(group_data)
                    
                    # D'Agostino's K^2 test
                    k2_stat, k2_p = stats.normaltest(group_data)
                    
                    # Anderson-Darling test
                    anderson_result = stats.anderson(group_data, dist='norm')
                    # Get p-value from critical values
                    anderson_p = 0.05  # Approximate
                    
                    results.append({
                        'Group': f"{group_by}={group_val}",
                        'n': len(group_data),
                        'Shapiro-Wilk_stat': shapiro_stat,
                        'Shapiro-Wilk_p': shapiro_p,
                        'Normal (p>0.05)': shapiro_p > 0.05,
                        'K2_stat': k2_stat,
                        'K2_p': k2_p,
                        'Anderson_stat': anderson_result.statistic
                    })
                else:
                    results.append({
                        'Group': f"{group_by}={group_val}",
                        'n': len(group_data),
                        'Shapiro-Wilk_stat': None,
                        'Shapiro-Wilk_p': None,
                        'Normal (p>0.05)': None,
                        'K2_stat': None,
                        'K2_p': None,
                        'Anderson_stat': None
                    })
        else:
            data_values = self.data[outcome].dropna()
            
            if len(data_values) >= 3:
                # Shapiro-Wilk test
                shapiro_stat, shapiro_p = stats.shapiro(data_values)
                
                # D'Agostino's K^2 test
                k2_stat, k2_p = stats.normaltest(data_values)
                
                # Anderson-Darling test
                anderson_result = stats.anderson(data_values, dist='norm')
                
                results.append({
                    'Group': 'All Data',
                    'n': len(data_values),
                    'Shapiro-Wilk_stat': shapiro_stat,
                    'Shapiro-Wilk_p': shapiro_p,
                    'Normal (p>0.05)': shapiro_p > 0.05,
                    'K2_stat': k2_stat,
                    'K2_p': k2_p,
                    'Anderson_stat': anderson_result.statistic
                })
            else:
                results.append({
                    'Group': 'All Data',
                    'n': len(data_values),
                    'Shapiro-Wilk_stat': None,
                    'Shapiro-Wilk_p': None,
                    'Normal (p>0.05)': None,
                    'K2_stat': None,
                    'K2_p': None,
                    'Anderson_stat': None
                })
                
        logger.info("Normality test completed")
        return pd.DataFrame(results)
    
    def test_interaction(self, outcome: str, factor1: str, factor2: str) -> dict:
        """
        Test for interaction effects between two factors.
        
        Args:
            outcome (str): Outcome variable.
            factor1 (str): First factor.
            factor2 (str): Second factor.
            
        Returns:
            dict: Results of interaction test.
        """
        logger.info(f"Testing interaction between {factor1} and {factor2} for {outcome}")
        
        if self.data is None:
            logger.error("No data available for visualization")
            raise ValueError("No data set for visualization. Use set_data() first.")
            
        try:
            # Fit a two-way ANOVA model with interaction
            formula = f"{outcome} ~ C({factor1}) * C({factor2})"
            from statsmodels.formula.api import ols
            import statsmodels.api as sm
            
            model = ols(formula, data=self.data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            
            # Extract interaction term
            interaction_term = f"C({factor1}):C({factor2})"
            
            if interaction_term in anova_table.index:
                F_value = anova_table.loc[interaction_term, 'F']
                p_value = anova_table.loc[interaction_term, 'PR(>F)']
                
                return {
                    'F': F_value,
                    'p': p_value,
                    'significant': p_value < 0.05
                }
            else:
                logger.warning(f"Interaction term {interaction_term} not found in ANOVA table")
                return {
                    'F': 0,
                    'p': 1.0,
                    'significant': False,
                    'error': 'Interaction term not found'
                }
        except Exception as e:
            logger.error(f"Error testing interaction: {str(e)}")
            return {
                'F': 0,
                'p': 1.0,
                'significant': False,
                'error': str(e)
            } 