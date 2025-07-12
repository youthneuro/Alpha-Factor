import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class PerformanceAnalyzer:
    """
    Analyzes and visualizes backtesting performance results.
    """
    
    def __init__(self):
        self.results = {}
        self.plots = {}
    
    def calculate_metrics(self, results: Dict) -> pd.DataFrame:
        """
        Calculate comprehensive performance metrics for all strategies.
        
        Args:
            results: Dictionary with backtest results
            
        Returns:
            DataFrame with performance metrics
        """
        metrics = []
        
        for strategy, result in results.items():
            if 'avg_sharpe_ratio' in result:
                # Multi-symbol results
                metrics.append({
                    'Strategy': strategy,
                    'Total Return (%)': result['avg_total_return'] * 100,
                    'Annual Return (%)': result['avg_annual_return'] * 100,
                    'Sharpe Ratio': result['avg_sharpe_ratio'],
                    'Max Drawdown (%)': result['avg_max_drawdown'] * 100,
                    'Win Rate (%)': result['avg_win_rate'] * 100,
                    'Profit Factor': result['avg_profit_factor'],
                    'Calmar Ratio': result['avg_calmar_ratio'],
                    'Num Symbols': result['num_symbols']
                })
            else:
                # Single symbol results
                metrics.append({
                    'Strategy': strategy,
                    'Total Return (%)': result['total_return'] * 100,
                    'Annual Return (%)': result['annual_return'] * 100,
                    'Sharpe Ratio': result['sharpe_ratio'],
                    'Max Drawdown (%)': result['max_drawdown'] * 100,
                    'Win Rate (%)': result['win_rate'] * 100,
                    'Profit Factor': result['profit_factor'],
                    'Calmar Ratio': result['calmar_ratio'],
                    'Num Trades': result['num_trades']
                })
        
        return pd.DataFrame(metrics)
    
    def plot_equity_curves(self, results: Dict, 
                          strategies: Optional[List[str]] = None,
                          figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Plot equity curves for different strategies.
        
        Args:
            results: Dictionary with backtest results
            strategies: List of strategies to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if strategies is None:
            strategies = list(results.keys())
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for strategy in strategies:
            if strategy in results:
                result = results[strategy]
                
                if 'portfolio_values' in result:
                    # Single symbol result
                    portfolio_values = result['portfolio_values']
                    normalized_values = portfolio_values / portfolio_values.iloc[0]
                    ax.plot(normalized_values.index, normalized_values.values, 
                           label=strategy, linewidth=2)
                elif 'symbol_results' in result:
                    # Multi-symbol result - plot average
                    symbol_results = result['symbol_results']
                    all_values = []
                    
                    for symbol_result in symbol_results.values():
                        if 'portfolio_values' in symbol_result:
                            values = symbol_result['portfolio_values']
                            normalized = values / values.iloc[0]
                            all_values.append(normalized.values)
                    
                    if all_values:
                        avg_values = np.mean(all_values, axis=0)
                        ax.plot(portfolio_values.index, avg_values, 
                               label=f"{strategy} (Avg)", linewidth=2)
        
        ax.set_title('Equity Curves Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value (Normalized)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_drawdown(self, results: Dict, 
                     strategies: List[str] = None,
                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot drawdown curves for different strategies.
        
        Args:
            results: Dictionary with backtest results
            strategies: List of strategies to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if strategies is None:
            strategies = list(results.keys())
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for strategy in strategies:
            if strategy in results:
                result = results[strategy]
                
                if 'returns' in result:
                    # Single symbol result
                    returns = result['returns']
                    cumulative_returns = (1 + returns).cumprod()
                    running_max = cumulative_returns.expanding().max()
                    drawdown = (cumulative_returns - running_max) / running_max
                    ax.plot(drawdown.index, drawdown.values * 100, 
                           label=strategy, linewidth=2)
                elif 'symbol_results' in result:
                    # Multi-symbol result - plot average
                    symbol_results = result['symbol_results']
                    all_drawdowns = []
                    
                    for symbol_result in symbol_results.values():
                        if 'returns' in symbol_result:
                            returns = symbol_result['returns']
                            cumulative_returns = (1 + returns).cumprod()
                            running_max = cumulative_returns.expanding().max()
                            drawdown = (cumulative_returns - running_max) / running_max
                            all_drawdowns.append(drawdown.values * 100)
                    
                    if all_drawdowns:
                        avg_drawdown = np.mean(all_drawdowns, axis=0)
                        # Use the first symbol's index for plotting
                        first_symbol_result = list(symbol_results.values())[0]
                        if 'returns' in first_symbol_result:
                            first_returns = first_symbol_result['returns']
                            first_cumulative = (1 + first_returns).cumprod()
                            ax.plot(first_cumulative.index, avg_drawdown, 
                                   label=f"{strategy} (Avg)", linewidth=2)
        
        ax.set_title('Drawdown Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        return fig
    
    def plot_returns_distribution(self, results: Dict, 
                                strategies: List[str] = None,
                                figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot returns distribution for different strategies.
        
        Args:
            results: Dictionary with backtest results
            strategies: List of strategies to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if strategies is None:
            strategies = list(results.keys())
        
        n_strategies = len(strategies)
        n_cols = min(3, n_strategies)
        n_rows = (n_strategies + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_strategies == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, strategy in enumerate(strategies):
            if strategy in results:
                result = results[strategy]
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                
                if 'returns' in result:
                    # Single symbol result
                    returns = result['returns']
                    ax.hist(returns, bins=50, alpha=0.7, edgecolor='black')
                    ax.axvline(returns.mean(), color='red', linestyle='--', 
                              label=f'Mean: {returns.mean():.4f}')
                    ax.axvline(returns.median(), color='green', linestyle='--', 
                              label=f'Median: {returns.median():.4f}')
                    ax.set_title(f'{strategy} Returns Distribution', fontweight='bold')
                    ax.set_xlabel('Returns')
                    ax.set_ylabel('Frequency')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                elif 'symbol_results' in result:
                    # Multi-symbol result
                    symbol_results = result['symbol_results']
                    all_returns = []
                    
                    for symbol_result in symbol_results.values():
                        if 'returns' in symbol_result:
                            all_returns.extend(symbol_result['returns'].dropna().values)
                    
                    if all_returns:
                        ax.hist(all_returns, bins=50, alpha=0.7, edgecolor='black')
                        ax.axvline(np.mean(all_returns), color='red', linestyle='--', 
                                  label=f'Mean: {np.mean(all_returns):.4f}')
                        ax.axvline(np.median(all_returns), color='green', linestyle='--', 
                                  label=f'Median: {np.median(all_returns):.4f}')
                        ax.set_title(f'{strategy} Returns Distribution (All Symbols)', fontweight='bold')
                        ax.set_xlabel('Returns')
                        ax.set_ylabel('Frequency')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_strategies, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_comparison(self, results: Dict, 
                              metrics: List[str] = None,
                              figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Create bar plots comparing different metrics across strategies.
        
        Args:
            results: Dictionary with backtest results
            metrics: List of metrics to compare
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)']
        
        df_metrics = self.calculate_metrics(results)
        
        n_metrics = len(metrics)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, metric in enumerate(metrics):
            if metric in df_metrics.columns:
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                
                # Sort by metric value
                df_sorted = df_metrics.sort_values(metric, ascending=False)
                
                bars = ax.bar(df_sorted['Strategy'], df_sorted[metric], 
                             color='skyblue', edgecolor='navy', alpha=0.7)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom')
                
                ax.set_title(f'{metric} Comparison', fontweight='bold')
                ax.set_xlabel('Strategy')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_dashboard(self, results: Dict) -> go.Figure:
        """
        Create an interactive Plotly dashboard with multiple charts.
        
        Args:
            results: Dictionary with backtest results
            
        Returns:
            Plotly figure with subplots
        """
        strategies = list(results.keys())
        n_strategies = len(strategies)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Equity Curves', 'Drawdown', 'Returns Distribution', 'Metrics Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Equity curves
        for strategy in strategies:
            result = results[strategy]
            if 'portfolio_values' in result:
                portfolio_values = result['portfolio_values']
                normalized_values = portfolio_values / portfolio_values.iloc[0]
                fig.add_trace(
                    go.Scatter(x=normalized_values.index, y=normalized_values.values,
                              mode='lines', name=strategy),
                    row=1, col=1
                )
        
        # Drawdown
        for strategy in strategies:
            result = results[strategy]
            if 'returns' in result:
                returns = result['returns']
                cumulative_returns = (1 + returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                fig.add_trace(
                    go.Scatter(x=drawdown.index, y=drawdown.values * 100,
                              mode='lines', name=strategy, showlegend=False),
                    row=1, col=2
                )
        
        # Returns distribution
        for strategy in strategies:
            result = results[strategy]
            if 'returns' in result:
                returns = result['returns'].dropna()
                fig.add_trace(
                    go.Histogram(x=returns, name=strategy, opacity=0.7),
                    row=2, col=1
                )
        
        # Metrics comparison
        df_metrics = self.calculate_metrics(results)
        fig.add_trace(
            go.Bar(x=df_metrics['Strategy'], y=df_metrics['Sharpe Ratio'],
                   name='Sharpe Ratio'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Alpha Factor Performance Dashboard",
            showlegend=True,
            height=800
        )
        
        return fig
    
    def generate_performance_report(self, results: Dict, 
                                 output_file: str = None) -> str:
        """
        Generate a comprehensive performance report.
        
        Args:
            results: Dictionary with backtest results
            output_file: File path to save report
            
        Returns:
            Report text
        """
        df_metrics = self.calculate_metrics(results)
        
        report = []
        report.append("=" * 60)
        report.append("ALPHA FACTOR PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS:")
        report.append("-" * 30)
        for _, row in df_metrics.iterrows():
            report.append(f"Strategy: {row['Strategy']}")
            report.append(f"  Total Return: {row['Total Return (%)']:.2f}%")
            report.append(f"  Annual Return: {row['Annual Return (%)']:.2f}%")
            report.append(f"  Sharpe Ratio: {row['Sharpe Ratio']:.3f}")
            report.append(f"  Max Drawdown: {row['Max Drawdown (%)']:.2f}%")
            report.append(f"  Win Rate: {row['Win Rate (%)']:.2f}%")
            if 'Profit Factor' in row:
                report.append(f"  Profit Factor: {row['Profit Factor']:.3f}")
            if 'Calmar Ratio' in row:
                report.append(f"  Calmar Ratio: {row['Calmar Ratio']:.3f}")
            report.append("")
        
        # Best performing strategy
        best_strategy = df_metrics.loc[df_metrics['Sharpe Ratio'].idxmax()]
        report.append("BEST PERFORMING STRATEGY:")
        report.append("-" * 30)
        report.append(f"Strategy: {best_strategy['Strategy']}")
        report.append(f"Sharpe Ratio: {best_strategy['Sharpe Ratio']:.3f}")
        report.append(f"Total Return: {best_strategy['Total Return (%)']:.2f}%")
        report.append("")
        
        # Risk analysis
        report.append("RISK ANALYSIS:")
        report.append("-" * 30)
        for _, row in df_metrics.iterrows():
            report.append(f"{row['Strategy']}:")
            report.append(f"  Volatility: {row['Sharpe Ratio']:.3f}")
            report.append(f"  Max Drawdown: {row['Max Drawdown (%)']:.2f}%")
            report.append(f"  Win Rate: {row['Win Rate (%)']:.2f}%")
            report.append("")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def save_plots(self, results: Dict, output_dir: str = "results"):
        """
        Save all plots to files.
        
        Args:
            results: Dictionary with backtest results
            output_dir: Output directory for plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Equity curves
        fig_equity = self.plot_equity_curves(results)
        fig_equity.savefig(f"{output_dir}/equity_curves.png", dpi=300, bbox_inches='tight')
        plt.close(fig_equity)
        
        # Drawdown
        fig_drawdown = self.plot_drawdown(results)
        fig_drawdown.savefig(f"{output_dir}/drawdown.png", dpi=300, bbox_inches='tight')
        plt.close(fig_drawdown)
        
        # Returns distribution
        fig_dist = self.plot_returns_distribution(results)
        fig_dist.savefig(f"{output_dir}/returns_distribution.png", dpi=300, bbox_inches='tight')
        plt.close(fig_dist)
        
        # Metrics comparison
        fig_metrics = self.plot_metrics_comparison(results)
        fig_metrics.savefig(f"{output_dir}/metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig_metrics)
        
        # Interactive dashboard
        fig_dashboard = self.create_interactive_dashboard(results)
        fig_dashboard.write_html(f"{output_dir}/interactive_dashboard.html")
        
        print(f"All plots saved to {output_dir}/") 