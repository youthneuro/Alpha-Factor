#!/usr/bin/env python3
"""
Alpha Factor Research - Main Execution Script

This script demonstrates the complete workflow for alpha factor research:
1. Data collection from yfinance
2. Technical indicator calculation
3. Alpha factor signal generation
4. Backtesting
5. Performance analysis and visualization
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

# Import our modules
from data_collector import DataCollector
from indicators import TechnicalIndicators
from factors import AlphaFactors
from backtester import Backtester
from performance import PerformanceAnalyzer

def main():
    """
    Main execution function for alpha factor research.
    """
    print("=" * 60)
    print("ALPHA FACTOR RESEARCH SYSTEM")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Data Collection
    print("STEP 1: DATA COLLECTION")
    print("-" * 30)
    
    collector = DataCollector()
    
    # Fetch S&P 500 data (limit to 10 stocks for demonstration)
    print("Fetching S&P 500 stock data...")
    data_dict = collector.get_sp500_data(
        start_date='2020-01-01',
        end_date='2024-01-01',
        max_stocks=10
    )
    
    if not data_dict:
        print("Error: No data collected. Exiting.")
        return
    
    print(f"Successfully collected data for {len(data_dict)} stocks")
    print()
    
    # Step 2: Technical Indicators
    print("STEP 2: TECHNICAL INDICATORS")
    print("-" * 30)
    
    indicators = TechnicalIndicators()
    
    # Add indicators to each stock's data
    data_with_indicators = {}
    for symbol, data in data_dict.items():
        print(f"Adding indicators for {symbol}...")
        data_with_indicators[symbol] = indicators.add_all_indicators(data)
    
    print("All technical indicators added successfully!")
    print()
    
    # Step 3: Alpha Factors
    print("STEP 3: ALPHA FACTORS")
    print("-" * 30)
    
    factors = AlphaFactors()
    
    # Generate signals for each stock
    data_with_signals = {}
    for symbol, data in data_with_indicators.items():
        print(f"Generating signals for {symbol}...")
        data_with_signals[symbol] = factors.generate_all_signals(data)
    
    print("All alpha factor signals generated successfully!")
    print()
    
    # Combine all data for backtesting
    print("Combining data for backtesting...")
    combined_data = collector.get_combined_data(data_with_signals)
    
    if combined_data.empty:
        print("Error: No combined data available. Exiting.")
        return
    
    print(f"Combined data shape: {combined_data.shape}")
    print()
    
    # Step 4: Backtesting
    print("STEP 4: BACKTESTING")
    print("-" * 30)
    
    # Initialize backtester with realistic parameters
    backtester = Backtester(
        initial_capital=100000,  # $100k starting capital
        transaction_cost=0.001,  # 0.1% transaction cost
        slippage=0.0005,        # 0.05% slippage
        max_position_size=0.1    # Max 10% of capital per position
    )
    
    # Run backtests for all signals
    print("Running backtests...")
    backtest_results = backtester.run_backtest(combined_data)
    
    if not backtest_results:
        print("Error: No backtest results. Exiting.")
        return
    
    print("Backtesting completed successfully!")
    print()
    
    # Step 5: Performance Analysis
    print("STEP 5: PERFORMANCE ANALYSIS")
    print("-" * 30)
    
    analyzer = PerformanceAnalyzer()
    
    # Calculate performance metrics
    print("Calculating performance metrics...")
    metrics_df = analyzer.calculate_metrics(backtest_results)
    
    # Display results
    print("\nPERFORMANCE METRICS:")
    print("=" * 50)
    print(metrics_df.to_string(index=False))
    print()
    
    # Get best strategy
    best_strategy, best_result = backtester.get_best_strategy()
    if best_strategy:
        print(f"BEST STRATEGY: {best_strategy}")
        if 'avg_sharpe_ratio' in best_result:
            print(f"Average Sharpe Ratio: {best_result['avg_sharpe_ratio']:.3f}")
            print(f"Average Total Return: {best_result['avg_total_return']*100:.2f}%")
            print(f"Average Max Drawdown: {best_result['avg_max_drawdown']*100:.2f}%")
        else:
            print(f"Sharpe Ratio: {best_result['sharpe_ratio']:.3f}")
            print(f"Total Return: {best_result['total_return']*100:.2f}%")
            print(f"Max Drawdown: {best_result['max_drawdown']*100:.2f}%")
        print()
    
    # Generate strategy ranking
    print("STRATEGY RANKING (by Sharpe Ratio):")
    print("=" * 50)
    ranking_df = backtester.get_strategy_ranking()
    if not ranking_df.empty:
        print(ranking_df.to_string(index=False))
        print()
    
    # Step 6: Visualization and Reporting
    print("STEP 6: VISUALIZATION AND REPORTING")
    print("-" * 30)
    
    # Generate plots
    print("Generating performance plots...")
    
    # Equity curves
    fig_equity = analyzer.plot_equity_curves(backtest_results)
    fig_equity.savefig(f"{output_dir}/equity_curves.png", dpi=300, bbox_inches='tight')
    print("✓ Equity curves saved")
    
    # Drawdown
    fig_drawdown = analyzer.plot_drawdown(backtest_results)
    fig_drawdown.savefig(f"{output_dir}/drawdown.png", dpi=300, bbox_inches='tight')
    print("✓ Drawdown plot saved")
    
    # Returns distribution
    fig_dist = analyzer.plot_returns_distribution(backtest_results)
    fig_dist.savefig(f"{output_dir}/returns_distribution.png", dpi=300, bbox_inches='tight')
    print("✓ Returns distribution saved")
    
    # Metrics comparison
    fig_metrics = analyzer.plot_metrics_comparison(backtest_results)
    fig_metrics.savefig(f"{output_dir}/metrics_comparison.png", dpi=300, bbox_inches='tight')
    print("✓ Metrics comparison saved")
    
    # Interactive dashboard
    fig_dashboard = analyzer.create_interactive_dashboard(backtest_results)
    fig_dashboard.write_html(f"{output_dir}/interactive_dashboard.html")
    print("✓ Interactive dashboard saved")
    
    # Generate performance report
    print("Generating performance report...")
    report = analyzer.generate_performance_report(backtest_results, 
                                               f"{output_dir}/performance_report.txt")
    print("✓ Performance report saved")
    
    # Save metrics to CSV
    metrics_df.to_csv(f"{output_dir}/performance_metrics.csv", index=False)
    print("✓ Performance metrics saved to CSV")
    
    # Step 7: Factor Analysis
    print("STEP 7: FACTOR ANALYSIS")
    print("-" * 30)
    
    # Get signal summary
    print("Signal summary:")
    signal_summary = factors.get_signal_summary(combined_data)
    for signal, summary in signal_summary.items():
        print(f"  {signal}:")
        print(f"    Total signals: {summary['total_signals']}")
        print(f"    Buy signals: {summary['buy_signals']}")
        print(f"    Sell signals: {summary['sell_signals']}")
        print(f"    Signal strength mean: {summary['signal_strength_mean']:.3f}")
        print()
    
    # Calculate factor correlations
    print("Calculating factor correlations...")
    correlation_matrix = factors.get_factor_correlation(combined_data)
    correlation_matrix.to_csv(f"{output_dir}/factor_correlations.csv")
    print("✓ Factor correlations saved")
    
    # Summary
    print("=" * 60)
    print("ALPHA FACTOR RESEARCH COMPLETED")
    print("=" * 60)
    print(f"Results saved to: {output_dir}/")
    print(f"Files generated:")
    print(f"  - equity_curves.png")
    print(f"  - drawdown.png")
    print(f"  - returns_distribution.png")
    print(f"  - metrics_comparison.png")
    print(f"  - interactive_dashboard.html")
    print(f"  - performance_report.txt")
    print(f"  - performance_metrics.csv")
    print(f"  - factor_correlations.csv")
    print()
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def run_single_stock_example():
    """
    Run a simplified example with a single stock for demonstration.
    """
    print("=" * 60)
    print("SINGLE STOCK EXAMPLE")
    print("=" * 60)
    
    # Data collection
    collector = DataCollector()
    data = collector.get_stock_data('AAPL', '2020-01-01', '2024-01-01')
    
    if data is None:
        print("Error: Could not fetch AAPL data")
        return
    
    # Technical indicators
    indicators = TechnicalIndicators()
    data_with_indicators = indicators.add_all_indicators(data)
    
    # Alpha factors
    factors = AlphaFactors()
    data_with_signals = factors.generate_all_signals(data_with_indicators)
    
    # Backtesting
    backtester = Backtester()
    results = backtester.run_backtest(data_with_signals)
    
    # Performance analysis
    analyzer = PerformanceAnalyzer()
    metrics_df = analyzer.calculate_metrics(results)
    
    print("\nSINGLE STOCK RESULTS (AAPL):")
    print("=" * 40)
    print(metrics_df.to_string(index=False))
    
    # Save results
    os.makedirs("results", exist_ok=True)
    metrics_df.to_csv("results/single_stock_results.csv", index=False)
    
    # Generate plots
    fig_equity = analyzer.plot_equity_curves(results)
    fig_equity.savefig("results/single_stock_equity.png", dpi=300, bbox_inches='tight')
    
    print("\nResults saved to results/")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        run_single_stock_example()
    else:
        main() 