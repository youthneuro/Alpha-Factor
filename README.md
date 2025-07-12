# Alpha Factor Research System

A comprehensive system for discovering and backtesting alpha factors (predictive signals) in stock markets.

## Features

- **Data Collection**: Fetch S&P 500 stock data using yfinance
- **Technical Indicators**: Calculate RSI, MACD, moving averages, Bollinger Bands, and more
- **Alpha Factors**: Define and test various trading signals
- **Backtesting**: Simulate trading strategies with realistic constraints
- **Performance Metrics**: Track returns, Sharpe ratio, max drawdown
- **Factor Combination**: Optimize multiple factors together
- **Interactive Visualizations**: Plotly dashboards and matplotlib charts
- **Parameter Optimization**: Automated parameter tuning for factors
- **Risk Analysis**: VaR, Expected Shortfall, and other risk metrics

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Run the complete system
```bash
python main.py
```

### Option 2: Run single stock example
```bash
python main.py --single
```

## Usage

### Basic Usage

1. **Data Collection**:
   ```python
   from data_collector import DataCollector
   collector = DataCollector()
   data = collector.get_sp500_data()
   ```

2. **Technical Indicators**:
   ```python
   from indicators import TechnicalIndicators
   indicators = TechnicalIndicators()
   data_with_indicators = indicators.add_all_indicators(data)
   ```

3. **Alpha Factors**:
   ```python
   from factors import AlphaFactors
   factors = AlphaFactors()
   signals = factors.generate_signals(data_with_indicators)
   ```

4. **Backtesting**:
   ```python
   from backtester import Backtester
   backtester = Backtester()
   results = backtester.run_backtest(signals, data_with_indicators)
   ```

5. **Performance Analysis**:
   ```python
   from performance import PerformanceAnalyzer
   analyzer = PerformanceAnalyzer()
   metrics = analyzer.calculate_metrics(results)
   ```

### Advanced Usage

**Parameter Optimization**:
```python
from utils import AlphaFactorUtils
best_params, best_score = AlphaFactorUtils.optimize_parameters(
    data, signal_function, param_ranges
)
```

**Factor Portfolio Creation**:
```python
portfolio_signal = AlphaFactorUtils.create_factor_portfolio(
    data, signal_columns, weights
)
```

**Risk Analysis**:
```python
risk_metrics = backtester.calculate_risk_metrics(returns)
```

## Project Structure

```
alpha_factor/
├── data_collector.py      # Data fetching from yfinance
├── indicators.py          # Technical indicator calculations
├── factors.py            # Alpha factor definitions
├── backtester.py         # Backtesting framework
├── performance.py        # Performance metrics and visualization
├── utils.py             # Utility functions and optimization
├── main.py              # Main execution script
├── examples/            # Example notebooks and scripts
│   └── alpha_factor_example.ipynb  # Interactive Jupyter notebook
├── results/             # Output and analysis results
└── requirements.txt     # Python dependencies
```

## Technical Indicators Included

- **Moving Averages**: SMA, EMA (5, 10, 20, 50, 100, 200 periods)
- **Momentum**: RSI, MACD, Stochastic, Williams %R, ROC, MFI, CCI
- **Volatility**: Bollinger Bands, ATR, Keltner Channel, Donchian Channel
- **Volume**: OBV, VPT, ADL, Volume Ratio
- **Trend**: ADX, Parabolic SAR, Ichimoku Cloud

## Alpha Factors Implemented

1. **Moving Average Crossover**: Buy when fast MA crosses above slow MA
2. **RSI Signals**: Buy when RSI crosses above oversold level
3. **MACD Signals**: Buy when MACD line crosses above signal line
4. **Bollinger Band Signals**: Buy when price touches lower band
5. **Volume Breakout**: Buy when volume is high with price breakout
6. **Mean Reversion**: Buy when price is significantly below moving average
7. **Momentum Signals**: Buy when momentum is strong and positive
8. **Stochastic Signals**: Buy when stochastic crosses above oversold level
9. **ATR Breakout**: Buy when price breaks out of ATR range
10. **Combined Momentum-Volume**: Buy when both momentum and volume are strong

## Performance Metrics

- **Total Return**: Overall strategy performance
- **Annual Return**: Annualized return rate
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Calmar Ratio**: Annual return / Maximum drawdown
- **Volatility**: Standard deviation of returns
- **VaR (95% & 99%)**: Value at Risk
- **Expected Shortfall**: Conditional VaR

## Backtesting Features

- **Realistic Constraints**: Transaction costs, slippage, position sizing
- **Multi-Stock Support**: Test strategies across multiple symbols
- **Risk Management**: Maximum position size limits
- **Performance Tracking**: Detailed trade and portfolio tracking
- **Aggregation**: Average results across multiple stocks

## Visualization Features

- **Equity Curves**: Portfolio value over time
- **Drawdown Analysis**: Peak-to-trough decline visualization
- **Returns Distribution**: Histogram of strategy returns
- **Metrics Comparison**: Bar charts comparing different strategies
- **Interactive Dashboard**: Plotly-based interactive visualizations
- **Correlation Heatmaps**: Factor correlation analysis

## Output Files

When you run the system, it generates:

- `equity_curves.png`: Portfolio performance over time
- `drawdown.png`: Drawdown analysis
- `returns_distribution.png`: Returns distribution histograms
- `metrics_comparison.png`: Strategy comparison charts
- `interactive_dashboard.html`: Interactive Plotly dashboard
- `performance_report.txt`: Detailed performance report
- `performance_metrics.csv`: Performance metrics in CSV format
- `factor_correlations.csv`: Factor correlation matrix

## Examples

### Single Stock Analysis
```bash
python main.py --single
```

### Multi-Stock Analysis
```bash
python main.py
```

### Interactive Notebook
Open `examples/alpha_factor_example.ipynb` in Jupyter for step-by-step exploration.

## Contributing

1. **Add New Factors**: Implement new alpha factors in `factors.py`
2. **Add New Indicators**: Extend technical indicators in `indicators.py`
3. **Custom Backtesting**: Modify backtesting parameters in `backtester.py`
4. **Performance Analysis**: Add new metrics in `performance.py`

## Dependencies

- **Data**: yfinance (Yahoo Finance API)
- **Analysis**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn (for optimization)
- **Technical Analysis**: ta (Technical Analysis library)

## License

This project is for educational and research purposes. Please ensure compliance with data provider terms of service. 
