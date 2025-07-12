import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class Backtester:
    """
    Comprehensive backtesting framework for alpha factor strategies.
    """
    
    def __init__(self, initial_capital: float = 100000, 
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005,
                 max_position_size: float = 0.1):
        """
        Initialize backtester with parameters.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as percentage
            slippage: Slippage cost as percentage
            max_position_size: Maximum position size as percentage of capital
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.results = {}
    
    def calculate_position_size(self, signal_strength: float, 
                              current_price: float, 
                              available_capital: float) -> float:
        """
        Calculate position size based on signal strength and risk management.
        
        Args:
            signal_strength: Signal strength (-1 to 1)
            current_price: Current stock price
            available_capital: Available capital
            
        Returns:
            Position size in number of shares
        """
        # Base position size as percentage of capital
        base_position_pct = abs(signal_strength) * self.max_position_size
        
        # Calculate position value
        position_value = available_capital * base_position_pct
        
        # Calculate number of shares
        shares = position_value / current_price
        
        # Apply direction
        if signal_strength < 0:
            shares = -shares
            
        return shares
    
    def calculate_returns(self, positions: pd.Series, 
                         prices: pd.Series, 
                         signals: pd.Series) -> pd.Series:
        """
        Calculate returns for each period.
        
        Args:
            positions: Series with position sizes
            prices: Series with prices
            signals: Series with signals
            
        Returns:
            Series with returns
        """
        # Calculate price returns
        price_returns = prices.pct_change()
        
        # Calculate position returns
        position_returns = positions.shift(1) * price_returns
        
        # Calculate transaction costs
        position_changes = positions.diff().abs()
        transaction_costs = position_changes * prices * self.transaction_cost
        
        # Calculate slippage costs
        slippage_costs = position_changes * prices * self.slippage
        
        # Net returns
        net_returns = position_returns - transaction_costs - slippage_costs
        
        return net_returns
    
    def run_single_backtest(self, data: pd.DataFrame, 
                           signal_column: str,
                           symbol: Optional[str] = None) -> Dict:
        """
        Run backtest for a single signal.
        
        Args:
            data: DataFrame with price data and signals
            signal_column: Column name for the signal
            symbol: Stock symbol (for multi-stock data)
            
        Returns:
            Dictionary with backtest results
        """
        if symbol:
            df = data.loc[symbol].copy()
        else:
            df = data.copy()
        
        # Initialize tracking variables
        capital = self.initial_capital
        positions = pd.Series(0, index=df.index)
        portfolio_values = pd.Series(capital, index=df.index)
        trades = []
        
        # Run backtest
        for i in range(1, len(df)):
            current_price = df['Close'].iloc[i]
            current_signal = df[signal_column].iloc[i]
            
            # Calculate new position
            new_position = self.calculate_position_size(
                current_signal, current_price, capital
            )
            
            # Update position
            old_position = positions.iloc[i-1]
            position_change = new_position - old_position
            
            # Calculate transaction costs
            transaction_cost = abs(position_change) * current_price * self.transaction_cost
            slippage_cost = abs(position_change) * current_price * self.slippage
            
            # Update capital
            capital = capital - (position_change * current_price) - transaction_cost - slippage_cost
            
            # Update position
            positions.iloc[i] = new_position
            
            # Calculate portfolio value
            portfolio_values.iloc[i] = capital + (new_position * current_price)
            
            # Record trade if position changed
            if position_change != 0:
                trades.append({
                    'date': df.index[i],
                    'price': current_price,
                    'position_change': position_change,
                    'signal': current_signal,
                    'transaction_cost': transaction_cost,
                    'slippage_cost': slippage_cost
                })
        
        # Calculate returns
        returns = portfolio_values.pct_change()
        
        # Calculate metrics
        total_return = (portfolio_values.iloc[-1] - self.initial_capital) / self.initial_capital
        annual_return = total_return * (252 / len(df))
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        positive_returns = returns[returns > 0]
        win_rate = len(positive_returns) / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0
        
        # Calculate profit factor
        gross_profit = positive_returns.sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        results = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'num_trades': len(trades),
            'final_capital': portfolio_values.iloc[-1],
            'portfolio_values': portfolio_values,
            'returns': returns,
            'positions': positions,
            'trades': trades
        }
        
        return results
    
    def run_backtest(self, data: pd.DataFrame, 
                    signal_columns: Optional[List[str]] = None) -> Dict:
        """
        Run backtest for multiple signals.
        
        Args:
            data: DataFrame with price data and signals
            signal_columns: List of signal column names to test
            
        Returns:
            Dictionary with backtest results for each signal
        """
        if signal_columns is None:
            signal_columns = [col for col in data.columns if col.endswith('_Signal')]
        
        print(f"Running backtests for {len(signal_columns)} signals...")
        
        results = {}
        
        # Check if data has multi-level index (multiple stocks)
        if isinstance(data.index, pd.MultiIndex):
            symbols = data.index.get_level_values(0).unique()
            
            for signal in signal_columns:
                print(f"Testing {signal}...")
                signal_results = {}
                
                for symbol in symbols:
                    try:
                        symbol_data = data.loc[symbol]
                        symbol_result = self.run_single_backtest(symbol_data, signal, symbol)
                        signal_results[symbol] = symbol_result
                    except Exception as e:
                        print(f"Error testing {signal} for {symbol}: {e}")
                        continue
                
                # Aggregate results across symbols
                if signal_results:
                    results[signal] = self.aggregate_results(signal_results)
                else:
                    print(f"No valid results for {signal}")
        
        else:
            # Single stock backtest
            for signal in signal_columns:
                print(f"Testing {signal}...")
                try:
                    results[signal] = self.run_single_backtest(data, signal)
                except Exception as e:
                    print(f"Error testing {signal}: {e}")
                    continue
        
        self.results = results
        print("Backtesting completed!")
        return results
    
    def aggregate_results(self, symbol_results: Dict) -> Dict:
        """
        Aggregate results across multiple symbols.
        
        Args:
            symbol_results: Dictionary with results for each symbol
            
        Returns:
            Aggregated results
        """
        # Calculate average metrics
        total_returns = [result['total_return'] for result in symbol_results.values()]
        annual_returns = [result['annual_return'] for result in symbol_results.values()]
        sharpe_ratios = [result['sharpe_ratio'] for result in symbol_results.values()]
        max_drawdowns = [result['max_drawdown'] for result in symbol_results.values()]
        win_rates = [result['win_rate'] for result in symbol_results.values()]
        profit_factors = [result['profit_factor'] for result in symbol_results.values()]
        calmar_ratios = [result['calmar_ratio'] for result in symbol_results.values()]
        
        # Filter out infinite values
        profit_factors = [pf for pf in profit_factors if pf != float('inf')]
        calmar_ratios = [cr for cr in calmar_ratios if cr != float('inf')]
        
        aggregated = {
            'avg_total_return': np.mean(total_returns),
            'avg_annual_return': np.mean(annual_returns),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'avg_win_rate': np.mean(win_rates),
            'avg_profit_factor': np.mean(profit_factors) if profit_factors else 0,
            'avg_calmar_ratio': np.mean(calmar_ratios) if calmar_ratios else 0,
            'num_symbols': len(symbol_results),
            'symbol_results': symbol_results
        }
        
        return aggregated
    
    def get_best_strategy(self, results: Optional[Dict] = None) -> Tuple[Optional[str], Dict]:
        """
        Find the best performing strategy based on Sharpe ratio.
        
        Args:
            results: Backtest results (uses self.results if None)
            
        Returns:
            Tuple of (strategy_name, results)
        """
        if results is None:
            results = self.results
        
        if not results:
            return None, {}
        
        # Find strategy with highest Sharpe ratio
        best_strategy = max(results.keys(), 
                          key=lambda x: results[x].get('avg_sharpe_ratio', 
                                                     results[x].get('sharpe_ratio', -999)))
        
        return best_strategy, results[best_strategy]
    
    def get_strategy_ranking(self, results: Optional[Dict] = None, 
                           metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """
        Rank strategies by performance metric.
        
        Args:
            results: Backtest results
            metric: Metric to rank by
            
        Returns:
            DataFrame with strategy rankings
        """
        if results is None:
            results = self.results
        
        if not results:
            return pd.DataFrame()
        
        rankings = []
        
        for strategy, result in results.items():
            if 'avg_' + metric in result:
                # Multi-symbol results
                rankings.append({
                    'strategy': strategy,
                    metric: result['avg_' + metric],
                    'total_return': result['avg_total_return'],
                    'max_drawdown': result['avg_max_drawdown'],
                    'win_rate': result['avg_win_rate'],
                    'num_symbols': result['num_symbols']
                })
            elif metric in result:
                # Single symbol results
                rankings.append({
                    'strategy': strategy,
                    metric: result[metric],
                    'total_return': result['total_return'],
                    'max_drawdown': result['max_drawdown'],
                    'win_rate': result['win_rate'],
                    'num_trades': result['num_trades']
                })
        
        if rankings:
            df = pd.DataFrame(rankings)
            return df.sort_values(metric, ascending=False)
        else:
            return pd.DataFrame()
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        """
        Calculate additional risk metrics.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with risk metrics
        """
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Expected Shortfall (Conditional VaR)
        es_95 = returns[returns <= var_95].mean()
        es_99 = returns[returns <= var_99].mean()
        
        # Skewness and Kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Maximum consecutive losses
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Find longest drawdown period
        drawdown_periods = (drawdown < 0).astype(int)
        consecutive_losses = drawdown_periods.groupby(
            (drawdown_periods != drawdown_periods.shift()).cumsum()
        ).sum().max()
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'es_95': es_95,
            'es_99': es_99,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'max_consecutive_losses': consecutive_losses
        } 