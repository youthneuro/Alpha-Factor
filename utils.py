import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class AlphaFactorUtils:
    """
    Utility functions for alpha factor research.
    """
    
    @staticmethod
    def validate_data(data: pd.DataFrame) -> bool:
        """
        Validate data quality for alpha factor research.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if data.empty:
            print("Error: Data is empty")
            return False
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
        
        # Check for missing values
        missing_data = data[required_columns].isnull().sum()
        if missing_data.any():
            print(f"Warning: Missing values detected: {missing_data.to_dict()}")
        
        # Check for zero or negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        invalid_prices = (data[price_columns] <= 0).any().any()
        if invalid_prices:
            print("Error: Zero or negative prices detected")
            return False
        
        # Check for reasonable price ranges
        price_range = data['Close'].max() / data['Close'].min()
        if price_range > 1000:
            print(f"Warning: Unusual price range detected: {price_range:.2f}")
        
        return True
    
    @staticmethod
    def calculate_rolling_metrics(data: pd.DataFrame, 
                                window: int = 20) -> pd.DataFrame:
        """
        Calculate rolling statistics for data validation.
        
        Args:
            data: DataFrame with price data
            window: Rolling window size
            
        Returns:
            DataFrame with rolling metrics
        """
        df = data.copy()
        
        # Rolling statistics
        df['Rolling_Mean'] = df['Close'].rolling(window=window).mean()
        df['Rolling_Std'] = df['Close'].rolling(window=window).std()
        df['Rolling_Min'] = df['Close'].rolling(window=window).min()
        df['Rolling_Max'] = df['Close'].rolling(window=window).max()
        
        # Price volatility
        df['Price_Volatility'] = df['Close'].pct_change().rolling(window=window).std()
        
        # Volume statistics
        df['Volume_Mean'] = df['Volume'].rolling(window=window).mean()
        df['Volume_Std'] = df['Volume'].rolling(window=window).std()
        
        return df
    
    @staticmethod
    def detect_outliers(data: pd.DataFrame, 
                       columns: List[str] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> Dict[str, List[int]]:
        """
        Detect outliers in the data.
        
        Args:
            data: DataFrame to analyze
            columns: Columns to check for outliers
            method: Outlier detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with outlier indices for each column
        """
        if columns is None:
            columns = ['Close', 'Volume']
        
        outliers = {}
        
        for column in columns:
            if column not in data.columns:
                continue
            
            series = data[column].dropna()
            
            if method == 'iqr':
                # IQR method
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_indices = series[(series < lower_bound) | (series > upper_bound)].index
            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs((series - series.mean()) / series.std())
                outlier_indices = series[z_scores > threshold].index
            else:
                continue
            
            outliers[column] = list(outlier_indices)
        
        return outliers
    
    @staticmethod
    def optimize_parameters(data: pd.DataFrame,
                          signal_function: callable,
                          param_ranges: Dict[str, List],
                          metric: str = 'sharpe_ratio',
                          n_trials: int = 50) -> Tuple[Dict, float]:
        """
        Optimize parameters for a signal function using grid search.
        
        Args:
            data: DataFrame with price and indicator data
            signal_function: Function that generates signals
            param_ranges: Dictionary of parameter ranges to test
            metric: Performance metric to optimize
            n_trials: Number of trials for optimization
            
        Returns:
            Tuple of (best_parameters, best_score)
        """
        import itertools
        from backtester import Backtester
        
        best_params = None
        best_score = float('-inf')
        
        # Generate parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        # Limit number of combinations
        total_combinations = np.prod([len(vals) for vals in param_values])
        if total_combinations > n_trials:
            # Sample combinations randomly
            combinations = []
            for _ in range(n_trials):
                combination = {}
                for name, values in param_ranges.items():
                    combination[name] = np.random.choice(values)
                combinations.append(combination)
        else:
            # Test all combinations
            combinations = []
            for values in itertools.product(*param_values):
                combination = dict(zip(param_names, values))
                combinations.append(combination)
        
        # Test each combination
        backtester = Backtester()
        
        for i, params in enumerate(combinations):
            try:
                # Generate signals with current parameters
                signals = signal_function(data, **params)
                
                # Run backtest
                data_with_signals = data.copy()
                data_with_signals['Test_Signal'] = signals
                
                results = backtester.run_backtest(data_with_signals, ['Test_Signal'])
                
                if results and 'Test_Signal' in results:
                    result = results['Test_Signal']
                    score = result.get('sharpe_ratio', float('-inf'))
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                
                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{len(combinations)} trials")
                    
            except Exception as e:
                print(f"Error testing parameters {params}: {e}")
                continue
        
        return best_params, best_score
    
    @staticmethod
    def calculate_factor_importance(data: pd.DataFrame,
                                 signal_columns: List[str],
                                 target_column: str = 'Returns',
                                 method: str = 'correlation') -> pd.DataFrame:
        """
        Calculate importance of different factors.
        
        Args:
            data: DataFrame with signals and target
            signal_columns: List of signal column names
            target_column: Target variable column
            method: Method for importance calculation ('correlation' or 'regression')
            
        Returns:
            DataFrame with factor importance scores
        """
        importance_scores = []
        
        for signal in signal_columns:
            if signal in data.columns and target_column in data.columns:
                signal_data = data[signal].dropna()
                target_data = data[target_column].dropna()
                
                # Align data
                common_index = signal_data.index.intersection(target_data.index)
                if len(common_index) > 0:
                    signal_aligned = signal_data.loc[common_index]
                    target_aligned = target_data.loc[common_index]
                    
                    if method == 'correlation':
                        # Correlation-based importance
                        correlation = signal_aligned.corr(target_aligned)
                        importance = abs(correlation) if not pd.isna(correlation) else 0
                    elif method == 'regression':
                        # Regression-based importance
                        try:
                            from sklearn.linear_model import LinearRegression
                            from sklearn.preprocessing import StandardScaler
                            
                            X = signal_aligned.values.reshape(-1, 1)
                            y = target_aligned.values
                            
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            
                            model = LinearRegression()
                            model.fit(X_scaled, y)
                            
                            importance = abs(model.coef_[0])
                        except:
                            importance = 0
                    else:
                        importance = 0
                    
                    importance_scores.append({
                        'Factor': signal,
                        'Importance': importance,
                        'Correlation': signal_aligned.corr(target_aligned) if method == 'correlation' else 0
                    })
        
        return pd.DataFrame(importance_scores).sort_values('Importance', ascending=False)
    
    @staticmethod
    def create_factor_portfolio(data: pd.DataFrame,
                              signal_columns: List[str],
                              weights: List[float] = None) -> pd.Series:
        """
        Create a portfolio of factors with specified weights.
        
        Args:
            data: DataFrame with signals
            signal_columns: List of signal column names
            weights: List of weights for each factor (optional)
            
        Returns:
            Series with combined portfolio signal
        """
        if weights is None:
            # Equal weights
            weights = [1.0 / len(signal_columns)] * len(signal_columns)
        
        if len(weights) != len(signal_columns):
            raise ValueError("Number of weights must match number of signals")
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Combine signals
        portfolio_signal = pd.Series(0, index=data.index)
        
        for signal, weight in zip(signal_columns, weights):
            if signal in data.columns:
                portfolio_signal += weight * data[signal]
        
        return portfolio_signal
    
    @staticmethod
    def calculate_rolling_performance(data: pd.DataFrame,
                                   signal_column: str,
                                   window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling performance metrics for a signal.
        
        Args:
            data: DataFrame with price and signal data
            signal_column: Name of signal column
            window: Rolling window size
            
        Returns:
            DataFrame with rolling performance metrics
        """
        if signal_column not in data.columns:
            return pd.DataFrame()
        
        # Calculate returns
        returns = data['Close'].pct_change()
        
        # Calculate signal returns
        signal_returns = data[signal_column].shift(1) * returns
        
        # Rolling metrics
        rolling_metrics = pd.DataFrame()
        rolling_metrics['Rolling_Return'] = signal_returns.rolling(window=window).sum()
        rolling_metrics['Rolling_Volatility'] = signal_returns.rolling(window=window).std() * np.sqrt(252)
        rolling_metrics['Rolling_Sharpe'] = (rolling_metrics['Rolling_Return'] * 252 / window) / rolling_metrics['Rolling_Volatility']
        
        # Rolling drawdown
        cumulative_returns = (1 + signal_returns).cumprod()
        running_max = cumulative_returns.rolling(window=window).max()
        rolling_metrics['Rolling_Drawdown'] = (cumulative_returns - running_max) / running_max
        
        return rolling_metrics
    
    @staticmethod
    def generate_factor_report(data: pd.DataFrame,
                             signal_columns: List[str]) -> str:
        """
        Generate a comprehensive factor analysis report.
        
        Args:
            data: DataFrame with signals and price data
            signal_columns: List of signal column names
            
        Returns:
            Report text
        """
        report = []
        report.append("=" * 60)
        report.append("FACTOR ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Signal statistics
        report.append("SIGNAL STATISTICS:")
        report.append("-" * 30)
        for signal in signal_columns:
            if signal in data.columns:
                signal_data = data[signal].dropna()
                report.append(f"{signal}:")
                report.append(f"  Mean: {signal_data.mean():.4f}")
                report.append(f"  Std: {signal_data.std():.4f}")
                report.append(f"  Min: {signal_data.min():.4f}")
                report.append(f"  Max: {signal_data.max():.4f}")
                report.append(f"  Non-zero signals: {len(signal_data[signal_data != 0])}")
                report.append("")
        
        # Factor importance
        report.append("FACTOR IMPORTANCE:")
        report.append("-" * 30)
        importance_df = AlphaFactorUtils.calculate_factor_importance(data, signal_columns)
        for _, row in importance_df.iterrows():
            report.append(f"{row['Factor']}: {row['Importance']:.4f}")
        report.append("")
        
        # Correlation analysis
        report.append("FACTOR CORRELATIONS:")
        report.append("-" * 30)
        signal_data = data[signal_columns].dropna()
        if not signal_data.empty:
            correlation_matrix = signal_data.corr()
            for i, signal1 in enumerate(signal_columns):
                for j, signal2 in enumerate(signal_columns):
                    if i < j:  # Avoid duplicates
                        corr = correlation_matrix.loc[signal1, signal2]
                        report.append(f"{signal1} vs {signal2}: {corr:.4f}")
        report.append("")
        
        return "\n".join(report) 