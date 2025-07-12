import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class AlphaFactors:
    """
    Defines and generates alpha factors (trading signals) based on technical indicators.
    """
    
    def __init__(self):
        self.factors = {}
        self.factor_descriptions = {}
    
    def moving_average_crossover(self, data: pd.DataFrame, fast_ma: str = 'SMA_50', 
                               slow_ma: str = 'SMA_200') -> pd.Series:
        """
        Moving Average Crossover signal.
        Buy when fast MA crosses above slow MA, sell when it crosses below.
        
        Args:
            data: DataFrame with technical indicators
            fast_ma: Fast moving average column name
            slow_ma: Slow moving average column name
            
        Returns:
            Series with signals (1 for buy, -1 for sell, 0 for hold)
        """
        df = data.copy()
        
        # Calculate crossover signals
        df['MA_Crossover'] = 0
        df.loc[df[fast_ma] > df[slow_ma], 'MA_Crossover'] = 1
        df.loc[df[fast_ma] < df[slow_ma], 'MA_Crossover'] = -1
        
        # Generate signals on crossover
        df['MA_Signal'] = df['MA_Crossover'].diff()
        
        return df['MA_Signal']
    
    def rsi_signals(self, data: pd.DataFrame, oversold: float = 30, 
                   overbought: float = 70) -> pd.Series:
        """
        RSI-based signals.
        Buy when RSI crosses above oversold level, sell when it crosses below overbought.
        
        Args:
            data: DataFrame with RSI indicator
            oversold: Oversold threshold
            overbought: Overbought threshold
            
        Returns:
            Series with RSI signals
        """
        df = data.copy()
        
        df['RSI_Signal'] = 0
        
        # Buy signal: RSI crosses above oversold level
        df.loc[(df['RSI'] > oversold) & (df['RSI'].shift(1) <= oversold), 'RSI_Signal'] = 1
        
        # Sell signal: RSI crosses below overbought level
        df.loc[(df['RSI'] < overbought) & (df['RSI'].shift(1) >= overbought), 'RSI_Signal'] = -1
        
        return df['RSI_Signal']
    
    def macd_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        MACD-based signals.
        Buy when MACD line crosses above signal line, sell when it crosses below.
        
        Args:
            data: DataFrame with MACD indicators
            
        Returns:
            Series with MACD signals
        """
        df = data.copy()
        
        df['MACD_Signal'] = 0
        
        # Buy signal: MACD crosses above signal line
        df.loc[(df['MACD'] > df['MACD_Signal']) & 
               (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1)), 'MACD_Signal'] = 1
        
        # Sell signal: MACD crosses below signal line
        df.loc[(df['MACD'] < df['MACD_Signal']) & 
               (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1)), 'MACD_Signal'] = -1
        
        return df['MACD_Signal']
    
    def bollinger_band_signals(self, data: pd.DataFrame, 
                              oversold_threshold: float = 0.1,
                              overbought_threshold: float = 0.9) -> pd.Series:
        """
        Bollinger Band-based signals.
        Buy when price touches lower band, sell when it touches upper band.
        
        Args:
            data: DataFrame with Bollinger Bands
            oversold_threshold: Threshold for oversold condition
            overbought_threshold: Threshold for overbought condition
            
        Returns:
            Series with Bollinger Band signals
        """
        df = data.copy()
        
        df['BB_Signal'] = 0
        
        # Buy signal: Price near lower band
        df.loc[df['BB_Position'] <= oversold_threshold, 'BB_Signal'] = 1
        
        # Sell signal: Price near upper band
        df.loc[df['BB_Position'] >= overbought_threshold, 'BB_Signal'] = -1
        
        return df['BB_Signal']
    
    def volume_breakout_signals(self, data: pd.DataFrame, 
                               volume_multiplier: float = 2.0,
                               price_change_threshold: float = 0.02) -> pd.Series:
        """
        Volume breakout signals.
        Buy when volume is high and price breaks out.
        
        Args:
            data: DataFrame with volume and price data
            volume_multiplier: Volume threshold multiplier
            price_change_threshold: Minimum price change threshold
            
        Returns:
            Series with volume breakout signals
        """
        df = data.copy()
        
        df['Volume_Breakout_Signal'] = 0
        
        # Calculate price change
        df['Price_Change'] = df['Close'].pct_change()
        
        # Volume breakout condition
        volume_condition = df['Volume_Ratio'] >= volume_multiplier
        price_condition = df['Price_Change'] >= price_change_threshold
        
        # Buy signal: High volume with price breakout
        df.loc[volume_condition & price_condition, 'Volume_Breakout_Signal'] = 1
        
        return df['Volume_Breakout_Signal']
    
    def mean_reversion_signals(self, data: pd.DataFrame, 
                              lookback_period: int = 20,
                              std_dev_threshold: float = 2.0) -> pd.Series:
        """
        Mean reversion signals.
        Buy when price is significantly below moving average.
        
        Args:
            data: DataFrame with price and moving average data
            lookback_period: Period for moving average
            std_dev_threshold: Standard deviation threshold
            
        Returns:
            Series with mean reversion signals
        """
        df = data.copy()
        
        df['Mean_Reversion_Signal'] = 0
        
        # Calculate rolling mean and standard deviation
        df['Rolling_Mean'] = df['Close'].rolling(window=lookback_period).mean()
        df['Rolling_Std'] = df['Close'].rolling(window=lookback_period).std()
        
        # Calculate z-score
        df['Z_Score'] = (df['Close'] - df['Rolling_Mean']) / df['Rolling_Std']
        
        # Buy signal: Price significantly below mean (oversold)
        df.loc[df['Z_Score'] <= -std_dev_threshold, 'Mean_Reversion_Signal'] = 1
        
        # Sell signal: Price significantly above mean (overbought)
        df.loc[df['Z_Score'] >= std_dev_threshold, 'Mean_Reversion_Signal'] = -1
        
        return df['Mean_Reversion_Signal']
    
    def momentum_signals(self, data: pd.DataFrame, 
                        momentum_period: int = 20,
                        momentum_threshold: float = 0.05) -> pd.Series:
        """
        Momentum-based signals.
        Buy when momentum is positive and strong.
        
        Args:
            data: DataFrame with price data
            momentum_period: Period for momentum calculation
            momentum_threshold: Threshold for momentum signal
            
        Returns:
            Series with momentum signals
        """
        df = data.copy()
        
        df['Momentum_Signal'] = 0
        
        # Calculate momentum (rate of change)
        df['Momentum'] = df['Close'].pct_change(periods=momentum_period)
        
        # Buy signal: Strong positive momentum
        df.loc[df['Momentum'] >= momentum_threshold, 'Momentum_Signal'] = 1
        
        # Sell signal: Strong negative momentum
        df.loc[df['Momentum'] <= -momentum_threshold, 'Momentum_Signal'] = -1
        
        return df['Momentum_Signal']
    
    def stochastic_signals(self, data: pd.DataFrame, 
                          oversold: float = 20,
                          overbought: float = 80) -> pd.Series:
        """
        Stochastic oscillator signals.
        Buy when stochastic crosses above oversold level.
        
        Args:
            data: DataFrame with stochastic indicators
            oversold: Oversold threshold
            overbought: Overbought threshold
            
        Returns:
            Series with stochastic signals
        """
        df = data.copy()
        
        df['Stochastic_Signal'] = 0
        
        # Buy signal: Stochastic crosses above oversold level
        df.loc[(df['Stoch_K'] > oversold) & (df['Stoch_K'].shift(1) <= oversold), 'Stochastic_Signal'] = 1
        
        # Sell signal: Stochastic crosses below overbought level
        df.loc[(df['Stoch_K'] < overbought) & (df['Stoch_K'].shift(1) >= overbought), 'Stochastic_Signal'] = -1
        
        return df['Stochastic_Signal']
    
    def atr_breakout_signals(self, data: pd.DataFrame, 
                            atr_multiplier: float = 1.5) -> pd.Series:
        """
        ATR-based breakout signals.
        Buy when price breaks out of ATR range.
        
        Args:
            data: DataFrame with ATR indicator
            atr_multiplier: ATR multiplier for breakout threshold
            
        Returns:
            Series with ATR breakout signals
        """
        df = data.copy()
        
        df['ATR_Breakout_Signal'] = 0
        
        # Calculate ATR-based upper and lower bands
        df['ATR_Upper'] = df['Close'].rolling(window=20).mean() + (df['ATR'] * atr_multiplier)
        df['ATR_Lower'] = df['Close'].rolling(window=20).mean() - (df['ATR'] * atr_multiplier)
        
        # Buy signal: Price breaks above upper band
        df.loc[df['Close'] > df['ATR_Upper'], 'ATR_Breakout_Signal'] = 1
        
        # Sell signal: Price breaks below lower band
        df.loc[df['Close'] < df['ATR_Lower'], 'ATR_Breakout_Signal'] = -1
        
        return df['ATR_Breakout_Signal']
    
    def combined_momentum_volume(self, data: pd.DataFrame) -> pd.Series:
        """
        Combined momentum and volume signal.
        Buy when both momentum and volume are strong.
        
        Args:
            data: DataFrame with momentum and volume indicators
            
        Returns:
            Series with combined signals
        """
        df = data.copy()
        
        df['Combined_Signal'] = 0
        
        # Strong momentum condition
        momentum_condition = (df['ROC_10'] > 0.02) & (df['ROC_20'] > 0.05)
        
        # High volume condition
        volume_condition = df['Volume_Ratio'] > 1.5
        
        # Buy signal: Both momentum and volume are strong
        df.loc[momentum_condition & volume_condition, 'Combined_Signal'] = 1
        
        return df['Combined_Signal']
    
    def generate_all_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all alpha factor signals.
        
        Args:
            data: DataFrame with technical indicators
            
        Returns:
            DataFrame with all signals added
        """
        print("Generating alpha factor signals...")
        
        df = data.copy()
        
        # Generate individual signals
        df['MA_Signal'] = self.moving_average_crossover(df)
        print("✓ Moving Average Crossover signals generated")
        
        df['RSI_Signal'] = self.rsi_signals(df)
        print("✓ RSI signals generated")
        
        df['MACD_Signal'] = self.macd_signals(df)
        print("✓ MACD signals generated")
        
        df['BB_Signal'] = self.bollinger_band_signals(df)
        print("✓ Bollinger Band signals generated")
        
        df['Volume_Breakout_Signal'] = self.volume_breakout_signals(df)
        print("✓ Volume Breakout signals generated")
        
        df['Mean_Reversion_Signal'] = self.mean_reversion_signals(df)
        print("✓ Mean Reversion signals generated")
        
        df['Momentum_Signal'] = self.momentum_signals(df)
        print("✓ Momentum signals generated")
        
        df['Stochastic_Signal'] = self.stochastic_signals(df)
        print("✓ Stochastic signals generated")
        
        df['ATR_Breakout_Signal'] = self.atr_breakout_signals(df)
        print("✓ ATR Breakout signals generated")
        
        df['Combined_Signal'] = self.combined_momentum_volume(df)
        print("✓ Combined Momentum-Volume signals generated")
        
        # Create composite signal (simple average of all signals)
        signal_columns = [col for col in df.columns if col.endswith('_Signal')]
        df['Composite_Signal'] = df[signal_columns].mean(axis=1)
        
        # Normalize composite signal to [-1, 1] range
        df['Composite_Signal'] = np.clip(df['Composite_Signal'], -1, 1)
        
        print("✓ Composite signal generated")
        print("All alpha factor signals generated successfully!")
        
        return df
    
    def get_signal_summary(self, data: pd.DataFrame) -> Dict:
        """
        Get summary statistics for all signals.
        
        Args:
            data: DataFrame with signals
            
        Returns:
            Dictionary with signal summaries
        """
        summary = {}
        
        # Get all signal columns
        signal_columns = [col for col in data.columns if col.endswith('_Signal')]
        
        for col in signal_columns:
            signal_data = data[col].dropna()
            summary[col] = {
                'total_signals': len(signal_data[signal_data != 0]),
                'buy_signals': len(signal_data[signal_data > 0]),
                'sell_signals': len(signal_data[signal_data < 0]),
                'signal_strength_mean': signal_data.mean(),
                'signal_strength_std': signal_data.std()
            }
        
        return summary
    
    def get_factor_correlation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix between different factors.
        
        Args:
            data: DataFrame with signals
            
        Returns:
            Correlation matrix
        """
        signal_columns = [col for col in data.columns if col.endswith('_Signal')]
        return data[signal_columns].corr() 