import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """
    Calculates various technical indicators for alpha factor research.
    """
    
    def __init__(self):
        self.indicators = {}
    
    def add_moving_averages(self, data: pd.DataFrame, 
                           periods: List[int] = [5, 10, 20, 50, 100, 200]) -> pd.DataFrame:
        """
        Add simple moving averages to the data.
        
        Args:
            data: DataFrame with OHLCV data
            periods: List of periods for moving averages
            
        Returns:
            DataFrame with moving averages added
        """
        df = data.copy()
        
        for period in periods:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        return df
    
    def add_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI).
        
        Args:
            data: DataFrame with OHLCV data
            period: Period for RSI calculation
            
        Returns:
            DataFrame with RSI added
        """
        df = data.copy()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=period).rsi()
        return df
    
    def add_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, 
                 signal: int = 9) -> pd.DataFrame:
        """
        Add MACD (Moving Average Convergence Divergence).
        
        Args:
            data: DataFrame with OHLCV data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            DataFrame with MACD indicators added
        """
        df = data.copy()
        
        macd_indicator = ta.trend.MACD(df['Close'], window_fast=fast, 
                                      window_slow=slow, window_sign=signal)
        
        df['MACD'] = macd_indicator.macd()
        df['MACD_Signal'] = macd_indicator.macd_signal()
        df['MACD_Histogram'] = macd_indicator.macd_diff()
        
        return df
    
    def add_bollinger_bands(self, data: pd.DataFrame, period: int = 20, 
                           std_dev: float = 2) -> pd.DataFrame:
        """
        Add Bollinger Bands.
        
        Args:
            data: DataFrame with OHLCV data
            period: Period for moving average
            std_dev: Number of standard deviations
            
        Returns:
            DataFrame with Bollinger Bands added
        """
        df = data.copy()
        
        bb_indicator = ta.volatility.BollingerBands(df['Close'], window=period, 
                                                   window_dev=std_dev)
        
        df['BB_Upper'] = bb_indicator.bollinger_hband()
        df['BB_Middle'] = bb_indicator.bollinger_mavg()
        df['BB_Lower'] = bb_indicator.bollinger_lband()
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        return df
    
    def add_stochastic(self, data: pd.DataFrame, k_period: int = 14, 
                      d_period: int = 3) -> pd.DataFrame:
        """
        Add Stochastic Oscillator.
        
        Args:
            data: DataFrame with OHLCV data
            k_period: %K period
            d_period: %D period
            
        Returns:
            DataFrame with Stochastic indicators added
        """
        df = data.copy()
        
        stoch_indicator = ta.momentum.StochasticOscillator(df['High'], df['Low'], 
                                                          df['Close'], window=k_period, 
                                                          smooth_window=d_period)
        
        df['Stoch_K'] = stoch_indicator.stoch()
        df['Stoch_D'] = stoch_indicator.stoch_signal()
        
        return df
    
    def add_atr(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Average True Range (ATR).
        
        Args:
            data: DataFrame with OHLCV data
            period: Period for ATR calculation
            
        Returns:
            DataFrame with ATR added
        """
        df = data.copy()
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], 
                                                   df['Close'], window=period).average_true_range()
        return df
    
    def add_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume indicators added
        """
        df = data.copy()
        
        # Volume moving average
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # On-Balance Volume (OBV)
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        
        # Volume Price Trend (VPT)
        df['VPT'] = ta.volume.VolumePriceTrendIndicator(df['Close'], df['Volume']).volume_price_trend()
        
        # Accumulation/Distribution Line
        df['ADL'] = ta.volume.AccDistIndexIndicator(df['High'], df['Low'], 
                                                   df['Close'], df['Volume']).acc_dist_index()
        
        return df
    
    def add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with momentum indicators added
        """
        df = data.copy()
        
        # Rate of Change (ROC)
        df['ROC_10'] = ta.momentum.ROCIndicator(df['Close'], window=10).roc()
        df['ROC_20'] = ta.momentum.ROCIndicator(df['Close'], window=20).roc()
        
        # Williams %R
        df['Williams_R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], 
                                                         df['Close'], lbp=14).williams_r()
        
        # Money Flow Index (MFI)
        df['MFI'] = ta.volume.MFIIndicator(df['High'], df['Low'], 
                                           df['Close'], df['Volume'], window=14).money_flow_index()
        
        # Commodity Channel Index (CCI)
        df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], 
                                         df['Close'], window=20).cci()
        
        return df
    
    def add_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with trend indicators added
        """
        df = data.copy()
        
        # Average Directional Index (ADX)
        adx_indicator = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        df['ADX'] = adx_indicator.adx()
        df['DI_Plus'] = adx_indicator.adx_pos()
        df['DI_Minus'] = adx_indicator.adx_neg()
        
        # Parabolic SAR
        df['PSAR'] = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close']).psar()
        
        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
        df['Ichimoku_A'] = ichimoku.ichimoku_a()
        df['Ichimoku_B'] = ichimoku.ichimoku_b()
        df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
        df['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
        
        return df
    
    def add_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility indicators added
        """
        df = data.copy()
        
        # Keltner Channel
        keltner = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
        df['Keltner_Upper'] = keltner.keltner_channel_hband()
        df['Keltner_Middle'] = keltner.keltner_channel_mband()
        df['Keltner_Lower'] = keltner.keltner_channel_lband()
        
        # Donchian Channel
        donchian = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'])
        df['Donchian_Upper'] = donchian.donchian_channel_hband()
        df['Donchian_Middle'] = donchian.donchian_channel_mband()
        df['Donchian_Lower'] = donchian.donchian_channel_lband()
        
        return df
    
    def add_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators added
        """
        print("Adding technical indicators...")
        
        df = data.copy()
        
        # Add moving averages
        df = self.add_moving_averages(df)
        print("✓ Moving averages added")
        
        # Add RSI
        df = self.add_rsi(df)
        print("✓ RSI added")
        
        # Add MACD
        df = self.add_macd(df)
        print("✓ MACD added")
        
        # Add Bollinger Bands
        df = self.add_bollinger_bands(df)
        print("✓ Bollinger Bands added")
        
        # Add Stochastic
        df = self.add_stochastic(df)
        print("✓ Stochastic added")
        
        # Add ATR
        df = self.add_atr(df)
        print("✓ ATR added")
        
        # Add volume indicators
        df = self.add_volume_indicators(df)
        print("✓ Volume indicators added")
        
        # Add momentum indicators
        df = self.add_momentum_indicators(df)
        print("✓ Momentum indicators added")
        
        # Add trend indicators
        df = self.add_trend_indicators(df)
        print("✓ Trend indicators added")
        
        # Add volatility indicators
        df = self.add_volatility_indicators(df)
        print("✓ Volatility indicators added")
        
        print("All technical indicators added successfully!")
        return df
    
    def get_indicator_summary(self, data: pd.DataFrame) -> Dict:
        """
        Get summary statistics for all indicators.
        
        Args:
            data: DataFrame with indicators
            
        Returns:
            Dictionary with indicator summaries
        """
        summary = {}
        
        # Get all indicator columns (exclude OHLCV and basic columns)
        basic_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 
                        'Log_Returns', 'Symbol']
        indicator_columns = [col for col in data.columns if col not in basic_columns]
        
        for col in indicator_columns:
            if data[col].dtype in ['float64', 'int64']:
                summary[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'null_count': data[col].isnull().sum()
                }
        
        return summary 