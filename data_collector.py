import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class DataCollector:
    """
    Collects stock market data from yfinance for alpha factor research.
    """
    
    def __init__(self):
        self.sp500_symbols = self._get_sp500_symbols()
    
    def _get_sp500_symbols(self) -> List[str]:
        """
        Get S&P 500 symbols. For simplicity, using a subset of major stocks.
        """
        # Major S&P 500 stocks for demonstration
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'JPM', 'JNJ', 'PG', 'UNH', 'HD', 'DIS', 'PYPL', 'ADBE', 'CRM',
            'INTC', 'V', 'WMT', 'MA', 'BAC', 'KO', 'PFE', 'ABT', 'TMO',
            'AVGO', 'COST', 'PEP', 'DHR', 'ABBV', 'LLY', 'TXN', 'NKE'
        ]
        return symbols
    
    def get_stock_data(self, symbol: str, start_date: str = '2020-01-01', 
                       end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Fetch individual stock data from yfinance.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data collection
            end_date: End date for data collection (defaults to today)
        
        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                print(f"No data found for {symbol}")
                return None
            
            # Add symbol column
            data['Symbol'] = symbol
            
            # Calculate daily returns
            data['Returns'] = data['Close'].pct_change()
            
            # Calculate log returns
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_sp500_data(self, start_date: str = '2020-01-01', 
                       end_date: str = None, max_stocks: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for S&P 500 stocks.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            max_stocks: Maximum number of stocks to fetch (for performance)
        
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        print(f"Fetching data for {min(max_stocks, len(self.sp500_symbols))} stocks...")
        
        data_dict = {}
        symbols_to_fetch = self.sp500_symbols[:max_stocks]
        
        for i, symbol in enumerate(symbols_to_fetch, 1):
            print(f"Fetching {symbol} ({i}/{len(symbols_to_fetch)})")
            data = self.get_stock_data(symbol, start_date, end_date)
            if data is not None:
                data_dict[symbol] = data
        
        print(f"Successfully fetched data for {len(data_dict)} stocks")
        return data_dict
    
    def get_combined_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine all stock data into a single DataFrame with multi-level index.
        
        Args:
            data_dict: Dictionary of stock data
            
        Returns:
            Combined DataFrame with (Symbol, Date) multi-index
        """
        combined_data = []
        
        for symbol, data in data_dict.items():
            data_copy = data.copy()
            data_copy['Symbol'] = symbol
            combined_data.append(data_copy)
        
        if combined_data:
            combined_df = pd.concat(combined_data, axis=0)
            combined_df = combined_df.set_index(['Symbol', combined_df.index])
            return combined_df
        else:
            return pd.DataFrame()
    
    def get_market_data(self, symbol: str = '^GSPC', start_date: str = '2020-01-01',
                       end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Fetch market index data (S&P 500).
        
        Args:
            symbol: Market index symbol (default: S&P 500)
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with market data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                print(f"No market data found for {symbol}")
                return None
            
            # Add market returns
            data['Market_Returns'] = data['Close'].pct_change()
            data['Market_Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            
            return data
            
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate data quality.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if data.empty:
            return False
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # Check for missing values
        if data[required_columns].isnull().any().any():
            print("Warning: Missing values detected in data")
        
        # Check for zero or negative prices
        if (data[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
            print("Warning: Zero or negative prices detected")
            return False
        
        return True 