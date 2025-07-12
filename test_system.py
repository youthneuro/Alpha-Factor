#!/usr/bin/env python3
"""
Test script to verify the alpha factor research system works correctly.
"""

import sys
import traceback

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from data_collector import DataCollector
        print("✓ DataCollector imported successfully")
    except Exception as e:
        print(f"✗ DataCollector import failed: {e}")
        return False
    
    try:
        from indicators import TechnicalIndicators
        print("✓ TechnicalIndicators imported successfully")
    except Exception as e:
        print(f"✗ TechnicalIndicators import failed: {e}")
        return False
    
    try:
        from factors import AlphaFactors
        print("✓ AlphaFactors imported successfully")
    except Exception as e:
        print(f"✗ AlphaFactors import failed: {e}")
        return False
    
    try:
        from backtester import Backtester
        print("✓ Backtester imported successfully")
    except Exception as e:
        print(f"✗ Backtester import failed: {e}")
        return False
    
    try:
        from performance import PerformanceAnalyzer
        print("✓ PerformanceAnalyzer imported successfully")
    except Exception as e:
        print(f"✗ PerformanceAnalyzer import failed: {e}")
        return False
    
    try:
        from utils import AlphaFactorUtils
        print("✓ AlphaFactorUtils imported successfully")
    except Exception as e:
        print(f"✗ AlphaFactorUtils import failed: {e}")
        return False
    
    return True

def test_data_collection():
    """Test data collection functionality."""
    print("\nTesting data collection...")
    
    try:
        from data_collector import DataCollector
        collector = DataCollector()
        
        # Test single stock data collection
        data = collector.get_stock_data('AAPL', '2023-01-01', '2023-12-31')
        if data is not None and not data.empty:
            print("✓ Single stock data collection successful")
            print(f"  Data shape: {data.shape}")
            print(f"  Columns: {list(data.columns)}")
        else:
            print("✗ Single stock data collection failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Data collection test failed: {e}")
        traceback.print_exc()
        return False

def test_indicators():
    """Test technical indicators calculation."""
    print("\nTesting technical indicators...")
    
    try:
        from data_collector import DataCollector
        from indicators import TechnicalIndicators
        
        # Get sample data
        collector = DataCollector()
        data = collector.get_stock_data('AAPL', '2023-01-01', '2023-12-31')
        
        if data is None or data.empty:
            print("✗ No data available for indicator testing")
            return False
        
        # Test indicators
        indicators = TechnicalIndicators()
        data_with_indicators = indicators.add_all_indicators(data)
        
        if data_with_indicators is not None and not data_with_indicators.empty:
            print("✓ Technical indicators calculation successful")
            print(f"  Original columns: {len(data.columns)}")
            print(f"  Columns with indicators: {len(data_with_indicators.columns)}")
            
            # Check for some key indicators
            key_indicators = ['RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'ATR']
            found_indicators = [ind for ind in key_indicators if ind in data_with_indicators.columns]
            print(f"  Key indicators found: {found_indicators}")
            
        else:
            print("✗ Technical indicators calculation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Technical indicators test failed: {e}")
        traceback.print_exc()
        return False

def test_factors():
    """Test alpha factor generation."""
    print("\nTesting alpha factors...")
    
    try:
        from data_collector import DataCollector
        from indicators import TechnicalIndicators
        from factors import AlphaFactors
        
        # Get data with indicators
        collector = DataCollector()
        data = collector.get_stock_data('AAPL', '2023-01-01', '2023-12-31')
        
        if data is None or data.empty:
            print("✗ No data available for factor testing")
            return False
        
        indicators = TechnicalIndicators()
        data_with_indicators = indicators.add_all_indicators(data)
        
        # Test factors
        factors = AlphaFactors()
        data_with_signals = factors.generate_all_signals(data_with_indicators)
        
        if data_with_signals is not None and not data_with_signals.empty:
            print("✓ Alpha factor generation successful")
            
            # Check for signal columns
            signal_columns = [col for col in data_with_signals.columns if col.endswith('_Signal')]
            print(f"  Signal columns generated: {len(signal_columns)}")
            print(f"  Signal columns: {signal_columns}")
            
        else:
            print("✗ Alpha factor generation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Alpha factors test failed: {e}")
        traceback.print_exc()
        return False

def test_backtesting():
    """Test backtesting functionality."""
    print("\nTesting backtesting...")
    
    try:
        from data_collector import DataCollector
        from indicators import TechnicalIndicators
        from factors import AlphaFactors
        from backtester import Backtester
        
        # Get data with signals
        collector = DataCollector()
        data = collector.get_stock_data('AAPL', '2023-01-01', '2023-12-31')
        
        if data is None or data.empty:
            print("✗ No data available for backtesting")
            return False
        
        indicators = TechnicalIndicators()
        data_with_indicators = indicators.add_all_indicators(data)
        
        factors = AlphaFactors()
        data_with_signals = factors.generate_all_signals(data_with_indicators)
        
        # Test backtesting
        backtester = Backtester()
        results = backtester.run_backtest(data_with_signals)
        
        if results and len(results) > 0:
            print("✓ Backtesting successful")
            print(f"  Strategies tested: {len(results)}")
            
            # Check for key metrics
            for strategy, result in results.items():
                if 'sharpe_ratio' in result:
                    print(f"  {strategy}: Sharpe={result['sharpe_ratio']:.3f}, Return={result['total_return']*100:.2f}%")
            
        else:
            print("✗ Backtesting failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Backtesting test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("ALPHA FACTOR RESEARCH SYSTEM - TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_data_collection,
        test_indicators,
        test_factors,
        test_backtesting
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ Test {test.__name__} failed")
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 