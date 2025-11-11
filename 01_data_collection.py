"""
Data Collection Script for NVDA Trading Strategy
Downloads 15 years of historical daily price data
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def download_stock_data(ticker="NVDA", years=15):
    """
    Download historical stock data
    
    Args:
        ticker (str): Stock ticker symbol
        years (int): Number of years of historical data
    
    Returns:
        pd.DataFrame: Historical OHLCV data
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    print(f"Downloading {ticker} data from {start_date.date()} to {end_date.date()}...")
    
    # Download data
    data = yf.download(ticker, start=start_date, end=end_date, progress=True)
    
    # Basic data info
    print(f"\nData downloaded successfully!")
    print(f"Total rows: {len(data)}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"\nColumns: {list(data.columns)}")
    
    # Check for missing values
    missing = data.isnull().sum()
    if missing.any():
        print(f"\n⚠️ Missing values detected:")
        print(missing[missing > 0])
    else:
        print("\n✅ No missing values detected!")
    
    return data

def save_raw_data(data, ticker="NVDA"):
    """
    Save raw data to CSV
    
    Args:
        data (pd.DataFrame): Stock data
        ticker (str): Ticker symbol for filename
    """
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save to CSV
    filepath = f"data/{ticker}_raw_data.csv"
    data.to_csv(filepath)
    print(f"\n💾 Data saved to: {filepath}")
    
    return filepath

def basic_data_quality_check(data):
    """
    Perform basic data quality checks
    
    Args:
        data (pd.DataFrame): Stock data
    """
    print("\n" + "="*50)
    print("DATA QUALITY CHECK")
    print("="*50)
    
    # Check for duplicates
    duplicates = data.index.duplicated().sum()
    print(f"Duplicate dates: {duplicates}")
    
    # Check for gaps (missing trading days)
    date_diff = data.index.to_series().diff()
    max_gap = date_diff.max().days
    print(f"Maximum gap between dates: {max_gap} days")
    
    # Price statistics
    print(f"\nPrice Statistics (Close):")
    print(f"  Min: ${data['Close'].min():.2f}")
    print(f"  Max: ${data['Close'].max():.2f}")
    print(f"  Mean: ${data['Close'].mean():.2f}")
    print(f"  Current: ${data['Close'].iloc[-1]:.2f}")
    
    # Volume statistics
    print(f"\nVolume Statistics:")
    print(f"  Min: {data['Volume'].min():,.0f}")
    print(f"  Max: {data['Volume'].max():,.0f}")
    print(f"  Mean: {data['Volume'].mean():,.0f}")
    
    # Check for zero or negative prices
    zero_prices = (data['Close'] <= 0).sum()
    if zero_prices > 0:
        print(f"\n⚠️ Warning: {zero_prices} rows with zero or negative prices!")
    else:
        print(f"\n✅ All prices are positive")
    
    # Check for zero volume
    zero_volume = (data['Volume'] == 0).sum()
    if zero_volume > 0:
        print(f"⚠️ Warning: {zero_volume} rows with zero volume")
    else:
        print(f"✅ No zero volume days")

def main():
    """Main execution function"""
    
    print("="*50)
    print("NVDA TRADING STRATEGY - DATA COLLECTION")
    print("="*50)
    
    # Download data
    ticker = "NVDA"
    data = download_stock_data(ticker=ticker, years=15)
    
    # Quality check
    basic_data_quality_check(data)
    
    # Save data
    filepath = save_raw_data(data, ticker=ticker)
    
    print("\n" + "="*50)
    print("✅ Phase 1 Complete: Data Collection Done!")
    print("="*50)
    print(f"\nNext steps:")
    print(f"1. Review the data in: {filepath}")
    print(f"2. Run EDA notebook for deeper analysis")
    print(f"3. Move to Phase 2: Feature Engineering")
    
    return data

if __name__ == "__main__":
    data = main()
