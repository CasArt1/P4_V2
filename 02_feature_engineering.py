"""
Feature Engineering for NVDA Trading Strategy
Creates 20+ technical indicators from OHLCV data
"""

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath="data/NVDA_raw_data.csv"):
    """
    Load raw OHLCV data
    
    Args:
        filepath (str): Path to raw data CSV
    
    Returns:
        pd.DataFrame: Raw price data
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f" Loaded {len(df)} rows")
    return df

def add_momentum_indicators(df):
    """
    Add momentum-based technical indicators
    
    Args:
        df (pd.DataFrame): Price data with OHLCV columns
    
    Returns:
        pd.DataFrame: Data with momentum features added
    """
    print("\n Adding Momentum Indicators...")
    
    # RSI - Multiple periods
    rsi_14 = RSIIndicator(close=df['Close'], window=14)
    df['rsi_14'] = rsi_14.rsi()
    
    rsi_21 = RSIIndicator(close=df['Close'], window=21)
    df['rsi_21'] = rsi_21.rsi()
    
    # MACD
    macd = MACD(close=df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Rate of Change
    roc = ROCIndicator(close=df['Close'], window=12)
    df['roc'] = roc.roc()
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=14,
        smooth_window=3
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # ADX - Trend Strength
    adx = ADXIndicator(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=14
    )
    df['adx'] = adx.adx()
    
    print(f"   Added 10 momentum features")
    return df

def add_volatility_indicators(df):
    """
    Add volatility-based technical indicators
    
    Args:
        df (pd.DataFrame): Price data with OHLCV columns
    
    Returns:
        pd.DataFrame: Data with volatility features added
    """
    print("\n📊 Adding Volatility Indicators...")
    
    # Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
    df['bb_position'] = (df['Close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
    
    # ATR - Average True Range
    atr = AverageTrueRange(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=14
    )
    df['atr'] = atr.average_true_range()
    df['atr_percent'] = (df['atr'] / df['Close']) * 100
    
    # Historical Volatility
    df['returns'] = df['Close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
    
    print(f"   Added 8 volatility features")
    return df

def add_volume_indicators(df):
    """
    Add volume-based technical indicators
    
    Args:
        df (pd.DataFrame): Price data with OHLCV columns
    
    Returns:
        pd.DataFrame: Data with volume features added
    """
    print("\n📊 Adding Volume Indicators...")
    
    # On-Balance Volume (OBV)
    obv = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'])
    df['obv'] = obv.on_balance_volume()
    
    # Volume Rate of Change
    df['volume_roc'] = df['Volume'].pct_change(periods=5) * 100
    
    # VWAP
    vwap = VolumeWeightedAveragePrice(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        volume=df['Volume'],
        window=14
    )
    df['vwap'] = vwap.volume_weighted_average_price()
    df['vwap_distance'] = ((df['Close'] - df['vwap']) / df['vwap']) * 100
    
    # Volume moving average
    df['volume_ma_20'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma_20']
    
    print(f"   Added 6 volume features")
    return df

def add_price_features(df):
    """
    Add price-based features
    
    Args:
        df (pd.DataFrame): Price data
    
    Returns:
        pd.DataFrame: Data with price features added
    """
    print("\n Adding Price Features...")
    
    # Simple Moving Averages
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    
    # Price position relative to moving averages
    df['price_to_sma20'] = ((df['Close'] - df['sma_20']) / df['sma_20']) * 100
    df['price_to_sma50'] = ((df['Close'] - df['sma_50']) / df['sma_50']) * 100
    
    # Daily price change
    df['price_change'] = df['Close'].pct_change() * 100
    
    print(f"   Added 5 price features")
    return df

def create_target_variable(df, forward_period=5, threshold=2.0):
    """
    Create target variable based on forward returns
    
    Target Labels:
    - Long (1): forward return > +threshold%
    - Hold (0): forward return between -threshold% and +threshold%
    - Short (-1): forward return < -threshold%
    
    Args:
        df (pd.DataFrame): Price data
        forward_period (int): Days to look forward
        threshold (float): Percentage threshold for signals
    
    Returns:
        pd.DataFrame: Data with target variable added
    """
    print(f"\n🎯 Creating Target Variable...")
    print(f"  Forward period: {forward_period} days")
    print(f"  Threshold: ±{threshold}%")
    
    # Calculate forward returns
    df['forward_return'] = (
        (df['Close'].shift(-forward_period) - df['Close']) / df['Close']
    ) * 100
    
    # Create target labels
    df['target'] = 0  # Default: Hold
    df.loc[df['forward_return'] > threshold, 'target'] = 1   # Long
    df.loc[df['forward_return'] < -threshold, 'target'] = -1  # Short
    
    # Remove last rows where we can't calculate forward returns
    df = df[:-forward_period]
    
    # Class distribution
    target_counts = df['target'].value_counts().sort_index()
    print(f"\n  Target Distribution:")
    print(f"    Short (-1): {target_counts.get(-1, 0):,} ({target_counts.get(-1, 0)/len(df)*100:.1f}%)")
    print(f"    Hold  ( 0): {target_counts.get(0, 0):,} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"    Long  ( 1): {target_counts.get(1, 0):,} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    return df

def normalize_features(df, feature_columns):
    """
    Normalize features using z-score normalization
    
    Args:
        df (pd.DataFrame): Data with features
        feature_columns (list): List of feature column names
    
    Returns:
        pd.DataFrame: Data with normalized features
    """
    print(f"\n Normalizing {len(feature_columns)} features...")
    
    for col in feature_columns:
        mean = df[col].mean()
        std = df[col].std()
        df[f'{col}_norm'] = (df[col] - mean) / (std + 1e-8)  # Add small value to avoid division by zero
    
    print(f"   Normalization complete")
    return df

def split_data(df, train_pct=0.6, val_pct=0.2, test_pct=0.2):
    """
    Split data chronologically into train/validation/test sets
    
    Args:
        df (pd.DataFrame): Full dataset
        train_pct (float): Training set percentage
        val_pct (float): Validation set percentage
        test_pct (float): Test set percentage
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print(f"\n Splitting Data (Train: {train_pct*100}%, Val: {val_pct*100}%, Test: {test_pct*100}%)...")
    
    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"  Training:   {len(train_df):,} rows ({train_df.index[0]} to {train_df.index[-1]})")
    print(f"  Validation: {len(val_df):,} rows ({val_df.index[0]} to {val_df.index[-1]})")
    print(f"  Test:       {len(test_df):,} rows ({test_df.index[0]} to {test_df.index[-1]})")
    
    return train_df, val_df, test_df

def save_processed_data(df, train_df, val_df, test_df):
    """
    Save processed datasets to CSV
    
    Args:
        df (pd.DataFrame): Full processed dataset
        train_df (pd.DataFrame): Training set
        val_df (pd.DataFrame): Validation set
        test_df (pd.DataFrame): Test set
    """
    print(f"\n Saving Processed Data...")
    
    # Save full dataset
    df.to_csv('data/NVDA_features.csv')
    print(f"   Full dataset: data/NVDA_features.csv")
    
    # Save splits
    train_df.to_csv('data/NVDA_train.csv')
    val_df.to_csv('data/NVDA_val.csv')
    test_df.to_csv('data/NVDA_test.csv')
    print(f"   Train set: data/NVDA_train.csv")
    print(f"   Val set: data/NVDA_val.csv")
    print(f"   Test set: data/NVDA_test.csv")

def main():
    """Main feature engineering pipeline"""
    
    print("="*60)
    print("NVDA TRADING STRATEGY - FEATURE ENGINEERING")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Add features
    df = add_momentum_indicators(df)
    df = add_volatility_indicators(df)
    df = add_volume_indicators(df)
    df = add_price_features(df)
    
    # Create target variable
    df = create_target_variable(df, forward_period=5, threshold=2.0)
    
    # Drop rows with NaN values (from indicator calculations)
    print(f"\n🧹 Cleaning data...")
    initial_rows = len(df)
    df = df.dropna()
    final_rows = len(df)
    print(f"  Removed {initial_rows - final_rows:,} rows with NaN values")
    print(f"  Final dataset: {final_rows:,} rows")
    
    # Get feature columns (exclude OHLCV, target, and intermediate columns)
    feature_columns = [col for col in df.columns if col not in [
        'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
        'target', 'forward_return', 'returns'
    ]]
    
    print(f"\n📋 Total Features Created: {len(feature_columns)}")
    print(f"  Features: {', '.join(feature_columns[:5])}...")
    
    # Normalize features
    df = normalize_features(df, feature_columns)
    
    # Split data
    train_df, val_df, test_df = split_data(df)
    
    # Save processed data
    save_processed_data(df, train_df, val_df, test_df)
    
    print("\n" + "="*60)
    print(" Phase 2 Complete: Feature Engineering Done!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Review the features in: data/NVDA_features.csv")
    print(f"2. Check class balance in target variable")
    print(f"3. Move to Phase 3: Model Development")
    
    return df

if __name__ == "__main__":
    df = main()
