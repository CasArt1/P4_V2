"""
Phase 6: Backtesting Engine
Simulate trading strategy with realistic transaction costs and calculate performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================
# CONFIGURATION
# ============================================================

class BacktestConfig:
    """Backtesting configuration parameters"""
    
    # Model
    MODEL_PATH = "models/saved_models/CustomCNN.h5"  # Best model
    
    # Trading costs
    COMMISSION_RATE = 0.00125  # 0.125% per trade
    BORROW_RATE_ANNUAL = 0.0025  # 0.25% annualized for shorts
    
    # Strategy parameters
    INITIAL_CAPITAL = 100000  # $100,000
    POSITION_SIZE = 1.0  # 100% of capital (all-in strategy)
    
    # Risk management (optional)
    STOP_LOSS = 0.10  # 10% stop loss
    TAKE_PROFIT = None  # Set to percentage (e.g., 0.10 for 10%) or None
    
    # Sequence length (must match model training)
    SEQUENCE_LENGTH = 10

# ============================================================
# DATA LOADING
# ============================================================

def load_model():
    """Load trained model"""
    print(f"Loading model from {BacktestConfig.MODEL_PATH}...")
    model = keras.models.load_model(BacktestConfig.MODEL_PATH)
    print(f" Model loaded successfully")
    return model

def load_backtest_data():
    """Load data for backtesting"""
    print("Loading data...")
    
    train_df = pd.read_csv('data/NVDA_train.csv', index_col=0, parse_dates=True)
    val_df = pd.read_csv('data/NVDA_val.csv', index_col=0, parse_dates=True)
    test_df = pd.read_csv('data/NVDA_test.csv', index_col=0, parse_dates=True)
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df

# ============================================================
# SIGNAL GENERATION
# ============================================================

def create_sequences(features, sequence_length=10):
    """Create sequences for model prediction"""
    sequences = []
    for i in range(len(features) - sequence_length + 1):
        sequences.append(features[i:i+sequence_length])
    return np.array(sequences)

def generate_signals(model, df, sequence_length=10):
    """
    Generate trading signals using the model
    
    Returns:
        DataFrame with signals added
    """
    print("\n Generating trading signals...")
    
    # Get normalized features
    feature_cols = [col for col in df.columns if col.endswith('_norm')]
    features = df[feature_cols].values
    
    # Create sequences
    sequences = create_sequences(features, sequence_length)
    
    # Generate predictions
    predictions = model.predict(sequences, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Convert from [0,1,2] to [-1,0,1]
    signals = predicted_classes - 1
    
    # Add signals to dataframe (skip first sequence_length-1 rows)
    df_with_signals = df.iloc[sequence_length-1:].copy()
    df_with_signals['signal'] = signals
    df_with_signals['confidence'] = np.max(predictions, axis=1)
    
    # Signal distribution
    signal_counts = pd.Series(signals).value_counts().sort_index()
    print(f"  Short (-1): {signal_counts.get(-1, 0)} ({signal_counts.get(-1, 0)/len(signals)*100:.1f}%)")
    print(f"  Hold  ( 0): {signal_counts.get(0, 0)} ({signal_counts.get(0, 0)/len(signals)*100:.1f}%)")
    print(f"  Long  ( 1): {signal_counts.get(1, 0)} ({signal_counts.get(1, 0)/len(signals)*100:.1f}%)")
    
    return df_with_signals

# ============================================================
# BACKTESTING ENGINE
# ============================================================

class BacktestEngine:
    """Backtesting engine with realistic transaction costs"""
    
    def __init__(self, config):
        self.config = config
        self.trades = []
        self.equity_curve = []
        
    def calculate_commission(self, trade_value):
        """Calculate commission for a trade"""
        return abs(trade_value) * self.config.COMMISSION_RATE
    
    def calculate_borrow_cost(self, position_value, days_held):
        """Calculate borrowing cost for short positions"""
        if position_value < 0:  # Short position
            daily_rate = self.config.BORROW_RATE_ANNUAL / 252  # Trading days
            return abs(position_value) * daily_rate * days_held
        return 0
    
    def run_backtest(self, df):
        """
        Run backtest on data with signals
        
        Args:
            df: DataFrame with signals and prices
        
        Returns:
            results: Dictionary with backtest results
        """
        print("\n Running backtest...")
        
        # Initialize
        capital = self.config.INITIAL_CAPITAL
        position = 0  # Current position: -1 (short), 0 (none), 1 (long)
        shares = 0
        entry_price = 0
        entry_date = None
        
        # Track equity
        equity = capital
        
        for idx, row in df.iterrows():
            current_price = row['Close']
            signal = row['signal']
            
            # Calculate current equity
            if position != 0:
                position_value = shares * current_price
                equity = capital + position_value
            else:
                equity = capital
            
            self.equity_curve.append({
                'date': idx,
                'equity': equity,
                'position': position
            })
            
            # Check stop loss / take profit
            stop_loss_triggered = False
            take_profit_triggered = False
            
            if position != 0 and entry_price > 0:
                price_change_pct = (current_price - entry_price) / entry_price
                
                # For long positions
                if position == 1:
                    if self.config.STOP_LOSS and price_change_pct <= -self.config.STOP_LOSS:
                        stop_loss_triggered = True
                    if self.config.TAKE_PROFIT and price_change_pct >= self.config.TAKE_PROFIT:
                        take_profit_triggered = True
                
                # For short positions (inverse logic)
                elif position == -1:
                    if self.config.STOP_LOSS and price_change_pct >= self.config.STOP_LOSS:
                        stop_loss_triggered = True
                    if self.config.TAKE_PROFIT and price_change_pct <= -self.config.TAKE_PROFIT:
                        take_profit_triggered = True
            
            # Trading logic
            if signal != position or stop_loss_triggered or take_profit_triggered:  # Signal changed or risk management triggered
                
                # Close existing position
                if position != 0:
                    exit_price = current_price
                    exit_value = shares * exit_price
                    
                    # Calculate P&L
                    if position == 1:  # Long position
                        pnl = exit_value - (shares * entry_price)
                    else:  # Short position
                        pnl = (shares * entry_price) - exit_value
                    
                    # Calculate costs
                    exit_commission = self.calculate_commission(exit_value)
                    days_held = (idx - entry_date).days
                    borrow_cost = self.calculate_borrow_cost(
                        shares * entry_price if position == -1 else 0,
                        days_held
                    )
                    
                    total_costs = exit_commission + borrow_cost
                    net_pnl = pnl - total_costs
                    
                    # Update capital
                    capital += net_pnl
                    
                    # Determine exit reason
                    if stop_loss_triggered:
                        exit_reason = 'STOP_LOSS'
                    elif take_profit_triggered:
                        exit_reason = 'TAKE_PROFIT'
                    else:
                        exit_reason = 'SIGNAL'
                    
                    # Record trade
                    self.trades.append({
                        'entry_date': entry_date,
                        'exit_date': idx,
                        'position_type': 'LONG' if position == 1 else 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': shares,
                        'days_held': days_held,
                        'gross_pnl': pnl,
                        'commission': exit_commission,
                        'borrow_cost': borrow_cost,
                        'net_pnl': net_pnl,
                        'return_pct': (net_pnl / (shares * entry_price)) * 100,
                        'exit_reason': exit_reason
                    })
                    
                    # Reset position
                    position = 0
                    shares = 0
                
                # Open new position (if signal is not hold)
                if signal != 0:
                    position = signal
                    entry_price = current_price
                    entry_date = idx
                    
                    # Calculate shares (use percentage of capital)
                    position_value = capital * self.config.POSITION_SIZE
                    shares = position_value / entry_price
                    
                    # Entry commission
                    entry_commission = self.calculate_commission(position_value)
                    capital -= entry_commission
        
        # Close any remaining position at end
        if position != 0:
            exit_price = df.iloc[-1]['Close']
            exit_value = shares * exit_price
            
            if position == 1:
                pnl = exit_value - (shares * entry_price)
            else:
                pnl = (shares * entry_price) - exit_value
            
            exit_commission = self.calculate_commission(exit_value)
            days_held = (df.index[-1] - entry_date).days
            borrow_cost = self.calculate_borrow_cost(
                shares * entry_price if position == -1 else 0,
                days_held
            )
            
            net_pnl = pnl - exit_commission - borrow_cost
            capital += net_pnl
            
            self.trades.append({
                'entry_date': entry_date,
                'exit_date': df.index[-1],
                'position_type': 'LONG' if position == 1 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'shares': shares,
                'days_held': days_held,
                'gross_pnl': pnl,
                'commission': exit_commission,
                'borrow_cost': borrow_cost,
                'net_pnl': net_pnl,
                'return_pct': (net_pnl / (shares * entry_price)) * 100,
                'exit_reason': 'END_OF_PERIOD'
            })
        
        # Calculate final equity
        final_equity = capital
        
        print(f" Backtest complete!")
        print(f"  Total trades: {len(self.trades)}")
        print(f"  Final equity: ${final_equity:,.2f}")
        
        return self.calculate_metrics(final_equity)
    
    def calculate_metrics(self, final_equity):
        """Calculate performance metrics"""
        
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Basic metrics
        total_return = ((final_equity - self.config.INITIAL_CAPITAL) / 
                       self.config.INITIAL_CAPITAL) * 100
        
        # Trade statistics
        if len(trades_df) > 0:
            winning_trades = trades_df[trades_df['net_pnl'] > 0]
            losing_trades = trades_df[trades_df['net_pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades_df) * 100
            avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0
            
            # Calculate returns for Sharpe/Sortino
            equity_df['returns'] = equity_df['equity'].pct_change()
            returns = equity_df['returns'].dropna()
            
            # Sharpe Ratio (assuming 252 trading days, 0% risk-free rate)
            if returns.std() != 0:
                sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
            else:
                sharpe_ratio = 0
            
            # Sortino Ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() != 0:
                sortino_ratio = np.sqrt(252) * (returns.mean() / downside_returns.std())
            else:
                sortino_ratio = 0
            
            # Maximum Drawdown
            equity_df['cummax'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
            max_drawdown = equity_df['drawdown'].min() * 100
            
            # Calmar Ratio
            if max_drawdown != 0:
                calmar_ratio = total_return / abs(max_drawdown)
            else:
                calmar_ratio = 0
            
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            sharpe_ratio = 0
            sortino_ratio = 0
            max_drawdown = 0
            calmar_ratio = 0
        
        metrics = {
            'initial_capital': self.config.INITIAL_CAPITAL,
            'final_equity': final_equity,
            'total_return_pct': total_return,
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades) if len(trades_df) > 0 else 0,
            'losing_trades': len(losing_trades) if len(trades_df) > 0 else 0,
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown,
            'calmar_ratio': calmar_ratio
        }
        
        return metrics, trades_df, equity_df

# ============================================================
# VISUALIZATION
# ============================================================

def plot_equity_curve(equity_df, period_name, save_path=None):
    """Plot equity curve"""
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(equity_df['date'], equity_df['equity'], linewidth=2, color='#1f77b4')
    ax.axhline(y=BacktestConfig.INITIAL_CAPITAL, color='red', linestyle='--', 
               label='Initial Capital', alpha=0.7)
    
    ax.set_title(f'Equity Curve - {period_name}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    
    plt.close()

def plot_drawdown(equity_df, period_name, save_path=None):
    """Plot drawdown"""
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.fill_between(equity_df['date'], equity_df['drawdown'] * 100, 0, 
                     color='red', alpha=0.3)
    ax.plot(equity_df['date'], equity_df['drawdown'] * 100, 
            color='darkred', linewidth=1.5)
    
    ax.set_title(f'Drawdown - {period_name}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    
    plt.close()

def plot_trade_distribution(trades_df, period_name, save_path=None):
    """Plot trade P&L distribution"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # P&L histogram
    ax1.hist(trades_df['net_pnl'], bins=30, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Net P&L ($)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Cumulative P&L
    trades_df['cumulative_pnl'] = trades_df['net_pnl'].cumsum()
    ax2.plot(range(len(trades_df)), trades_df['cumulative_pnl'], 
             linewidth=2, color='#2ca02c')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_title('Cumulative P&L', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Trade Number', fontsize=12)
    ax2.set_ylabel('Cumulative P&L ($)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Trade Analysis - {period_name}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    
    plt.close()

def create_performance_summary(metrics, period_name, save_path=None):
    """Create performance summary table"""
    
    summary_text = f"""
{'='*60}
BACKTEST RESULTS - {period_name.upper()}
{'='*60}

CAPITAL
  Initial Capital:        ${metrics['initial_capital']:>15,.2f}
  Final Equity:           ${metrics['final_equity']:>15,.2f}
  Total Return:           {metrics['total_return_pct']:>15.2f}%

TRADE STATISTICS
  Total Trades:           {metrics['total_trades']:>15}
  Winning Trades:         {metrics['winning_trades']:>15}
  Losing Trades:          {metrics['losing_trades']:>15}
  Win Rate:               {metrics['win_rate_pct']:>15.2f}%
  
  Average Win:            ${metrics['avg_win']:>15,.2f}
  Average Loss:           ${metrics['avg_loss']:>15,.2f}

RISK METRICS
  Sharpe Ratio:           {metrics['sharpe_ratio']:>15.3f}
  Sortino Ratio:          {metrics['sortino_ratio']:>15.3f}
  Calmar Ratio:           {metrics['calmar_ratio']:>15.3f}
  Maximum Drawdown:       {metrics['max_drawdown_pct']:>15.2f}%

{'='*60}
"""
    
    print(summary_text)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(summary_text)
        print(f"   Saved: {save_path}")

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main backtesting pipeline"""
    
    print("="*60)
    print("PHASE 6: BACKTESTING NVDA TRADING STRATEGY")
    print("="*60)
    
    # Load model and data
    model = load_model()
    train_df, val_df, test_df = load_backtest_data()
    
    # Configuration
    config = BacktestConfig()
    
    print(f"\n  Backtest Configuration:")
    print(f"  Initial Capital: ${config.INITIAL_CAPITAL:,.2f}")
    print(f"  Commission Rate: {config.COMMISSION_RATE*100:.3f}%")
    print(f"  Borrow Rate: {config.BORROW_RATE_ANNUAL*100:.3f}% (annual)")
    print(f"  Position Size: {config.POSITION_SIZE*100:.0f}%")
    
    # Backtest each period
    periods = [
        ('Train', train_df),
        ('Validation', val_df),
        ('Test', test_df)
    ]
    
    all_results = {}
    
    for period_name, df in periods:
        print(f"\n{'#'*60}")
        print(f"BACKTESTING: {period_name.upper()} PERIOD")
        print(f"{'#'*60}")
        
        # Generate signals
        df_with_signals = generate_signals(model, df, config.SEQUENCE_LENGTH)
        
        # Run backtest
        engine = BacktestEngine(config)
        metrics, trades_df, equity_df = engine.run_backtest(df_with_signals)
        
        # Store results
        all_results[period_name] = {
            'metrics': metrics,
            'trades': trades_df,
            'equity': equity_df
        }
        
        # Create visualizations
        import os
        os.makedirs('backtesting/results', exist_ok=True)
        
        plot_equity_curve(
            equity_df, 
            period_name,
            f'backtesting/results/{period_name.lower()}_equity_curve.png'
        )
        
        plot_drawdown(
            equity_df,
            period_name,
            f'backtesting/results/{period_name.lower()}_drawdown.png'
        )
        
        if len(trades_df) > 0:
            plot_trade_distribution(
                trades_df,
                period_name,
                f'backtesting/results/{period_name.lower()}_trades.png'
            )
        
        create_performance_summary(
            metrics,
            period_name,
            f'backtesting/results/{period_name.lower()}_summary.txt'
        )
    
    # Comparison summary
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    comparison_df = pd.DataFrame({
        period: results['metrics'] 
        for period, results in all_results.items()
    }).T
    
    print(comparison_df[[
        'total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 
        'win_rate_pct', 'total_trades'
    ]].to_string())
    
    # Save comparison
    comparison_df.to_csv('backtesting/results/performance_comparison.csv')
    print(f"\n💾 Saved: backtesting/results/performance_comparison.csv")
    
    print(f"\n{'='*60}")
    print("✅ PHASE 6 COMPLETE!")
    print(f"{'='*60}")
    print("\nResults saved to: backtesting/results/")
    print("\nNext steps:")
    print("1. Review equity curves and performance metrics")
    print("2. Analyze trade statistics")
    print("3. Move to Phase 7: Executive Report")

if __name__ == "__main__":
    main()
