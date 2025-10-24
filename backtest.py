"""
Backtesting Script

Evaluate trained PPO agent on test data with realistic market conditions.
"""

import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
import sys
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))

from models.env.forex_env import ForexTradingEnv
from utils.logger import logger, setup_logger


def load_test_data(config: dict, symbol: str) -> pd.DataFrame:
    """Load test data for a symbol."""
    data_dir = Path(config['data']['processed_data_dir'])
    timeframe = config['trading']['timeframe']
    symbol_clean = symbol.replace('.', '')
    filename = f"{symbol_clean}_{timeframe}_test.parquet"

    df = pd.read_parquet(data_dir / filename)
    return df


def run_backtest(model_path: str, config_path: str, symbol: str, render: bool = False):
    """Run backtest on a trained model."""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load test data
    logger.info(f"Loading test data for {symbol}...")
    test_data = load_test_data(config, symbol)
    logger.info(f"Loaded {len(test_data):,} bars")

    # Create environment
    logger.info("Creating test environment...")
    env = ForexTradingEnv(
        data=test_data,
        config_path=config_path,
        initial_balance=100_000.0,
        render_mode='human' if render else None,
    )

    # Load model
    logger.info(f"Loading model from {model_path}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PPO.load(model_path, device=device)
    logger.info(f"Model loaded on {device}")

    # Run episode
    logger.info("\n" + "="*80)
    logger.info("Starting Backtest")
    logger.info("="*80)

    obs, info = env.reset()
    done = False
    step = 0

    equity_curve = [env.initial_balance]
    balance_curve = [env.initial_balance]

    while not done:
        # Get action from model
        action, _states = model.predict(obs, deterministic=True)

        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Track metrics
        equity_curve.append(info['equity'])
        balance_curve.append(info['balance'])

        if render and step % 100 == 0:
            env.render()

        step += 1

    # Results
    logger.info("\n" + "="*80)
    logger.info("Backtest Results")
    logger.info("="*80)

    final_balance = info['balance']
    final_equity = info['equity']
    total_profit = info['total_profit']
    n_trades = info['n_closed_trades']
    max_dd = info['max_drawdown']

    # Calculate metrics
    initial_balance = env.initial_balance
    total_return = (final_balance - initial_balance) / initial_balance * 100

    # Count wins/losses
    wins = len([p for p in env.closed_positions if p['profit'] > 0])
    losses = n_trades - wins
    win_rate = wins / n_trades * 100 if n_trades > 0 else 0

    # Average win/loss
    if wins > 0:
        avg_win = np.mean([p['profit'] for p in env.closed_positions if p['profit'] > 0])
    else:
        avg_win = 0

    if losses > 0:
        avg_loss = np.mean([p['profit'] for p in env.closed_positions if p['profit'] <= 0])
    else:
        avg_loss = 0

    # Profit factor
    total_wins = sum([p['profit'] for p in env.closed_positions if p['profit'] > 0])
    total_losses = abs(sum([p['profit'] for p in env.closed_positions if p['profit'] <= 0]))
    profit_factor = total_wins / total_losses if total_losses > 0 else np.inf

    # Sharpe ratio (simplified)
    equity_returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = np.mean(equity_returns) / np.std(equity_returns) * np.sqrt(252) if len(equity_returns) > 0 else 0

    # Display results
    logger.info(f"Initial Balance:    ${initial_balance:,.2f}")
    logger.info(f"Final Balance:      ${final_balance:,.2f}")
    logger.info(f"Final Equity:       ${final_equity:,.2f}")
    logger.info(f"Total Profit:       ${total_profit:,.2f}")
    logger.info(f"Total Return:       {total_return:.2f}%")
    logger.info(f"Max Drawdown:       {max_dd*100:.2f}%")
    logger.info(f"\nTrades:             {n_trades}")
    logger.info(f"Wins:               {wins}")
    logger.info(f"Losses:             {losses}")
    logger.info(f"Win Rate:           {win_rate:.2f}%")
    logger.info(f"Avg Win:            ${avg_win:.2f}")
    logger.info(f"Avg Loss:           ${avg_loss:.2f}")
    logger.info(f"Profit Factor:      {profit_factor:.2f}")
    logger.info(f"Sharpe Ratio:       {sharpe:.2f}")

    # Plot equity curve
    plot_results(equity_curve, balance_curve, env.closed_positions)

    return {
        'final_balance': final_balance,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
    }


def plot_results(equity_curve, balance_curve, trades):
    """Plot backtest results."""

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Equity curve
    axes[0].plot(equity_curve, label='Equity', linewidth=2)
    axes[0].plot(balance_curve, label='Balance', linewidth=2, alpha=0.7)
    axes[0].set_title('Equity Curve', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Value ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Trade P&L
    trade_profits = [t['profit'] for t in trades]
    colors = ['green' if p > 0 else 'red' for p in trade_profits]
    axes[1].bar(range(len(trade_profits)), trade_profits, color=colors, alpha=0.6)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_title('Trade P&L', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Trade Number')
    axes[1].set_ylabel('Profit ($)')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150)
    logger.info("\nEquity curve saved to backtest_results.png")
    plt.close()


def main():
    """Main backtest script."""
    import argparse

    parser = argparse.ArgumentParser(description="Backtest trained PPO model")
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--symbol', default='EURUSD.', help='Symbol to test')
    parser.add_argument('--render', action='store_true', help='Render during backtest')
    args = parser.parse_args()

    # Setup logger
    setup_logger(name="TradingEngine", level="INFO", log_file="logs/backtest.log")

    # Run backtest
    results = run_backtest(
        model_path=args.model,
        config_path=args.config,
        symbol=args.symbol,
        render=args.render,
    )

    logger.info("\nBacktest complete!")


if __name__ == '__main__':
    main()
