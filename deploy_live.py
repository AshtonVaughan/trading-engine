"""
Live Trading Deployment

Deploy trained PPO agent to MT5 for live trading.
Includes safety checks, paper trading mode, and monitoring.
"""

import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
import time
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent))

from utils.mt5_bridge import MT5Bridge
from utils.logger import logger, setup_logger


class LiveTrader:
    """
    Live trading system with trained RL agent.
    """

    def __init__(self, model_path: str, config_path: str, mt5_config_path: str):
        """Initialize live trader."""

        # Load configs
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        with open(mt5_config_path, 'r') as f:
            self.mt5_config = yaml.safe_load(f)

        self.live_config = self.config['live']
        self.paper_trading = self.live_config['paper_trading']

        # Connect to MT5
        logger.info("Connecting to MT5...")
        self.mt5 = MT5Bridge(mt5_config_path)
        if not self.mt5.connect():
            raise RuntimeError("Failed to connect to MT5")

        logger.info(f"Connected to MT5 (Account: {self.mt5_config['mt5']['login']})")

        # Load model
        logger.info(f"Loading model from {model_path}...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = PPO.load(model_path, device=device)
        logger.info(f"Model loaded on {device}")

        # Trading state
        self.symbols = self.config['trading']['pairs']
        self.magic_number = self.live_config['mt5']['magic_number']
        self.positions = {}  # Track our positions
        self.last_trade_time = {}

        # Feature engineering (need to replicate from data pipeline)
        from data.features.feature_engineer import FeatureEngineer
        self.feature_engineer = FeatureEngineer(config_path=config_path)

        # Risk limits
        self.risk_config = self.config['risk']
        self.trades_today = 0
        self.daily_profit = 0.0
        self.start_balance = self.mt5.get_balance()

    def get_current_state(self, symbol: str) -> np.ndarray:
        """Get current market state for a symbol."""

        # Get recent bars
        lookback = self.config['features']['lookback_bars']
        timeframe = self.config['trading']['timeframe']

        bars = self.mt5.get_recent_bars(symbol, timeframe, lookback + 200)  # Extra for indicators

        if bars is None or len(bars) < lookback + 50:
            logger.warning(f"Insufficient data for {symbol}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(bars)

        # Add features
        df = self.feature_engineer.add_all_features(df)

        # Get last lookback bars
        recent_data = df.tail(lookback)

        # Extract features (exclude OHLC time columns)
        feature_cols = [c for c in recent_data.columns
                       if c not in ['time', 'open', 'high', 'low', 'close',
                                  'tick_volume', 'spread', 'real_volume']]

        features = recent_data[feature_cols].values.flatten()

        # Add account state
        balance = self.mt5.get_balance()
        equity = self.mt5.get_equity()
        n_positions = len(self.positions)

        account_state = np.array([
            balance / self.start_balance,
            equity / self.start_balance,
            n_positions / self.risk_config['max_concurrent_trades'],
            0.0,  # max_drawdown (calculate separately)
            0.0,  # consecutive_wins
            0.0,  # consecutive_losses
            self.trades_today / self.risk_config['max_daily_trades'],
            self.daily_profit / self.start_balance,
            float(n_positions > 0),
            (balance - self.start_balance) / self.start_balance,
        ], dtype=np.float32)

        # Concatenate
        state = np.concatenate([features, account_state]).astype(np.float32)

        return state

    def execute_action(self, symbol: str, action: np.ndarray, current_price: float):
        """Execute trading action."""

        action_type = int(action[0])
        lot_size = float(action[1])
        sl_pips = float(action[2])
        tp_pips = float(action[3])

        # Risk checks
        if self.trades_today >= self.risk_config['max_daily_trades']:
            logger.info(f"Daily trade limit reached ({self.trades_today})")
            return

        if len(self.positions) >= self.risk_config['max_concurrent_trades']:
            logger.info(f"Max concurrent positions reached ({len(self.positions)})")
            return

        # Daily loss limit
        daily_loss_pct = (self.daily_profit / self.start_balance) * 100
        if daily_loss_pct <= -self.risk_config['daily_loss_limit_pct']:
            logger.warning(f"Daily loss limit hit: {daily_loss_pct:.2f}%")
            return

        # Execute action
        if action_type == 0:  # BUY
            logger.info(f"BUY {symbol}: {lot_size:.2f} lots, SL={sl_pips:.0f}, TP={tp_pips:.0f}")

            if self.paper_trading:
                logger.info("  [PAPER TRADE - Not executed]")
            else:
                # Calculate SL/TP prices
                sl_price = current_price - (sl_pips * 0.0001)
                tp_price = current_price + (tp_pips * 0.0001)

                # Execute order
                result = self.mt5.execute_market_order(
                    symbol=symbol,
                    order_type='buy',
                    lot_size=lot_size,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    magic=self.magic_number,
                )

                if result:
                    self.positions[symbol] = result
                    self.trades_today += 1
                    logger.info(f"  Order executed: {result}")
                else:
                    logger.error(f"  Order failed")

        elif action_type == 1:  # SELL
            logger.info(f"SELL {symbol}: {lot_size:.2f} lots, SL={sl_pips:.0f}, TP={tp_pips:.0f}")

            if self.paper_trading:
                logger.info("  [PAPER TRADE - Not executed]")
            else:
                sl_price = current_price + (sl_pips * 0.0001)
                tp_price = current_price - (tp_pips * 0.0001)

                result = self.mt5.execute_market_order(
                    symbol=symbol,
                    order_type='sell',
                    lot_size=lot_size,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    magic=self.magic_number,
                )

                if result:
                    self.positions[symbol] = result
                    self.trades_today += 1
                    logger.info(f"  Order executed: {result}")
                else:
                    logger.error(f"  Order failed")

        elif action_type == 2:  # CLOSE ALL
            logger.info(f"CLOSE ALL positions for {symbol}")

            if self.paper_trading:
                logger.info("  [PAPER TRADE - Not executed]")
            else:
                if symbol in self.positions:
                    result = self.mt5.close_position(self.positions[symbol])
                    if result:
                        del self.positions[symbol]
                        logger.info(f"  Position closed")
                    else:
                        logger.error(f"  Failed to close position")

        elif action_type == 3:  # HOLD
            pass  # Do nothing

    def trading_loop(self):
        """Main trading loop."""

        logger.info("\n" + "="*80)
        logger.info("Live Trading Started")
        if self.paper_trading:
            logger.info("MODE: PAPER TRADING (No real orders)")
        else:
            logger.info("MODE: LIVE TRADING (Real money at risk!)")
        logger.info("="*80)

        while True:
            try:
                # Check each symbol
                for symbol in self.symbols:
                    # Get current state
                    state = self.get_current_state(symbol)

                    if state is None:
                        continue

                    # Get action from model
                    action, _states = self.model.predict(state, deterministic=True)

                    # Get current price
                    current_price = self.mt5.get_current_price(symbol)

                    if current_price is None:
                        logger.warning(f"Could not get price for {symbol}")
                        continue

                    # Execute action
                    self.execute_action(symbol, action, current_price)

                # Update account metrics
                current_balance = self.mt5.get_balance()
                self.daily_profit = current_balance - self.start_balance

                # Log status
                if int(time.time()) % 60 == 0:  # Every minute
                    logger.info(f"\nStatus: Balance=${current_balance:,.2f} | "
                              f"Daily P&L=${self.daily_profit:,.2f} | "
                              f"Positions={len(self.positions)} | "
                              f"Trades today={self.trades_today}")

                # Sleep (check every bar)
                timeframe = self.config['trading']['timeframe']
                if timeframe == 'M1':
                    time.sleep(60)
                elif timeframe == 'H1':
                    time.sleep(3600)
                else:
                    time.sleep(60)

            except KeyboardInterrupt:
                logger.info("\nShutting down...")
                break

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(60)

        # Disconnect
        self.mt5.disconnect()
        logger.info("Trading stopped")


def main():
    """Main deployment script."""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy trained model for live trading")
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--mt5-config', default='config/mt5_config.yaml', help='Path to MT5 config')
    parser.add_argument('--live', action='store_true', help='Enable live trading (default: paper trading)')
    args = parser.parse_args()

    # Setup logger
    setup_logger(name="TradingEngine", level="INFO", log_file="logs/live_trading.log")

    # Warning for live trading
    if args.live:
        logger.warning("\n" + "="*80)
        logger.warning("LIVE TRADING MODE ENABLED")
        logger.warning("Real money is at risk!")
        logger.warning("Press Ctrl+C within 10 seconds to cancel...")
        logger.warning("="*80)
        time.sleep(10)

    # Create trader
    trader = LiveTrader(
        model_path=args.model,
        config_path=args.config,
        mt5_config_path=args.mt5_config,
    )

    # Override paper trading if --live flag used
    if args.live:
        trader.paper_trading = False

    # Start trading
    trader.trading_loop()


if __name__ == '__main__':
    main()
