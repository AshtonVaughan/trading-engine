"""
FOREX Trading Gymnasium Environment

Realistic market simulation with:
- Spread, slippage, commission
- Dynamic position sizing
- Multiple concurrent positions
- Aggressive reward function for max profit
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import yaml


class ForexTradingEnv(gym.Env):
    """
    Gymnasium environment for FOREX trading with RL.

    State: Last N bars with all features
    Action: [action_type, position_size, stop_loss_pips, take_profit_pips]
    Reward: Aggressive profit-focused with light risk penalties
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        data: pd.DataFrame,
        config_path: str = "config/config.yaml",
        initial_balance: float = 100000.0,
        render_mode: Optional[str] = None
    ):
        super().__init__()

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.render_mode = render_mode

        # Extract config
        self.lookback = self.config['features']['lookback_bars']
        self.reward_config = self.config['reward']
        self.risk_config = self.config['risk']

        # Spread configuration (from backtest config)
        symbol_name = self.config['trading']['pairs'][0].replace('.', '')
        self.spread_pips = self.config['backtest']['spread_pips'].get(symbol_name[:6], 1.0)
        self.slippage_pips = self.config['backtest'].get('expected_slippage_pips', 0.5)

        # Feature columns (exclude OHLC time columns)
        self.feature_cols = [c for c in self.data.columns
                            if c not in ['time', 'open', 'high', 'low', 'close',
                                       'tick_volume', 'spread', 'real_volume']]
        self.n_features = len(self.feature_cols)

        # Action space: [discrete_action, position_size, stop_loss, take_profit]
        # Discrete: 0=BUY, 1=SELL, 2=CLOSE_ALL, 3=HOLD
        self.action_space = spaces.Box(
            low=np.array([0, 0.01, 5, 10], dtype=np.float32),
            high=np.array([3, 10.0, 50, 150], dtype=np.float32),
            shape=(4,),
            dtype=np.float32
        )

        # Observation space: lookback bars × features + account state
        account_state_size = 10  # balance, equity, margin, positions, etc.
        obs_size = self.lookback * self.n_features + account_state_size
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )

        # Trading state
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.positions = []  # List of open positions
        self.closed_positions = []  # Track closed positions for stats

        self.current_step = self.lookback  # Start after lookback period
        self.total_profit = 0.0
        self.max_equity = self.initial_balance
        self.max_drawdown = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.trades_today = 0
        self.daily_profit = 0.0

        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""

        # Parse action
        action_type = int(action[0])  # 0=BUY, 1=SELL, 2=CLOSE_ALL, 3=HOLD
        position_size = float(action[1])  # Lot size
        stop_loss_pips = float(action[2])
        take_profit_pips = float(action[3])

        # Get current price
        current_bar = self.data.iloc[self.current_step]
        current_price = current_bar['close']

        # Update open positions (check SL/TP)
        reward_from_positions = self._update_positions(current_bar)

        # Execute action
        reward_from_action = 0.0

        # Risk management checks
        if action_type in [0, 1]:  # BUY or SELL
            # Check daily limits
            if self.trades_today >= self.risk_config['max_daily_trades']:
                action_type = 3  # Force HOLD
            elif len(self.positions) >= self.risk_config['max_concurrent_trades']:
                action_type = 3  # Force HOLD
            elif self.daily_profit <= -self.balance * (self.risk_config['daily_loss_limit_pct'] / 100):
                action_type = 3  # Stop trading after daily loss limit
            else:
                # Open new position
                reward_from_action = self._open_position(
                    action_type, position_size, stop_loss_pips, take_profit_pips, current_bar
                )

        elif action_type == 2:  # CLOSE_ALL
            reward_from_action = self._close_all_positions(current_bar)

        # Total reward
        reward = reward_from_positions + reward_from_action

        # Apply aggressive reward shaping
        reward = self._shape_reward(reward)

        # Update equity
        self._update_equity(current_price)

        # Check termination conditions
        terminated = False
        truncated = False

        # Max drawdown stop
        drawdown = (self.max_equity - self.equity) / self.max_equity
        if drawdown >= self.risk_config['max_drawdown_pct'] / 100:
            terminated = True
            reward -= 100  # Large penalty for blowing up account

        # Account blown
        if self.equity <= self.initial_balance * 0.1:
            terminated = True
            reward -= 200

        # Move to next step
        self.current_step += 1

        # Check if end of data
        if self.current_step >= len(self.data) - 1:
            truncated = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _open_position(
        self, action_type: int, lot_size: float, sl_pips: float, tp_pips: float, bar: pd.Series
    ) -> float:
        """Open a new position."""

        # Clamp lot size to risk limits
        lot_size = np.clip(lot_size,
                          self.risk_config['min_lot_size'],
                          self.risk_config['max_lot_size'])

        price = bar['close']

        # Apply spread
        if action_type == 0:  # BUY
            entry_price = price + (self.spread_pips * 0.0001) + (self.slippage_pips * 0.0001)
            stop_loss = entry_price - (sl_pips * 0.0001)
            take_profit = entry_price + (tp_pips * 0.0001)
        else:  # SELL
            entry_price = price - (self.spread_pips * 0.0001) - (self.slippage_pips * 0.0001)
            stop_loss = entry_price + (sl_pips * 0.0001)
            take_profit = entry_price - (tp_pips * 0.0001)

        # Create position
        position = {
            'type': 'BUY' if action_type == 0 else 'SELL',
            'entry_price': entry_price,
            'lot_size': lot_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_step': self.current_step,
            'highest_profit': 0.0,  # For trailing stop
        }

        self.positions.append(position)
        self.trades_today += 1

        return -0.01  # Small penalty for opening position (holding cost)

    def _close_position(self, position: dict, exit_price: float, reason: str) -> float:
        """Close a position and calculate profit."""

        # Calculate profit/loss
        if position['type'] == 'BUY':
            profit_pips = (exit_price - position['entry_price']) / 0.0001
        else:  # SELL
            profit_pips = (position['entry_price'] - exit_price) / 0.0001

        # Convert pips to dollars (rough estimate: $10 per pip per lot)
        profit_dollars = profit_pips * 10 * position['lot_size']

        # Update balance
        self.balance += profit_dollars
        self.total_profit += profit_dollars
        self.daily_profit += profit_dollars

        # Track position
        position['exit_price'] = exit_price
        position['profit'] = profit_dollars
        position['profit_pips'] = profit_pips
        position['exit_reason'] = reason
        position['exit_step'] = self.current_step
        self.closed_positions.append(position)

        # Update win streak
        if profit_dollars > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        return profit_dollars

    def _update_positions(self, bar: pd.Series) -> float:
        """Update all open positions, check SL/TP."""

        total_reward = 0.0
        current_price = bar['close']
        positions_to_close = []

        for i, pos in enumerate(self.positions):
            # Check stop loss
            if pos['type'] == 'BUY':
                if current_price <= pos['stop_loss']:
                    profit = self._close_position(pos, pos['stop_loss'], 'SL')
                    total_reward += profit
                    positions_to_close.append(i)
                    continue
                elif current_price >= pos['take_profit']:
                    profit = self._close_position(pos, pos['take_profit'], 'TP')
                    total_reward += profit
                    positions_to_close.append(i)
                    continue
            else:  # SELL
                if current_price >= pos['stop_loss']:
                    profit = self._close_position(pos, pos['stop_loss'], 'SL')
                    total_reward += profit
                    positions_to_close.append(i)
                    continue
                elif current_price <= pos['take_profit']:
                    profit = self._close_position(pos, pos['take_profit'], 'TP')
                    total_reward += profit
                    positions_to_close.append(i)
                    continue

            # Update highest profit for trailing stop
            if pos['type'] == 'BUY':
                current_profit = (current_price - pos['entry_price']) / 0.0001
            else:
                current_profit = (pos['entry_price'] - current_price) / 0.0001

            pos['highest_profit'] = max(pos['highest_profit'], current_profit)

            # Trailing stop logic
            if self.risk_config['trailing_stop_enabled']:
                trigger_pips = self.risk_config['trailing_stop_trigger_pips']
                if pos['highest_profit'] >= trigger_pips:
                    # Move SL to breakeven + 50% of profit
                    if pos['type'] == 'BUY':
                        new_sl = pos['entry_price'] + (pos['highest_profit'] * 0.5 * 0.0001)
                        pos['stop_loss'] = max(pos['stop_loss'], new_sl)
                    else:
                        new_sl = pos['entry_price'] - (pos['highest_profit'] * 0.5 * 0.0001)
                        pos['stop_loss'] = min(pos['stop_loss'], new_sl)

            # Holding cost
            total_reward -= self.reward_config['holding_cost']

        # Remove closed positions
        for i in reversed(positions_to_close):
            self.positions.pop(i)

        return total_reward

    def _close_all_positions(self, bar: pd.Series) -> float:
        """Close all open positions."""
        total_reward = 0.0
        current_price = bar['close']

        for pos in self.positions:
            profit = self._close_position(pos, current_price, 'MANUAL')
            total_reward += profit

        self.positions = []
        return total_reward

    def _shape_reward(self, base_reward: float) -> float:
        """Apply aggressive reward shaping."""

        # Profit multiplier
        if base_reward > 0:
            reward = base_reward * self.reward_config['profit_multiplier']

            # Big win bonus
            if len(self.closed_positions) > 0:
                last_trade = self.closed_positions[-1]
                if last_trade['profit_pips'] >= self.reward_config['big_win_threshold_pips']:
                    reward += self.reward_config['big_win_bonus']

            # Win streak bonus
            if self.consecutive_wins >= 3:
                reward += self.reward_config['win_streak_bonus'] * self.consecutive_wins

        else:
            # Loss penalty
            reward = base_reward * self.reward_config['loss_multiplier']

            # Drawdown penalty (light)
            drawdown = (self.max_equity - self.equity) / self.max_equity
            reward -= drawdown * self.reward_config['drawdown_penalty']

        return reward

    def _update_equity(self, current_price: float):
        """Update equity based on open positions."""

        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        for pos in self.positions:
            if pos['type'] == 'BUY':
                profit_pips = (current_price - pos['entry_price']) / 0.0001
            else:
                profit_pips = (pos['entry_price'] - current_price) / 0.0001

            unrealized_pnl += profit_pips * 10 * pos['lot_size']

        self.equity = self.balance + unrealized_pnl
        self.max_equity = max(self.max_equity, self.equity)

        # Update max drawdown
        if self.max_equity > 0:
            drawdown = (self.max_equity - self.equity) / self.max_equity
            self.max_drawdown = max(self.max_drawdown, drawdown)

    def _get_observation(self) -> np.ndarray:
        """Get current observation (state)."""

        # Get last N bars of features
        start_idx = max(0, self.current_step - self.lookback)
        end_idx = self.current_step

        feature_data = self.data.iloc[start_idx:end_idx][self.feature_cols].values

        # Pad if necessary
        if feature_data.shape[0] < self.lookback:
            padding = np.zeros((self.lookback - feature_data.shape[0], self.n_features))
            feature_data = np.vstack([padding, feature_data])

        # Flatten
        features_flat = feature_data.flatten()

        # Account state
        account_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.equity / self.initial_balance,    # Normalized equity
            len(self.positions) / self.risk_config['max_concurrent_trades'],  # Position ratio
            self.max_drawdown,
            self.consecutive_wins / 10.0,  # Normalized
            self.consecutive_losses / 10.0,
            self.trades_today / self.risk_config['max_daily_trades'],
            self.daily_profit / self.initial_balance,
            float(len(self.positions) > 0),  # Has open positions
            self.total_profit / self.initial_balance,
        ], dtype=np.float32)

        # Concatenate
        observation = np.concatenate([features_flat, account_state])

        return observation.astype(np.float32)

    def _get_info(self) -> Dict:
        """Get environment info."""
        return {
            'balance': self.balance,
            'equity': self.equity,
            'total_profit': self.total_profit,
            'n_positions': len(self.positions),
            'n_closed_trades': len(self.closed_positions),
            'max_drawdown': self.max_drawdown,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
        }

    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            print(f"Step: {self.current_step} | Balance: ${self.balance:.2f} | "
                  f"Equity: ${self.equity:.2f} | Positions: {len(self.positions)} | "
                  f"Profit: ${self.total_profit:.2f}")
