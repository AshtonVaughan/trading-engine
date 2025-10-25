# TradingEngine - H1 Swing Trading DRL Bot

**Deep Reinforcement Learning FOREX trading bot optimized for H1 (hourly) swing trading with strict risk management.**

> ⚠️ **MEDIUM RISK STRATEGY:** This bot targets <5% drawdown with 100x leverage through conservative position sizing and H1 swing trading (1-4 hour holds). **ALWAYS test on demo account first!**

## Strategy Overview

- **Timeframe**: H1 (1-hour bars)
- **Trading Style**: Medium-term swing trading (1-4 hour holds)
- **Pairs**: EURUSD, GBPUSD, USDJPY
- **Leverage**: 100x (with conservative position sizing)
- **Target**: <5% maximum drawdown, 130-200% annual returns
- **Risk per Trade**: 0.5% (0.05 lot per $10K account)

## Features

- **Deep Reinforcement Learning:** PPO agent trained on 2 years of H1 data
- **H200 NVL Optimized:** 128 parallel environments, 50M parameter models
- **Conservative Risk Management:** <5% drawdown target with 100x leverage
- **MT5 Integration:** Real-time data collection and order execution (Windows only)
- **69 Features:** Price action, technical indicators, volatility measures
- **Curriculum Learning:** Progressive training from calm → volatile markets
- **GPU-Accelerated Backtesting:** Vectorized backtesting with realistic spread/slippage
- **Walk-Forward Validation:** Rolling 6-month train, 1-month test

## Architecture

```
State (100+ features, 60-bar lookback)
    ↓
[LSTM (1024×8) + Transformer (768d×12)]  ← Feature Extraction
    ↓
┌─────────────────┬─────────────────┐
│   Actor Network │  Critic Network │
│  (Policy π)     │  (Value V)      │
└─────────────────┴─────────────────┘
    ↓                     ↓
Actions:                 Value
- BUY/SELL/CLOSE/HOLD   Estimate
- Position size (0.01-10 lots)
- Stop loss (5-50 pips)
- Take profit (10-150 pips)
```

## Project Structure

```
TradingEngine/
├── config/
│   ├── config.yaml           # Main configuration
│   └── mt5_config.yaml       # MT5 credentials (don't commit!)
├── data/
│   ├── collectors/           # MT5 data collection
│   ├── features/             # Feature engineering (100+ features)
│   └── environment/          # Gym environment
├── models/
│   ├── rl_agent/             # PPO agent (actor/critic)
│   └── reward_shaping/       # Aggressive reward function
├── training/
│   ├── train_rl_h200.py      # H200-optimized training
│   ├── curriculum_learning.py
│   └── parallel_envs.py      # 128 parallel environments
├── backtesting/
│   ├── vectorized_backtest.py # GPU-accelerated backtesting
│   └── walk_forward.py
├── deployment/
│   ├── mt5_executor.py       # Live trading execution
│   └── monitoring/           # Performance tracking
└── utils/
    ├── logger.py
    └── mt5_bridge.py         # MT5 Python API wrapper
```

## Installation

### Cloud Training (H200 Linux)

```bash
git clone https://github.com/AshtonVaughan/TradingEngine.git
cd TradingEngine
pip install -r requirements-training.txt
```

**Note:** `requirements-training.txt` excludes Windows-only dependencies (MetaTrader5).

### Local Development (Windows)

#### 1. Prerequisites

- **Python 3.11+**
- **MetaTrader 5** (for data collection and live trading)
- **MT5 Demo/Live Account** (from any broker)

#### 2. Install Dependencies

```bash
cd TradingEngine
pip install -r requirements.txt
```

### 3. Configure MT5

Edit `config/mt5_config.yaml` with your MT5 credentials:

```yaml
mt5:
  login: YOUR_ACCOUNT_NUMBER
  password: "YOUR_PASSWORD"
  server: "YOUR_BROKER_SERVER"
  terminal_path:
    windows: "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
```

> **Security:** Never commit `mt5_config.yaml` to git! Add it to `.gitignore`.

### 4. Verify MT5 Connection

```bash
python utils/mt5_bridge.py
```

Expected output:
```
2024-01-15 10:30:00 | TradingEngine | INFO | MT5 connection successful!
2024-01-15 10:30:01 | TradingEngine | INFO | Sample data:
                       time    open    high     low   close  tick_volume
0  2024-01-15 10:00:00  1.0950  1.0955  1.0948  1.0952         1234
```

## Usage

### Training on H200 Cloud

The H1 data is already in the repository. Start training immediately:

```bash
# Full training with curriculum learning (100M steps)
python train_ppo.py --config config/config.yaml

# Train on specific symbol
python train_ppo.py --config config/config.yaml --symbol EURUSD

# Train without curriculum
python train_ppo.py --config config/config.yaml --no-curriculum

# Resume from checkpoint
python train_ppo.py --config config/config.yaml --resume checkpoints/ppo_calm_final.zip
```

**Monitor training:**
```bash
tensorboard --logdir runs
```

**Expected training time on H200 NVL:** 24-48 hours for 100M timesteps

Training features:
- 128 parallel environments
- bfloat16 mixed precision
- Curriculum learning (calm → normal → volatile → mixed)
- Checkpoints every 10M steps

### Data Collection (Windows Only - Optional)

If you need to update or collect more H1 data:

```bash
# 1. Update date range in config/config.yaml
# 2. Collect data from Yahoo Finance
python data/collectors/free_data_collector.py

# 3. Generate features
python data/features/feature_engineer.py

# 4. Commit to git
git add data/raw/*.parquet
git commit -m "Update H1 data"
git push
```

### Step 4: Backtesting

**TODO:** Create backtesting engine
```bash
python backtesting/vectorized_backtest.py --model checkpoints/best_model.pth
```

### Step 5: Walk-Forward Validation

**TODO:** Create walk-forward validation
```bash
python backtesting/walk_forward.py --model checkpoints/best_model.pth
```

### Step 6: Paper Trading

**TODO:** Create live trading system
```bash
python deployment/live_trader.py --model checkpoints/best_model.pth --paper-trading
```

### Step 7: Live Trading

> ⚠️ **Only after 1+ month of successful paper trading!**

```bash
python deployment/live_trader.py --model checkpoints/best_model.pth
```

## Configuration

### Risk Settings (`config/config.yaml`)

```yaml
risk:
  base_lot_per_10k: 1.0      # 1 lot per $10K account
  max_lot_size: 10.0         # Max 10 lots (10% account)
  max_concurrent_trades: 5   # Max 5 simultaneous trades
  daily_loss_limit_pct: 40.0 # Stop after -40% day
  max_drawdown_pct: 60.0     # Emergency shutdown
```

### Reward Function

```python
reward = (
    profit * 2.0              # Maximize profit
    - drawdown * 0.3          # Light penalty
    + big_win_bonus * 1.5     # Bonus for >40 pip wins
    - holding_cost * 0.01     # Quick scalps
    + win_streak * 0.5        # Consistency bonus
)
```

### H200 Training Settings

```yaml
h200:
  num_envs: 128              # Parallel environments
  use_bfloat16: true         # Native H200 precision
  replay_buffer_size: 10M    # GPU-resident buffer
  checkpoint_freq: 10M       # Save every 10M steps
```

## Expected Performance

### Conservative Estimate
- **Annual return:** 150-250%
- **Win rate:** 55-65%
- **Profit factor:** 1.5-2.0
- **Max drawdown:** 40-50%
- **Sharpe ratio:** 1.5-2.0

### Optimistic Estimate
- **Annual return:** 300-500%+
- **Win rate:** 60-70%
- **Profit factor:** 2.0-3.0
- **Max drawdown:** 50-60%
- **Sharpe ratio:** 2.0-3.0

## Risk Warnings

⚠️ **THIS IS AN EXTREMELY AGGRESSIVE STRATEGY:**

1. **10%+ risk per trade** can lead to rapid account depletion
2. **40-60% drawdowns** are expected during losing streaks
3. **Martingale strategies** (if enabled) can wipe out accounts in 3-5 trades
4. **RL agents can fail** catastrophically during regime changes
5. **Market conditions change** - model drift is inevitable
6. **Past performance ≠ future results**

### Mandatory Safety Measures

- ✅ **Test on demo for 1+ month minimum**
- ✅ **Start live with <5% of total capital**
- ✅ **Set hard daily loss limits (-40%)**
- ✅ **Monitor trades daily**
- ✅ **Be prepared to shut down**
- ✅ **Never risk money you can't afford to lose**

## Development Roadmap

### Phase 1: Foundation (Completed ✓)
- [x] Project structure
- [x] MT5 data collector
- [x] Configuration files
- [x] Logging utilities

### Phase 2: Data & Features (TODO)
- [ ] Feature engineering pipeline (100+ features)
- [ ] Data augmentation
- [ ] Train/val/test split

### Phase 3: RL Agent (TODO)
- [ ] Gym environment with market simulation
- [ ] Actor/critic networks (50M params)
- [ ] Aggressive reward function
- [ ] PPO training loop

### Phase 4: Training (TODO)
- [ ] H200-optimized training pipeline
- [ ] 128 parallel environments
- [ ] Curriculum learning
- [ ] Model checkpointing

### Phase 5: Validation (TODO)
- [ ] GPU-accelerated backtesting
- [ ] Walk-forward validation
- [ ] Monte Carlo stress testing

### Phase 6: Deployment (TODO)
- [ ] MT5 live execution
- [ ] Performance monitoring
- [ ] Telegram alerts
- [ ] Emergency shutdown system

## Tech Stack

- **Python 3.11+**
- **PyTorch 2.1+** (RL agent)
- **Stable-Baselines3** (PPO implementation)
- **Gymnasium** (RL environment)
- **MetaTrader5 Python API**
- **Pandas, NumPy** (data processing)
- **TA-Lib** (technical indicators)
- **Optuna** (hyperparameter tuning)
- **TensorBoard** (training visualization)

## Contributing

This is a personal trading bot. If you want to contribute:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - Use at your own risk

## Disclaimer

This software is for educational purposes only. Trading FOREX is extremely risky and most traders lose money. The author is not responsible for any financial losses incurred from using this software.

**YOU HAVE BEEN WARNED.**

---

Built with ❤️ and a lot of GPU power.

**Status:** 🟡 Under Development (Phase 1 complete, Phase 2-6 in progress)
