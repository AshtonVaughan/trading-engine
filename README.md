<div align="center">

# TradingEngine

**Deep Reinforcement Learning forex bot for H1 swing trading**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-94A3B8?style=flat-square)](.)

*PPO agent trained on 2 years of H1 FOREX data. LSTM + Transformer feature extraction, curriculum learning, GPU-optimized training.*

</div>

<br>

## Strategy

| Parameter       | Value                              |
|-----------------|------------------------------------|
| Timeframe       | H1 (1-hour bars)                   |
| Style           | Swing trading, 1-4 hour holds      |
| Pairs           | EURUSD, GBPUSD, USDJPY             |
| Risk per trade  | 0.5% (0.05 lot per $10K)           |
| Target drawdown | <5%                                |
| Target return   | 130-200% annually                  |

<br>

## Architecture

```
State (100+ features, 60-bar lookback)
        |
[LSTM (1024x8) + Transformer (768d x12)]   <- feature extraction
        |
    +---------+----------+
    | Actor   |  Critic  |
    | (pi)    |  (V)     |
    +---------+----------+
        |            |
    Actions       Value estimate
    BUY/SELL/CLOSE/HOLD
    position size, SL, TP
```

**Features:** 100+ inputs including price action, 69 technical indicators, volatility measures  
**Training:** 128 parallel environments, bfloat16 mixed precision, curriculum learning (calm -> normal -> volatile)  
**Validation:** Rolling 6-month train, 1-month walk-forward test

<br>

## Installation

### Cloud training (Linux, H200)

```bash
git clone https://github.com/AshtonVaughan/TradingEngine.git
cd TradingEngine
pip install -r requirements-training.txt
```

### Local development (Windows, with MT5)

```bash
pip install -r requirements.txt
```

Configure `config/mt5_config.yaml` with your broker credentials (never commit this file).

<br>

## Usage

### Train

```bash
# Full training with curriculum learning (100M steps)
python train_ppo.py --config config/config.yaml

# Resume from checkpoint
python train_ppo.py --config config/config.yaml --resume checkpoints/ppo_calm_final.zip
```

Monitor with TensorBoard: `tensorboard --logdir runs`

Expected training time on H200 NVL: 24-48 hours for 100M timesteps.

### Collect data (Windows only, optional)

H1 data is already included in the repo. To update:

```bash
python data/collectors/free_data_collector.py
python data/features/feature_engineer.py
```

<br>

## Project Structure

```
config/           Main config, MT5 credentials
data/             Collectors, feature engineering, Gym environment
models/           PPO actor/critic, reward shaping
training/         H200-optimized training, curriculum, parallel envs
backtesting/      Vectorized GPU backtesting, walk-forward validation
deployment/       MT5 live execution, monitoring
utils/            Logger, MT5 bridge
```

<br>

## Risk Configuration

```yaml
risk:
  base_lot_per_10k: 1.0      # 1 lot per $10K
  max_lot_size: 10.0
  max_concurrent_trades: 5
  daily_loss_limit_pct: 40.0
  max_drawdown_pct: 60.0
```

<br>

## Disclaimer

This project is experimental and for educational purposes only. FOREX trading carries significant risk - most traders lose money. Always test on a demo account before any live use. The author accepts no liability for financial losses arising from use of this software.

<br>

## License

MIT
