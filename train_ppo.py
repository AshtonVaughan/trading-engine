"""
PPO Training Script for TradingEngine

Features:
- 128 parallel environments (H200 optimized)
- Curriculum learning (calm → volatile markets)
- Large Transformer-LSTM networks (~50M params)
- GPU-resident replay buffer
- bfloat16 precision
- TensorBoard logging
"""

import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import sys

sys.path.append(str(Path(__file__).parent))

from models.env.forex_env import ForexTradingEnv
from models.networks.transformer_policy import TransformerActorCriticPolicy, count_parameters
from utils.logger import logger, setup_logger


def load_data(config: dict, split: str = 'train') -> dict:
    """Load processed data for all symbols."""
    data_dir = Path(config['data']['processed_data_dir'])
    timeframe = config['trading']['timeframe']
    symbols = config['trading']['pairs']

    all_data = {}
    for symbol in symbols:
        symbol_clean = symbol.replace('.', '')
        filename = f"{symbol_clean}_{timeframe}_{split}.parquet"
        filepath = data_dir / filename

        if filepath.exists():
            df = pd.read_parquet(filepath)
            all_data[symbol] = df
            logger.info(f"Loaded {symbol} {split}: {len(df):,} bars")
        else:
            logger.warning(f"File not found: {filepath}")

    return all_data


def make_env(data: pd.DataFrame, config_path: str, rank: int = 0):
    """Create a single environment."""
    def _init():
        env = ForexTradingEnv(data=data, config_path=config_path)
        env = Monitor(env)
        return env
    return _init


def create_vec_env(data: pd.DataFrame, config: dict, config_path: str, n_envs: int = 128):
    """Create vectorized environment with multiple parallel instances."""
    logger.info(f"Creating {n_envs} parallel environments...")

    # Create multiple environments
    env_fns = [make_env(data, config_path, i) for i in range(n_envs)]

    # Use SubprocVecEnv for true parallelization
    vec_env = SubprocVecEnv(env_fns)

    # Normalize observations and rewards
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    logger.info(f"Vectorized environment created with {n_envs} parallel workers")

    return vec_env


def train_curriculum_stage(
    model: PPO,
    vec_env,
    stage_name: str,
    total_timesteps: int,
    checkpoint_dir: Path,
    log_dir: Path,
):
    """Train a single curriculum stage."""

    logger.info("\n" + "="*80)
    logger.info(f"Training Stage: {stage_name}")
    logger.info(f"Total timesteps: {total_timesteps:,}")
    logger.info("="*80)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000_000 // vec_env.num_envs,  # Save every 10M steps
        save_path=str(checkpoint_dir / stage_name),
        name_prefix=f"ppo_{stage_name}",
        save_replay_buffer=False,  # Too large for disk
        save_vecnormalize=True,
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback],
        log_interval=100,
        progress_bar=True,
    )

    # Save final model
    model.save(checkpoint_dir / f"ppo_{stage_name}_final")
    logger.info(f"Stage {stage_name} complete!")

    return model


def main():
    """Main training loop with curriculum learning."""
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO agent for FOREX trading")
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--symbol', default=None, help='Train on specific symbol (default: all)')
    parser.add_argument('--no-curriculum', action='store_true', help='Skip curriculum, train on all data')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    # Setup
    setup_logger(name="TradingEngine", level="INFO", log_file="logs/training.log")

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    if device == 'cuda':
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load training data
    logger.info("\nLoading training data...")
    train_data = load_data(config, split='train')

    if not train_data:
        logger.error("No training data found!")
        return

    # Use first symbol for simplicity (or combine all)
    if args.symbol and args.symbol in train_data:
        data = train_data[args.symbol]
        logger.info(f"Training on {args.symbol}: {len(data):,} bars")
    else:
        # Use first available symbol
        symbol = list(train_data.keys())[0]
        data = train_data[symbol]
        logger.info(f"Training on {symbol}: {len(data):,} bars")

    # Create directories
    checkpoint_dir = Path(config['paths']['checkpoints'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(config['paths']['logs'])
    log_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = Path(config['logging']['tensorboard_dir'])
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    # H200 configuration
    h200_config = config['h200']
    n_envs = h200_config['num_envs']  # 128 parallel environments

    # Create vectorized environment
    vec_env = create_vec_env(data, config, args.config, n_envs=n_envs)

    # PPO configuration
    ppo_config = config['rl']['ppo']
    network_config = config['rl']['network']

    # Policy kwargs
    policy_kwargs = {
        'net_arch': dict(pi=[], vf=[]),  # We use custom extractor
        'lstm_hidden': network_config['actor']['lstm_hidden'],
        'lstm_layers': network_config['actor']['lstm_layers'],
        'transformer_heads': network_config['actor']['transformer_heads'],
        'transformer_layers': network_config['actor']['transformer_layers'],
        'd_model': network_config['actor']['d_model'],
        'dropout': network_config['actor']['dropout'],
        'dual_critics': network_config['critic']['dual_critics'],
        'critic_hidden_size': network_config['critic']['hidden_size'],
        'critic_num_layers': network_config['critic']['num_layers'],
    }

    # Create or load model
    if args.resume:
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        model = PPO.load(
            args.resume,
            env=vec_env,
            device=device,
        )
    else:
        logger.info("\nCreating PPO model...")
        model = PPO(
            policy=TransformerActorCriticPolicy,
            env=vec_env,
            learning_rate=ppo_config['learning_rate'],
            n_steps=ppo_config['n_steps'],
            batch_size=ppo_config['batch_size'],
            n_epochs=ppo_config['n_epochs'],
            gamma=ppo_config['gamma'],
            gae_lambda=ppo_config['gae_lambda'],
            clip_range=ppo_config['clip_range'],
            clip_range_vf=ppo_config.get('clip_range_vf'),
            ent_coef=ppo_config['ent_coef'],
            vf_coef=ppo_config['vf_coef'],
            max_grad_norm=ppo_config['max_grad_norm'],
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(tensorboard_dir),
            device=device,
            verbose=1,
        )

    # Log model size
    n_params = count_parameters(model.policy)
    logger.info(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Curriculum learning stages
    if not args.no_curriculum and config['training']['curriculum']['enabled']:
        stages = config['training']['curriculum']['stages']

        for stage in stages:
            stage_name = stage['name']
            duration = stage['duration_steps']

            # TODO: Filter data by volatility percentile
            # For now, just train on all data

            model = train_curriculum_stage(
                model=model,
                vec_env=vec_env,
                stage_name=stage_name,
                total_timesteps=duration,
                checkpoint_dir=checkpoint_dir,
                log_dir=log_dir,
            )

    else:
        # Train without curriculum
        total_timesteps = config['training']['total_timesteps']

        logger.info("\n" + "="*80)
        logger.info("Training without curriculum")
        logger.info(f"Total timesteps: {total_timesteps:,}")
        logger.info("="*80)

        checkpoint_callback = CheckpointCallback(
            save_freq=10_000_000 // n_envs,
            save_path=str(checkpoint_dir),
            name_prefix="ppo",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback],
            log_interval=100,
            progress_bar=True,
        )

        # Save final model
        model.save(checkpoint_dir / "ppo_final")

    logger.info("\n" + "="*80)
    logger.info("Training Complete!")
    logger.info("="*80)
    logger.info(f"Model saved to: {checkpoint_dir}")
    logger.info(f"TensorBoard logs: {tensorboard_dir}")
    logger.info("\nTo view training progress:")
    logger.info(f"  tensorboard --logdir {tensorboard_dir}")

    # Close environment
    vec_env.close()


if __name__ == '__main__':
    main()
