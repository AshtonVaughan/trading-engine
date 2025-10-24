"""
Large Transformer-LSTM Policy Network for PPO

Architecture:
- LSTM feature extractor (1024x8 layers)
- Transformer encoder (768d, 12 layers, 12 heads)
- Separate actor/critic heads
- ~50M parameters total
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
from typing import Tuple


class TransformerLSTMExtractor(BaseFeaturesExtractor):
    """
    Feature extractor with LSTM + Transformer architecture.

    Optimized for H200 NVL with bfloat16 precision.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 768,
        lstm_hidden: int = 1024,
        lstm_layers: int = 8,
        transformer_heads: int = 12,
        transformer_layers: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)

        obs_dim = observation_space.shape[0]

        # Input projection
        self.input_proj = nn.Linear(obs_dim, features_dim)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=features_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Project LSTM output to transformer dimension
        self.lstm_proj = nn.Linear(lstm_hidden, features_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=features_dim,
            nhead=transformer_heads,
            dim_feedforward=features_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(features_dim)

        # Output dimension
        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            observations: (batch_size, obs_dim)

        Returns:
            features: (batch_size, features_dim)
        """
        # Project input
        x = self.input_proj(observations)  # (B, features_dim)

        # Add sequence dimension for LSTM
        x = x.unsqueeze(1)  # (B, 1, features_dim)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, 1, lstm_hidden)

        # Project to transformer dim
        x = self.lstm_proj(lstm_out)  # (B, 1, features_dim)

        # Transformer encoder
        x = self.transformer(x)  # (B, 1, features_dim)

        # Remove sequence dimension
        x = x.squeeze(1)  # (B, features_dim)

        # Layer norm
        x = self.layer_norm(x)

        return x


class DualCriticHead(nn.Module):
    """
    Dual critic heads to reduce overestimation (like TD3).
    """

    def __init__(self, features_dim: int, hidden_size: int = 1024, num_layers: int = 6):
        super().__init__()

        # Critic 1
        layers1 = []
        in_dim = features_dim
        for _ in range(num_layers):
            layers1.append(nn.Linear(in_dim, hidden_size))
            layers1.append(nn.LayerNorm(hidden_size))
            layers1.append(nn.GELU())
            in_dim = hidden_size
        layers1.append(nn.Linear(hidden_size, 1))
        self.critic1 = nn.Sequential(*layers1)

        # Critic 2
        layers2 = []
        in_dim = features_dim
        for _ in range(num_layers):
            layers2.append(nn.Linear(in_dim, hidden_size))
            layers2.append(nn.LayerNorm(hidden_size))
            layers2.append(nn.GELU())
            in_dim = hidden_size
        layers2.append(nn.Linear(hidden_size, 1))
        self.critic2 = nn.Sequential(*layers2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return minimum of two critic values."""
        v1 = self.critic1(features)
        v2 = self.critic2(features)
        return torch.min(v1, v2)


class TransformerActorCriticPolicy(ActorCriticPolicy):
    """
    Custom ActorCritic policy with Transformer-LSTM extractor.
    """

    def __init__(self, *args, **kwargs):
        # Extract custom kwargs
        lstm_hidden = kwargs.pop('lstm_hidden', 1024)
        lstm_layers = kwargs.pop('lstm_layers', 8)
        transformer_heads = kwargs.pop('transformer_heads', 12)
        transformer_layers = kwargs.pop('transformer_layers', 12)
        features_dim = kwargs.pop('d_model', 768)
        dropout = kwargs.pop('dropout', 0.1)

        # Set features extractor
        kwargs['features_extractor_class'] = TransformerLSTMExtractor
        kwargs['features_extractor_kwargs'] = {
            'features_dim': features_dim,
            'lstm_hidden': lstm_hidden,
            'lstm_layers': lstm_layers,
            'transformer_heads': transformer_heads,
            'transformer_layers': transformer_layers,
            'dropout': dropout,
        }

        super().__init__(*args, **kwargs)

        # Replace value network with dual critic
        if kwargs.get('dual_critics', True):
            self.value_net = DualCriticHead(
                features_dim=self.features_dim,
                hidden_size=kwargs.get('critic_hidden_size', 1024),
                num_layers=kwargs.get('critic_num_layers', 6),
            )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
