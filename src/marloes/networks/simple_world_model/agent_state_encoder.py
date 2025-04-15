import torch
import torch.nn as nn

from marloes.networks.simple_world_model.forecast_encoder import ForecastEncoder


class AgentStateEncoder(nn.Module):
    """
    Combines the encoded forecast with scalar features for a single asset.

    This module encodes scalar variables and optionally a forecast sequence for a single agent,
    and outputs a fixed-size representation of the agent's state.

    Attributes:
        forecast (bool): Whether the agent has forecast data to encode.
        forecast_encoder (ForecastEncoder, optional): GRU-based encoder for the forecast sequence.
        mlp (nn.Sequential): MLP for combining scalar variables and forecast encodings.
    """

    def __init__(
        self, world_model_config: dict, agent_scalar_dim: int, forecast: bool
    ) -> None:
        """
        Initialize the AgentStateEncoder.

        Args:
            world_model_config (dict): Configuration dictionary containing:
                - "forecast_hidden_size" (int, optional): Hidden size of the forecast GRU (default: 64).
                - "agent_enc_dim" (int, optional): Dimension of the agent's encoded state (default: 16).
                - "agent_hidden_size" (int, optional): Hidden size of the MLP (default: 64).
            agent_scalar_dim (int): Dimension of the scalar variables for the agent.
            forecast (bool): Whether the agent has forecast data to encode.
        """
        super(AgentStateEncoder, self).__init__()
        self.forecast = forecast
        forecast_hidden_size = world_model_config.get("forecast_hidden_size", 64)
        agent_enc_dim = world_model_config.get("agent_enc_dim", 16)
        hidden_size = world_model_config.get("agent_hidden_size", 64)

        if forecast:
            self.forecast_encoder = ForecastEncoder(world_model_config)
        else:
            forecast_hidden_size = 0

        self.mlp = nn.Sequential(
            nn.Linear(forecast_hidden_size + agent_scalar_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, agent_enc_dim),
            nn.ReLU(),
        )

    def forward(
        self, scalar_vars: torch.Tensor, forecast: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Perform a forward pass through the agent state encoder.

        Args:
            scalar_vars (torch.Tensor): Tensor of scalar variables with shape (batch_size, agent_scalar_dim).
            forecast (torch.Tensor, optional): Tensor of forecast data with shape (batch_size, seq_len, 1).
                If None, only scalar variables are used.

        Returns:
            torch.Tensor: Encoded agent state tensor of shape (batch_size, agent_enc_dim).
        """
        if self.forecast:
            forecast_enc = self.forecast_encoder(forecast)
            combined = torch.cat([forecast_enc, scalar_vars], dim=-1)
        else:
            combined = scalar_vars

        agent_emb = self.mlp(combined)
        return agent_emb
