import torch
import torch.nn as nn

from marloes.networks.simple_world_model.forecast_encoder import ForecastEncoder


class AgentStateEncoder(nn.Module):
    """
    Combines the encoded forecast with scalar features for a single asset.
    """

    def __init__(self, world_model_config: dict, agent_scalar_dim: int, forecast: bool):
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

    def forward(self, scalar_vars, forecast):
        """
        Forward pass through the agent state encoder also encoding the forecast.
        """
        if self.forecast:
            forecast_enc = self.forecast_encoder(forecast)
            combined = torch.cat([forecast_enc, scalar_vars], dim=-1)
        else:
            combined = scalar_vars
        agent_emb = self.mlp(combined)
        return agent_emb
