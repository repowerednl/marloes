import torch
import torch.nn as nn

from marloes.networks.simple_world_model.forecast_encoder import ForecastEncoder


class AgentStateEncoder(nn.Module):
    """
    Combines the encoded forecast with scalar features for a single asset.
    """

    def __init__(self, world_model_config: dict, agent_scalar_dim: int):
        super(AgentStateEncoder, self).__init__()
        forecast_hidden_size = world_model_config.get("forecast_hidden_size", 64)
        agent_enc_dim = world_model_config.get("agent_enc_dim", 16)

        self.forecast_encoder = ForecastEncoder(world_model_config)
        self.mlp = nn.Sequential(
            nn.Linear(forecast_hidden_size + agent_scalar_dim, 64),
            nn.ReLU(),
            nn.Linear(64, agent_enc_dim),
            nn.ReLU(),
        )

    def forward(self, scalar_vars, forecast):
        """
        Forward pass through the agent state encoder also encoding the forecast.
        """
        forecast_enc = self.forecast_encoder(forecast)
        combined = torch.cat([forecast_enc, scalar_vars], dim=-1)
        agent_emb = self.mlp(combined)
        return agent_emb
