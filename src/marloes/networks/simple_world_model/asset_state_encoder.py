import torch
import torch.nn as nn

from marloes.networks.simple_world_model.forecast_encoder import ForecastEncoder


class AssetStateEncoder(nn.Module):
    """
    Combines the encoded forecast with scalar features for a single asset.

    This module encodes scalar variables and optionally a forecast sequence for a single handler,
    and outputs a fixed-size representation of the handler's state.

    Attributes:
        forecast (bool): Whether the handler has forecast data to encode.
        forecast_encoder (ForecastEncoder, optional): GRU-based encoder for the forecast sequence.
        mlp (nn.Sequential): MLP for combining scalar variables and forecast encodings.
    """

    def __init__(
        self,
        world_model_config: dict,
        handler_scalar_dim: int,
        forecast: bool,
        use_gru: bool = True,
    ) -> None:
        """
        Initialize the AssetStateEncoder.

        Args:
            world_model_config (dict): Configuration dictionary containing:
                - "forecast_hidden_size" (int, optional): Hidden size of the forecast GRU (default: 64).
                - "asset_enc_dim" (int, optional): Dimension of the handler's encoded state (default: 16).
                - "asset_hidden_size" (int, optional): Hidden size of the MLP (default: 64).
            handler_scalar_dim (int): Dimension of the scalar variables for the handler.
            forecast (bool): Whether the handler has forecast data to encode.
        """
        super(AssetStateEncoder, self).__init__()
        self.forecast = forecast
        self.use_gru = use_gru
        if self.use_gru:
            forecast_hidden_size = world_model_config.get("forecast_hidden_size", 64)
        else:
            forecast_hidden_size = 240
        asset_enc_dim = world_model_config.get("asset_enc_dim", 16)
        hidden_size = world_model_config.get("asset_hidden_size", 64)

        if forecast:
            self.forecast_encoder = ForecastEncoder(world_model_config)
        else:
            forecast_hidden_size = 0

        self.mlp = nn.Sequential(
            nn.Linear(forecast_hidden_size + handler_scalar_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, asset_enc_dim),
            nn.ReLU(),
        )

    def forward(
        self, scalar_vars: torch.Tensor, forecast: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Perform a forward pass through the handler state encoder.

        Args:
            scalar_vars (torch.Tensor): Tensor of scalar variables with shape (batch_size, handler_scalar_dim).
            forecast (torch.Tensor, optional): Tensor of forecast data with shape (batch_size, seq_len, 1).
                If None, only scalar variables are used.

        Returns:
            torch.Tensor: Encoded handler state tensor of shape (batch_size, asset_enc_dim).
        """
        if self.forecast:
            if self.use_gru:
                forecast_enc = self.forecast_encoder(forecast)
                combined = torch.cat([forecast_enc, scalar_vars], dim=-1)
            else:
                forecast_flat = forecast.squeeze(-1)
                combined = torch.cat([forecast_flat, scalar_vars], dim=-1)
        else:
            combined = scalar_vars

        handler_emb = self.mlp(combined)
        return handler_emb
