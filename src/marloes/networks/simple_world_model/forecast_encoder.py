import torch
import torch.nn as nn


class ForecastEncoder(nn.Module):
    """
    Encodes a single asset's forecast using a GRU to process the temporal sequence.

    This module takes a sequence of forecast values as input and encodes it into a fixed-size
    representation using the final hidden state of the GRU.

    Attributes:
        gru (nn.GRU): GRU layer for processing the temporal sequence.
    """

    def __init__(self, world_model_config: dict) -> None:
        """
        Initialize the ForecastEncoder.

        Args:
            world_model_config (dict): Configuration dictionary containing:
                - "forecast_hidden_size" (int, optional): Hidden size of the GRU (default: 64).
                - "forecast_num_layers" (int, optional): Number of GRU layers (default: 1).
        """
        super(ForecastEncoder, self).__init__()
        hidden_size = world_model_config.get("forecast_hidden_size", 64)
        num_layers = world_model_config.get("forecast_num_layers", 1)

        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, forecast_sequence: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the GRU encoder.

        Args:
            forecast_sequence (torch.Tensor): Input tensor of shape (batch_size, seq_len, 1),
                where seq_len is the length of the forecast sequence.

        Returns:
            torch.Tensor: Encoded representation of shape (batch_size, hidden_size), which is
                the final hidden state of the last GRU layer.
        """
        # RNN forward pass; only take the final hidden state (h_n)
        _, h_n = self.gru(forecast_sequence)
        # h_n shape: (num_layers, batch_size, hidden_size)
        # Extract the hidden state from the last GRU layer (index -1)
        # Resulting shape: (batch_size, hidden_size)
        return h_n[-1]
