import torch.nn as nn


class ForecastEncoder(nn.Module):
    """
    Encodes a single asset's forecast, using a GRU to process the temporal sequence.
    """

    def __init__(self, world_model_config: dict):
        super(ForecastEncoder, self).__init__()
        hidden_size = world_model_config.get("forecast_hidden_size", 64)
        num_layers = world_model_config.get("forecast_num_layers", 1)

        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, forecast_sequence):
        """
        Forward pass through the GRU encoder.
        """
        # RNN forward pass; only take final hidden state (h_n)
        _, h_n = self.gru(forecast_sequence)
        # h_n shape should be: (num_layers, batch_size, hidden_size)
        # For single-layer GRU, squeeze out the first dimension
        return h_n[-1]
