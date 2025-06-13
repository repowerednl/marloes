import torch
from marloes.networks.simple_world_model.asset_state_encoder import AssetStateEncoder


def test_asset_state_encoder_dimensions():
    # Use defaults for init
    config = {
        "forecast_hidden_size": 64,
        "scalar_dim": 3,
        "asset_enc_dim": 16,
        "forecast_num_layers": 1,
    }
    encoder = AssetStateEncoder(config, config["scalar_dim"], True)

    batch_size = 4
    seq_length = 10

    # Some input
    scalar_vars = torch.randn(batch_size, config["scalar_dim"])
    forecast = torch.randn(batch_size, seq_length, 1)

    # Forward pass
    output = encoder(scalar_vars, forecast)

    # The output should have dimensions: (batch_size, asset_enc_dim)
    assert output.shape == (batch_size, config["asset_enc_dim"])
