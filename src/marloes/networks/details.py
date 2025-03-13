from .base import LayerDetails

RSSM_LD = LayerDetails(
    input={},
    hidden={
        "recurrent": {
            "input_size": 64,
            "hidden_size": 256,
            "num_layers": 2,
            "nonlinearity": "relu",
            "bias": True,
            "batch_first": True,
            "dropout": 0.0,
            "bidirectional": False,
        },
        "dense": {
            "out_features": 64,
        },
    },
    output={},
)
