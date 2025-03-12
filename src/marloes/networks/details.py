from .base import LayerDetails

RSSM_LD = LayerDetails(
    input={},
    hidden={
        "recurrent": {
            "type": "GRU",
            "hidden_size": 256,
            "num_layers": 2,
            "batch_first": True,
        },
        "dense": {
            "type": "Linear",
            "hidden_size": 64,
        },
    },
    output={},
)
