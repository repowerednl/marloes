from .base import LayerDetails

RSSM_LD = LayerDetails(
    input={},
    hidden={
        "recurrent": {
            "input_size": 326,  # h_t (256), z_t (64), a_t (number of agents = 6)
            "hidden_size": 256,
            "num_layers": 2,
            "bias": True,
            "batch_first": True,
            "dropout": 0.0,
            "bidirectional": False,
        },
        "dense": {
            "out_features": 64,  # z_t
        },
    },
    output={},
)
