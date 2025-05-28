from .base import LayerDetails

RSSM_LD = LayerDetails(
    input={},
    hidden={
        "recurrent": {
            "input_size": 100,  # h_t (64), z_t (32), a_t (number of agents)
            "hidden_size": 64,
            "num_layers": 2,
            "bias": True,
            "batch_first": True,
            "dropout": 0.0,
            "bidirectional": False,
        },
        "dense": {
            "out_features": 32,  # z_t
        },
    },
    output={},
)
