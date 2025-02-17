"""
Utility functions for testing.
"""
from marloes.networks.base import BaseNetwork, LayerDetails


def get_valid_layerdetails(
    hidden_layers: int = 2, dropout: bool = True, sigmoid: bool = True
):
    """
    Returns valid layer details (input, hidden, output) to create LayerDetails object.
    """
    input_out = 2
    input_details = {
        "details": {"in_features": 10, "out_features": input_out},
        "activation": "relu",
    }
    hidden_details = {}
    # add a new hidden layer, and dropout if true
    for i in range(hidden_layers):
        hidden_out = input_out * (i + 1)
        hidden_details[f"layer_{i}"] = {
            "details": {"in_features": input_out, "out_features": hidden_out},
            "activation": "relu",
        }
        if dropout:
            hidden_details[f"dropout_{i}"] = {"details": {"p": 0.5}}
        input_out = hidden_out

    output_details = {
        "details": {"in_features": input_out, "out_features": 1 if sigmoid else 5},
        "activation": "sigmoid" if sigmoid else "relu",
    }
    result = LayerDetails(input_details, hidden_details, output_details)
    result.validate()
    return result


def get_valid_basenetwork():
    """
    Returns a valid BaseNetwork object.
    """
    return BaseNetwork(layer_details=get_valid_layerdetails())
