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
    base = BaseNetwork()
    base._initialize_layers_from_layer_details(get_valid_layerdetails())
    return base


def get_accurate_observation(algorithm):
    """
    This function takes an algorithm, which has an environment.
    It should return an "observation" which can be used to mock the step/reset function.
    """
    combined_states = algorithm.environment._combine_states()
    return combined_states


def get_mock_observation(
    battery_soc: list = [0.5],
    solar_power: float = 2.0,
    wind_power: float = 1.0,
    solar_nomination: float = 1.8,
    wind_nomination: float = 0.9,
):
    """
    Returns a mock observation, with 4 assets, wind, solar, battery and demand.
    As of 12/02/2025
    - battery has SOC, degradation
    - solar has forecast, power and available_power and nomination
    - wind has forecast, power and available_power and nomination
    - demand has forecast and power
    """
    observation = {
        "SolarAgent 0": {
            "forecast": [solar_power] * 3,
            "power": solar_power,
            "available_power": 0.1,
            "nomination": solar_nomination,
        },
        "WindAgent 0": {
            "forecast": [wind_power] * 3,
            "power": wind_power,
            "available_power": 0.1,
            "nomination": wind_nomination,
        },
        "DemandAgent 0": {
            "forecast": [-2.0, -3.0, -4.0],
            "power": 0.1,
        },
    }

    for i, soc in enumerate(battery_soc):
        observation[f"BatteryAgent {i}"] = {
            "state_of_charge": soc,
            "degradation": 0.2,
        }

    return observation


def get_new_config():
    """
    Function to return a configuration of agents for creating an Environment/Model.
    """
    return {
        "agents": [
            {
                "type": "demand",
                "scale": 1.5,
                "profile": "Farm",
            },
            {
                "type": "solar",
                "AC": 900,
                "DC": 1000,
                "orientation": "EW",
            },
            {
                "type": "battery",
                "efficiency": 0.9,
                "power": 100,
                "energy_capacity": 1000,
            },
            {
                "type": "demand",
                "scale": 1.5,
                "profile": "Farm",
            },
        ],
        "grid": {
            "name": "Grid_One",
            "max_power_in": 1000,
        },
    }
