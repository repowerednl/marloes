"""
Utility functions for testing.
"""


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
):
    """
    Returns a mock observation, with 4 assets, wind, solar, battery and demand.
    As of 12/02/2025
    - battery has SOC, degradation (and is_fcr).
    - solar has forecast, power and available_power.
    - wind has forecast, power and available_power.
    - demand has forecast and power.
    """
    observation = {
        "SolarAgent 0": {
            "forecast": [solar_power] * 3,
            "power": 0.1,
            "available_power": 0.1,
        },
        "WindAgent 0": {
            "forecast": [wind_power] * 3,
            "power": 0.1,
            "available_power": 0.1,
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
            "is_fcr": False,
        }

    return observation
