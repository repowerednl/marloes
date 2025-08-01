# imports
from marloes.algorithms.base import BaseAlgorithm
from marloes.validation.validate_battery import validate_battery
from marloes.validation.validate_solar import validate_solar
from marloes.validation.validate_demand import validate_demand


def validate_config(config: dict) -> str:
    """Validate the configuration dictionary."""
    # Check if the algorithm is valid
    valid_algorithms = [name for name, _ in BaseAlgorithm._registry.items()]
    algorithm = config.get("algorithm", None)
    if algorithm not in valid_algorithms:
        raise ValueError(f"Error: Invalid algorithm '{algorithm}'")

    if config["algorithm"] in ["model_based", "model_free"]:
        # Check if required keys are present
        required_keys = [
            "epochs",
            "learning_rate" if not config["grid_search"] else "coverage",
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Error: Missing key '{key}' in configuration")

        # Check if epochs is a positive integer
        if not isinstance(config["epochs"], int) or config["epochs"] <= 0:
            raise ValueError("Error: 'epochs' must be a positive integer")

        # Check if learning_rate is a positive float
        if not config["grid_search"] and (
            not isinstance(config["learning_rate"], float)
            or config["learning_rate"] <= 0
        ):
            raise ValueError("Error: 'learning_rate' must be a positive float")

    # Validate each agent configuration
    for agent in config["agents"]:
        globals().get(f"validate_{agent['type']}")(agent)

    return "Configuration is valid"
