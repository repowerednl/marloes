def validate_demand(config: dict) -> str:
    """Validate the demand agent configuration."""
    # Check if required keys are present
    required_keys = ["profile", "scale"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Error: Missing key '{key}' in demand configuration")

    # Check if profile is a string
    if not isinstance(config["profile"], str):
        raise ValueError("Error: 'profile' must be a string")

    # Check if scale is a positive float
    if not isinstance(config["scale"], float) or config["scale"] <= 0:
        raise ValueError("Error: 'scale' must be a positive float")

    return "Demand configuration is valid"
