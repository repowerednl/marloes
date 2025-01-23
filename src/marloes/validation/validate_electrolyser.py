def validate_electrolyser(config: dict) -> str:
    """Validate the electrolyser agent configuration."""
    # Check if required keys are present
    required_keys = [
        "energy_capacity",
        "power",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(
                f"Error: Missing key '{key}' in electrolyser configuration"
            )

    # Check if power is a positive float
    if config["power"] <= 0:
        raise ValueError("Error: 'power' must be a positive float")

    # Check if energy_capacity is a positive float
    if config["energy_capacity"] <= 0:
        raise ValueError("Error: 'energy_capacity' must be a positive float")

    return "Electrolyser configuration is valid"
