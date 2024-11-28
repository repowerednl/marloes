def validate_battery(config: dict) -> str:
    """Validate the battery agent configuration."""
    # Check if required keys are present
    required_keys = [
        "max_power_in",
        "energy_capacity",
    ]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Error: Missing key '{key}' in battery configuration")

    # Check if max_power_in is a positive float
    if config["max_power_in"] <= 0:
        raise ValueError("Error: 'max_power_in' must be a positive float")

    # Check if energy_capacity is a positive float
    if config["energy_capacity"] <= 0:
        raise ValueError("Error: 'energy_capacity' must be a positive float")

    return "Battery configuration is valid"
