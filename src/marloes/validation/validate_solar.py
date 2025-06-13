def validate_solar(config: dict) -> str:
    """Validate the solar handler configuration."""
    # Check if required keys are present
    required_keys = ["orientation", "DC", "AC"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Error: Missing key '{key}' in solar configuration")

    # Check if orientation is a string
    if not isinstance(config["orientation"], str):
        raise ValueError("Error: 'orientation' must be a string")

    # Check if DC is a positive float
    if config["DC"] <= 0:
        raise ValueError("Error: 'DC' must be a positive float")

    # Check if AC is a positive float
    if config["AC"] <= 0:
        raise ValueError("Error: 'AC' must be a positive float")

    return "Solar configuration is valid"
