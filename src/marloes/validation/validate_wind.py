def validate_wind(config: dict) -> str:
    """Validate the Wind agent configuration."""
    # Check if required keys are present
    required_keys = ["location", "AC", "power"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Error: Missing key '{key}' in solar configuration")

    # Check if orientation is a string
    if not isinstance(config["location"], str):
        raise ValueError("Error: 'location' must be a string")

    # Check if power is a positive float
    if config["power"] <= 0:
        raise ValueError("Error: 'power' must be a positive float")

    # Check if AC is a positive float
    if config["AC"] <= 0:
        raise ValueError("Error: 'AC' must be a positive float")

    return "Wind configuration is valid"
