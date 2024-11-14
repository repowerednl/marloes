

def validate_config(config: dict) -> str:
    """Validate the configuration dictionary."""
    # Check if all required keys are present
    required_keys = ["algorithm", "epochs", "learning_rate"]
    for key in required_keys:
        if key not in config:
            return f"Error: Missing key '{key}' in configuration"
    
    # Check if the algorithm is valid
    valid_algorithms = ["model_based", "model_free"]
    if config["algorithm"] not in valid_algorithms:
        return f"Error: Invalid algorithm '{config['algorithm']}'"
    
    # Check if epochs is a positive integer
    if not isinstance(config["epochs"], int) or config["epochs"] <= 0:
        return "Error: 'epochs' must be a positive integer"
    
    # Check if learning_rate is a positive float
    if not isinstance(config["learning_rate"], float) or config["learning_rate"] <= 0:
        return "Error: 'learning_rate' must be a positive float"
    
    return "Configuration is valid"