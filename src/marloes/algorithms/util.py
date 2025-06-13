def get_net_forecasted_power(observations: dict, period: int = 1) -> float:
    """
    Looks at the forecasts of each supply and demand handler to calculate the net power.
    Sum the forecasts of the next period, period is in minutes, defaults to 1.
    """
    forecasts = [
        observations[handler]["forecast"]
        for handler in observations.keys()
        if "forecast" in observations[handler]
    ]
    if forecasts:
        # Ensure the period does not exceed the length of any forecast
        period = min(period, min(len(forecast) for forecast in forecasts))
    else:
        # if no forecasts are available, return 0.0
        return 0.0

    return sum(sum(forecast[:period]) for forecast in forecasts)


def get_net_power(observations: dict, period: int = 1) -> float:
    """
    Looks at the powers of each supply and demand handler to calculate the net power.
    Sum the powers of the next period, period is in minutes, defaults to 1.
    """
    powers = [
        observations[handler]["power"]
        for handler in observations.keys()
        if "power" in observations[handler]
    ]

    return sum(powers)
