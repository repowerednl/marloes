import numpy as np
from simon.solver import Model

MINUTES_IN_A_YEAR = 525600


class Extractor:
    """
    Extractor class to extract information relevant to produce metrics.
    Can be adapted to extract more information if needed.
    """

    def __init__(self, chunk_size: int = 0):
        # Position tracking
        self.i = 0
        size = MINUTES_IN_A_YEAR // chunk_size

        # Marl(oes) info
        self.elapsed_time = np.zeros(size)
        self.loss = np.zeros(size)
        self.action_probability_distribution = np.zeros((size, 2))

        # Metrics/Reward info
        self.grid_to_demand = np.zeros(size)
        self.demand_state = np.zeros(size)
        self.grid_state = np.zeros(size)

        # Emmission info
        self.total_solar_production = np.zeros(size)
        self.total_battery_production = np.zeros(size)
        self.total_wind_production = np.zeros(size)
        self.total_grid_production = np.zeros(size)

    def clear(self):
        """Clear the extractor"""
        self.i = 0

    def from_model(self, model: Model):
        """Extract information from the Simon model"""
        pass

    def from_files(self, uid: int):
        """Extract information from files given a unique identifier"""
        pass


class ExtensiveExtractor(Extractor):
    pass
