from enum import StrEnum, auto


class Metrics(StrEnum):
    # Default metrics
    TIME = auto()
    LOSS = auto()
    REWARD = auto()
    CO2 = auto()
    SS = auto()
    NC = auto()
    NB = auto()
    GRID_STATE = auto()

    # Extensive metrics
    ACTION_PROB_DIST = auto()
    GRID_TO_DEMAND = auto()
    DEMAND_STATE = auto()
    ENERGY_FLOWS = auto()
