from enum import StrEnum, auto


class Metrics(StrEnum):
    # Default metrics
    TIME = "elapsed_time"
    LOSS = auto()
    REWARD = auto()
    CO2 = "CO2"
    SS = "SS"
    NC = "NC"
    NB = "NB"
    GRID_STATE = auto()

    # Extensive metrics
    ACTION_PROB_DIST = auto()
    GRID_TO_DEMAND = auto()
    DEMAND_STATE = auto()
    ENERGY_FLOWS = auto()
