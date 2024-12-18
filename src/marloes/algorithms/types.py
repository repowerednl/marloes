from enum import Enum, auto


class AlgorithmType(Enum):
    """
    Enum to represent different algorithm types.
    """

    MODEL_BASED = auto()
    MODEL_FREE = auto()
    PRIORITIES = auto()
    MADDPG = auto()
    SIMPLE_SETPOINT = auto()


def parse_algorithm_type(algorithm_name: str) -> AlgorithmType:
    """
    Parses the algorithm name into the corresponding AlgorithmType.
    """
    try:
        return AlgorithmType[algorithm_name.upper()]
    except KeyError:
        raise ValueError(f"Unknown algorithm type: '{algorithm_name}'")
