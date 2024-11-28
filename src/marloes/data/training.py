from pydantic import BaseModel
from marloes.algorithms.base import AlgorithmType


class TrainingData(BaseModel):
    """
    Data class for training data.
    """

    algorithm: AlgorithmType
    epoch: int
    reward: float
    loss: float
