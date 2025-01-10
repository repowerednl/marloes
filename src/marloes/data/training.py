from pydantic import BaseModel

from marloes.algorithms.base import BaseAlgorithm


class TrainingData(BaseModel):
    """
    Data class for training data.
    """

    algorithm: BaseAlgorithm
    epoch: int
    reward: float
    loss: float
