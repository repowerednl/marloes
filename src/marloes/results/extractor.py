import numpy as np


class Extractor:
    def __init__(self) -> None:
        self.data = {"metric": np.zeros(10)}


class ExtensiveExtractor(Extractor):
    pass
