"""
Base class for data sources
"""
from abc import ABC, abstractmethod
import pandas as pd

class DataSource(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.DataFrame()

    @abstractmethod
    def load_data(self, source: str):
        """ Load data from the source """
        pass

    @abstractmethod
    def forecast(self):
        """ Forecast data (potentail power)"""
        pass