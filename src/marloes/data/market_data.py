"""
Data source classes for the necessary market data
"""
from .base import DataSource

class MarketData(DataSource):
    def __init__(self):
        super().__init__()

    def load_data(self, source: str):
        pass

    def forecast(self):
        pass