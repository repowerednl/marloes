"""
Necessary datasource classes for the assets
"""
from .base import DataSource

class AssetData(DataSource):
    def __init__(self):
        super().__init__()

    def load_data(self, source: str):
        pass

    def forecast(self):
        pass