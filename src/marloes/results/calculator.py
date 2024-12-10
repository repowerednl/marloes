from .extractor import Extractor


class Calculator:
    def __init__(self):
        self.extractor = Extractor()

    def _get_marloes_info(self):
        """
        Function to get the necessary information for the marloes using the Extractor
        """
        all_marloes = {
            "marloes": "marloes",
            "marloes2": "marloes2",
        }
        marloes = {}
        for key, _ in all_marloes.items():
            series = self.extractor.from_file(key)
            marloes[key] = series

    def _get_metric_info(self):
        """
        Function to get the necessary information for the metrics using the Extractor
        """
        all_metrics = {
            "metric": "metric",
            "metric2": "metric2",
        }
        metrics = {}
        for key, _ in all_metrics.items():
            series = self.extractor.from_file(key)
            metrics[key] = series

    def calculate(self):
        return self.extractor.extract() + 1
