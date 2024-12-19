from .extractor import Extractor

# from marloes.valley.rewards import Reward


class Calculator:
    def __init__(self, uid: int | None = None, dir: str = "results"):
        # also initialize the Extractor
        self.extractor = Extractor(from_model=False)
        # and read the necessary information from files
        self.extractor.from_files(uid, dir)
        # information can be accessed through the extractor attributes

    def _get_metrics(self, metrics: list[str]) -> list:
        """
        Function to get the necessary information (metrics/attributes from the extractor)
        """
        # initialize the list to store the metrics
        metric_values = []
        for metric in metrics:
            attribute = self._match_metric_to_attribute(metric)
            if not attribute:
                metric_values.append(None)
                continue
            metric_values.append(getattr(self.extractor, attribute))

    def _match_metric_to_attribute(self, metric: str) -> str | None:
        """
        Function to match the metric to the corresponding attribute in the extractor
        """
        # check if the metric is in the extractor attributes
        if hasattr(self.extractor, metric):
            return metric
        # if not, return None
        return None

    def calculate(self):
        return self.extractor.extract() + 1
