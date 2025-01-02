from .extractor import Extractor

# from marloes.valley.rewards import Reward


class Calculator:
    def __init__(self, uid: int | None = None, dir: str = "results"):
        # also initialize the Extractor
        self.extractor = Extractor(from_model=False)
        # and read the necessary information from files
        self.extractor.from_files(uid, dir)
        # information can be accessed through the extractor attributes

    def get_metrics(self, metrics: list[str]) -> list:
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

    def _match_metrics_to_attributes(self, metrics: list[str]) -> list[str]:
        """
        Function to match the metric to the corresponding attribute in the extractor
        """
        attributes = []
        for metric in metrics:
            # check if the metric is in the extractor attributes
            if hasattr(self.extractor, metric):
                attributes.append(metric)
        # if not, return None
        return attributes

    def _calculate(self):
        return self.extractor.extract() + 1
