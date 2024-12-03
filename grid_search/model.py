"""Model that coordinates the grid search process."""
import yaml
from sklearn.model_selection import ParameterGrid


class GridSearch:
    def __init__(self, agents: int, coverage: float):
        self.agents = agents
        self.coverage = coverage
        self.results = []

    def load_config(self):
        """Load the configuration for the grid search."""
        with open(f"config_{self.agents}agents.yaml", "r") as file:
            self.config = yaml.safe_load(file)

    def _get_combinations(self):
        """Get coverage % of all combinations of hyperparameters."""
        grid = ParameterGrid(self.config)
        total_combinations = len(grid)
        combinations = int(total_combinations * self.coverage)
        # return random.sample?
        return grid[:combinations]

    def run(self):
        """Run the grid search."""
        configurations = self._get_combinations()
        for config in configurations:
            # run the experiment with the configuration
            continue

    def _save_results(self, config, performance):
        """Save the results of the grid search."""
        results = {}
        # save the parameter combination
        results["config"] = config
        # save the performance measures TODO: define performance and change this
        results["performance"] = performance
        self.results.append(results)

    def _write_results(self):
        """Write the results of the grid search to a file."""
        pass
