from .base import Algorithm


class Priorities(Algorithm):
    """
    This algorithm simply uses the priority functionality from simon to solve the energy flows.
    Stepping the model will automatically solve the energy flows.
    """

    def __init__(self, config: dict, save_energy_flows: bool = False):
        super().__init__(config, save_energy_flows)

    def train(self):
        """
        Run the simulation/training phase of the algorithm, can be overridden by subclasses.
        """
        # Get the initial observation
        observation = self.valley.reset()

        for epoch in range(self.epochs):
            # The priorities are set in the agents, so no actions should be provided
            actions = {}

            # Take a step in the environment
            observation, reward, done, info = self.valley.step(actions)

            # save results? (Extractor in the valley also saves the results)

            # no need to learn any algorithm with the reward

        # Stash any final results
        self.save()

    def get_actions(self, observation):
        pass

    def load(self):
        pass
