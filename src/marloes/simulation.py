from .valley.env import EnergyValley
import random


class Simulation:
    """
    Simulation class for the EnergyValley environment
    This will apply the algorithm to the environment training the model.
    Saving the energy flows is optional (costly).
    """

    def __init__(self, config: dict, save_energy_flows: bool = False):
        algorithm = config.pop("algorithm")
        self.epochs = config.pop("epochs")
        self.valley = EnergyValley(config)
        self.saving = save_energy_flows
        # TODO: initialize the EnergyFlows class/model if saving is True
        self.flows = [] if self.saving else None
        # TODO: initialize the algorithm
        self.algorithm = algorithm

    def run(self):
        """
        Run the simulation/training phase of the algorithm.
        """
        # Get the initial observation
        observation = self.valley.reset()
        for epoch in range(self.epochs):
            # TODO: Get the actions from the algorithm
            actions = [random.random(0, 1) for agent in self.valley.agents]
            # Take a step in the environment
            observation, reward, done, info = self.valley.step(actions)
            # TODO: Save the energy flows if saving is True
            if self.saving:
                self.flows.append(info)
            # TODO: Train the algorithm
            # TODO: Save the training results
        # Stash any final results
