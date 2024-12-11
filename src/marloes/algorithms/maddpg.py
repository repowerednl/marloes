from .base import Algorithm


class MADDPG(Algorithm):
    def __init__(self, config: dict, save_energy_flows: bool = False):
        super().__init__(config, save_energy_flows)  # Initialize the valley/epochs

    def get_actions(self, observation):
        pass

    def train(self, observation, reward, done, info):
        pass

    def save(self):
        pass

    def load(self):
        pass
