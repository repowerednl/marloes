from .base import Algorithm

from marloes.agents.grid import GridAgent
from ray.rllib.algorithms.sac import SACConfig


class SACfromRay(Algorithm):
    def __init__(self, config: dict, save_energy_flows: bool = False):
        super().__init__(config, save_energy_flows)  # Initialize the valley/epochs
        self.config = (
            SACConfig()
            .environment(env=lambda _: self.valley)
            .multi_agent(
                policies={
                    agent.id: (
                        None,
                        self.valley.observation_space,
                        self.valley.action_space,
                        {},
                    )
                    for agent in self.valley.agents
                    if not isinstance(agent, GridAgent)
                },
                policy_mapping_fn=lambda agent_id: agent_id,
            )
        )

    def get_actions(self, observation):
        pass

    def train(self, observation, reward, done, info):
        pass

    def save(self):
        pass

    def load(self):
        pass
