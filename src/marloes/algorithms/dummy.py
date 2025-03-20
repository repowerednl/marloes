from .base import BaseAlgorithm
from marloes.networks import WorldModel
import random
from torch.optim import Adam


class Dummy(BaseAlgorithm):
    __name__ = "Dummy"

    def __init__(self, config: dict):
        """
        Initializes the Dummy algorithm working with the WorldModel for Dreamer.
        """
        super().__init__(config)
        # initialize the world model
        self._initialize_world_model()
        self._initialize_actor_critic()
        parameters = [
            param
            for mod in (self.world_model.modules + self.actor_critic.modules)
            for param in mod.parameters()
        ]
        # TODO: hyperparams
        self.optimizer = Adam(
            params=parameters,
            lr=0.001,
            weight_decay=1e-4,
        )

    def _initialize_world_model(self):
        """
        Gets observation shape, and actions shape from the environment.
        """
        obs_shape = (1, 1)  # TODO: get observation shape from environment
        action_shape = (1, 1)  # TODO: get action shape from environment
        # TODO: load parameters in if provided in config (if UID is passed)
        self.world_model = WorldModel(obs_shape, action_shape)

    def _initialize_actor_critic(self):
        """
        Initializes the actor and critic networks.
        """
        pass

    def _get_actions(self, observations):
        """
        Random actions for each agent in the environment.
        """
        return {agent_id: random.uniform(-1, 1) for agent_id in observations.keys()}

    def _train_step(self, observations, rewards, dones, infos):
        """
        Dummy training step, simulating training process to test optimizer loop for ___
        """
