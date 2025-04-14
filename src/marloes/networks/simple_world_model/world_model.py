import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from marloes.networks.simple_world_model.agent_state_encoder import AgentStateEncoder
from marloes.networks.simple_world_model.util import (
    parse_actions,
    parse_batch,
    parse_state,
)
from marloes.networks.simple_world_model.world_dynamics import WorldDynamicsModel
from marloes.networks.simple_world_model.world_state_encoder import WorldStateEncoder


class WorldModel(nn.Module):
    """
    World model that combines all components.
    """

    def __init__(self, config: dict):
        super(WorldModel, self).__init__()
        self.world_model_config = config.get("WorldModel", {})
        self.num_agents = config["num_agents"]

        self.agent_state_encoders = nn.ModuleList(
            [AgentStateEncoder(self.world_model_config) for _ in range(self.num_agents)]
        )
        self.world_state_encoder = WorldStateEncoder(
            self.world_model_config, self.num_agents, config.get("global_dim", 0)
        )
        self.world_dynamics_model = WorldDynamicsModel(
            self.world_model_config, config["action_dim"]
        )
        self.optimizer = Adam(
            self.parameters(),
            lr=self.world_model_config.get("lr", 1e-3),
            weight_decay=self.world_model_config.get("weight_decay", 0.0),
        )
        self.loss = []

    def forward(self, scalar_list, forecast_list, global_context, actions):
        # Encode each agent
        agent_encodings = []
        for i in range(self.num_agents):
            agent_encoding = self.agent_state_encoders[i](
                scalar_list[i], forecast_list[i]
            )
            agent_encodings.append(agent_encoding)

        # Aggregate in the world state encoder
        world_state = self.world_state_encoder(agent_encodings, global_context)

        # Predict next state and reward
        next_state, reward = self.world_dynamics_model(world_state, actions)

        return next_state, reward

    def predict(self, states: list[dict], actions: list[dict], device: str = "cpu"):
        """
        Given environment states and actions, predict the next state and reward.
        """
        # Convert original formats to separable tensors
        parsed_states = parse_state(states, device=device)
        parsed_actions = parse_actions(actions, device=device)

        # Extract scalar variables and forecast from states
        scalar_list = [agent["scalars"] for agent in parsed_states["agents"]]
        forecast_list = [agent["forecast"] for agent in parsed_states["agents"]]

        # Extract global context if available
        global_context = None
        if "global_context" in parsed_states:
            global_context = parsed_states["global_context"]

        # Forward pass through the model
        next_states, rewards = self.forward(
            scalar_list, forecast_list, global_context, parsed_actions
        )

        # Reconvert next state to original format
        reconstructed_next_states = [
            self._reconstruct_state(curr, pred)
            for curr, pred in zip(states, next_states)
        ]

        return reconstructed_next_states, rewards

    def update(self, transitions_batch):
        """
        Update the world model using a batch of real transitions.
        """
        pass

    def _reconstruct_state(self, state, next_state):
        pass
