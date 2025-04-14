import numpy as np
import torch
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
        # Detach model since we are not training
        self.eval()
        with torch.no_grad():
            next_states, rewards = self.forward(
                scalar_list, forecast_list, global_context, parsed_actions
            )

        # Convert next states and rewards to numpy arrays
        next_states_list = next_states.tolist()
        rewards_list = rewards.squeeze(0).tolist()  # One dimension: squeeze

        # Reconvert next state to original format
        reconstructed_next_states = [
            self._reconstruct_state(state, next_state)
            for state, next_state in zip(states, next_states_list)
        ]

        return reconstructed_next_states, rewards_list

    def update(self, transitions_batch: list[dict], device: str = "cpu"):
        """
        Update the world model using a batch of real transitions.
        """
        # Set network to training mode
        self.train()

        # 1. Parse batch to fit world model expectations
        parsed_batch = parse_batch(transitions_batch)
        scalar_list = [agent["scalars"] for agent in parsed_batch["state"]["agents"]]
        forecast_list = [agent["forecast"] for agent in parsed_batch["state"]["agents"]]
        global_context = parsed_batch["state"].get("global_context", None)
        actions = parsed_batch["actions"]
        rewards = parsed_batch["rewards"]
        next_state = parsed_batch["next_state"]

        # 2. Forward pass
        latent_next_state, reward_pred = self.forward(
            scalar_list, forecast_list, global_context, actions
        )

        # 3. Construct targets
        next_state_target = torch.from_numpy(
            np.array([flatten_state(state) for state in next_state], dtype=np.float32)
        ).to(device)
        reward_target = torch.from_numpy(np.array(rewards, dtype=np.float32)).to(device)

        # 4. Compute loss (for now simple MSE)
        next_state_loss = F.mse_loss(latent_next_state, next_state_target)
        reward_loss = F.mse_loss(reward_pred, reward_target)
        total_loss = next_state_loss + reward_loss

        # 5. Backpropagation (also through all submodules)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.loss.append(total_loss.item())

    def _reconstruct_state(self, state, next_state):
        """
        Function to reformat output of the network to the original state format.
        """
        reconstructed_state = unflatten_state(state, next_state)
        return reconstructed_state


def unflatten_state(original, replacements):
    """
    Recursively update the state dictionary with new values.
    """
    if isinstance(original, dict):
        return {k: unflatten_state(v, replacements) for k, v in original.items()}

    elif isinstance(original, np.ndarray):
        # We do not predict the forecast currently, so we need to shift the array
        return np.append(original[1:], 0)  # Shift left, append 0

    elif isinstance(original, float):
        return next(replacements)  # Replace with next value from list

    else:
        return original  # Leave as-is


def flatten_state(state):
    """
    Recursively flattens a state: practically the exact opposite of unflatten_state.
    """
    flattened = []

    if isinstance(state, dict):
        for key in state:
            # We do not predict forecast, so skip it
            if key == "forecast":
                continue
            flattened.extend(flatten_state(state[key]))

    elif isinstance(state, (float, int)):
        flattened.append(float(state))

    else:
        # For now do nothing for other types
        pass

    return flattened
