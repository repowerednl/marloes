from typing import Iterator
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
from marloes.util import timethis


class WorldModel(nn.Module):
    """
    World model that combines all components to predict the next state and reward.

    This model integrates agent state encoders, a world state encoder, and a world dynamics model
    to process environment states and actions, and predict the next state and reward.

    Attributes:
        agent_state_encoders (nn.ModuleList): list of encoders for each agent's state.
        world_state_encoder (WorldStateEncoder): Encoder for aggregating agent encodings and global context.
        world_dynamics_model (WorldDynamicsModel): Model for predicting the next state and reward.
        optimizer (Adam): Optimizer for training the model.
        loss (list[float]): list to store the loss values during training.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the WorldModel.

        Args:
            config (dict): Configuration dictionary containing:
                - "WorldModel" (dict): Sub-configuration for the world model.
                - "num_agents" (int): Number of agents in the environment.
                - "agents_scalar_dim" (list[int]): list of scalar dimensions for each agent.
                - "forecasts" (list[int]): list of forecast dimensions for each agent.
                - "action_dim" (int): Dimension of the action input.
                - "global_dim" (int, optional): Dimension of the global context (default: 0).
        """
        super(WorldModel, self).__init__()
        self.name = "WorldModel"
        self.world_model_config = config.get("WorldModel", {})
        self.num_agents = config["action_dim"]
        agents_scalar_dim = config["agents_scalar_dim"]
        forecasts = config["forecasts"]

        self.agent_state_encoders = nn.ModuleList(
            [
                AgentStateEncoder(
                    self.world_model_config, agents_scalar_dim[i], forecasts[i]
                )
                for i in range(self.num_agents)
            ]
        )
        global_dim = config.get("global_dim", 0)
        self.world_state_encoder = WorldStateEncoder(
            self.world_model_config, self.num_agents, global_dim
        )
        self.world_dynamics_model = WorldDynamicsModel(
            self.world_model_config,
            config["action_dim"],
            agents_scalar_dim,
            global_dim,
        )
        self.optimizer = Adam(
            self.parameters(),
            lr=self.world_model_config.get("lr", 1e-3),
            weight_decay=self.world_model_config.get("weight_decay", 0.0),
        )
        self.loss = None

        # Load weights if uid is provided
        self.try_to_load_weights(config.get("uid", None))

    def try_to_load_weights(self, uid: int) -> None:
        """
        Load the network weights from a folder if the uid is provided.

        Args:
            uid (int): Unique identifier for the network weights.
        """
        self.was_loaded = False
        try:
            self.load_state_dict(torch.load(f"results/models/{self.name}/{uid}.pt"))
            self.was_loaded = True
        except FileNotFoundError:
            print(
                f"Model weights for {self.name} with uid {uid} not found. Initializing with random weights."
            )

    def forward(
        self,
        scalar_list: list[torch.Tensor],
        forecast_list: list[torch.Tensor],
        global_context: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the world model.

        Args:
            scalar_list (list[torch.Tensor]): list of scalar tensors for each agent.
            forecast_list (list[torch.Tensor]): list of forecast tensors for each agent.
            global_context (torch.Tensor): Global context tensor.
            actions (torch.Tensor): Action tensor of shape (batch_size, action_dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Next state prediction tensor of shape (batch_size, next_state_dim).
                - Reward prediction tensor of shape (batch_size, 1).
        """
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

    def predict(
        self, states: list[dict], actions: list[dict], device: str
    ) -> tuple[list[dict], list[float]]:
        """
        Given environment states and actions, predict the next state and reward.

        Args:
            states (list[dict[str, Any]]): list of state dictionaries for each sample in the batch.
            actions (list[dict[str, float]]): list of action dictionaries for each sample in the batch.
            device (str, optional): Device to place the tensors on (default: "cpu").

        Returns:
            tuple[list[dict[str, Any]], list[float]]:
                - list of reconstructed next state dictionaries.
                - list of predicted rewards.
        """
        # Convert original formats to separable tensors
        parsed_states = parse_state(states, device=device)
        parsed_actions = parse_actions(actions, device=device)

        # Extract scalar variables and forecast from states
        scalar_list = [agent["scalars"] for agent in parsed_states["agents"].values()]
        forecast_list = [
            agent["forecast"] for agent in parsed_states["agents"].values()
        ]

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
        rewards_list = rewards.squeeze(-1).tolist()  # One dimension: squeeze

        # Reconvert next state to original format
        reconstructed_next_states = [
            self._reconstruct_state(state, next_state)
            for state, next_state in zip(states, next_states_list)
        ]

        return reconstructed_next_states, rewards_list

    def update(self, transitions_batch: list[dict], device: str) -> None:
        """
        Update the world model using a batch of real transitions.

        Args:
            transitions_batch (list[dict[str, Any]]): Batch of transition dictionaries.
            device (str, optional): Device to place the tensors on (default: "cpu").
        """
        # Set network to training mode
        self.train()

        # 1. Parse batch to fit world model expectations
        parsed_batch = parse_batch(transitions_batch, device)
        scalar_list = [
            agent["scalars"] for agent in parsed_batch["state"]["agents"].values()
        ]
        forecast_list = [
            agent["forecast"] for agent in parsed_batch["state"]["agents"].values()
        ]
        global_context = parsed_batch["state"].get("global_context", None)
        actions = parsed_batch["actions"]
        rewards = parsed_batch["rewards"]

        # 2. Forward pass
        latent_next_state, reward_pred = self.forward(
            scalar_list, forecast_list, global_context, actions
        )

        # 3. Construct targets
        next_state_target = torch.from_numpy(
            np.array(
                [flatten_state(t.next_state) for t in transitions_batch],
                dtype=np.float32,
            )
        ).to(device)
        reward_target = (
            torch.from_numpy(np.array(rewards.cpu(), dtype=np.float32))
            .unsqueeze(-1)
            .to(device)
        )

        # 4. Compute loss (for now simple MSE)
        next_state_loss = F.mse_loss(latent_next_state, next_state_target)
        reward_loss = F.mse_loss(reward_pred, reward_target)
        total_loss = next_state_loss + reward_loss

        # 5. Backpropagation (also through all submodules)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.loss = total_loss.item()

    def _reconstruct_state(self, state: dict, next_state: list[float]) -> dict:
        """
        Function to reformat output of the network to the original state format.

        Args:
            state (dict[str, Any]): Original state dictionary.
            next_state (list[float]): Flattened next state values.

        Returns:
            dict[str, Any]: Reconstructed state dictionary.
        """
        reconstructed_state = unflatten_state(state, iter(next_state))
        return reconstructed_state


def unflatten_state(original: dict, replacements: Iterator) -> dict:
    """
    Recursively update the state dictionary with new values.

    Args:
        original (dict[str, Any]): Original state dictionary.
        replacements (Iterator): Iterator of replacement values.

    Returns:
        dict: Updated state dictionary.
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


def flatten_state(state: dict) -> list[float]:
    """
    Recursively flattens a state: practically the exact opposite of unflatten_state.

    Args:
        state (dict[str, Any]): State dictionary to flatten.

    Returns:
        list[float]: Flattened state values.
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
