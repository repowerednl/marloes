import torch
import torch.nn as nn
from typing import Tuple


class WorldDynamicsModel(nn.Module):
    """
    World dynamics model that predicts the next state and reward given the current state and action.

    This model takes the encoded world state and actions as input, processes them through shared
    hidden layers, and outputs predictions for the next state and reward.

    Attributes:
        shared_net (nn.Sequential): Shared network for processing the combined input of world state and actions.
        next_state_head (nn.Linear): Linear layer to predict the next state.
        reward_head (nn.Linear): Linear layer to predict the reward.
    """

    def __init__(
        self,
        world_model_config: dict,
        action_dim: int,
        scalar_dims: list[int],
        global_dim: int,
    ) -> None:
        """
        Initialize the WorldDynamicsModel.

        Args:
            world_model_config (dict): Configuration dictionary containing:
                - "world_enc_dim" (int, optional): Dimension of the encoded world state (default: 64).
                - "world_dynamics_hidden_size" (int, optional): Hidden size of the shared network (default: 128).
            action_dim (int): Dimension of the action input.
            scalar_dims (list[int]): List of scalar dimensions for each agent, used to determine the
                output size for the next state prediction.
            global_dim (int): Dimension of the global context, also used in the output size.
        """
        super().__init__()
        world_enc_dim = world_model_config.get("world_enc_dim", 64)
        hidden_size = world_model_config.get("world_dynamics_hidden_size", 128)
        # Only predict the next scalars for each agent
        # We don't need to predict forecast
        # TODO: Explain this in the paper
        next_state_dim = sum(scalar_dims) + global_dim

        self.shared_net = nn.Sequential(
            nn.Linear(world_enc_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Head for next-state and one for reward
        self.next_state_head = nn.Linear(hidden_size, next_state_dim)
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(
        self, world_state: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the world dynamics model.

        Args:
            world_state (torch.Tensor): Encoded world state tensor of shape (batch_size, world_enc_dim).
            actions (torch.Tensor): Action tensor of shape (batch_size, action_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Next state prediction tensor of shape (batch_size, next_state_dim).
                - Reward prediction tensor of shape (batch_size, 1).
        """
        # Concatenate world state and actions
        x = torch.cat([world_state, actions], dim=-1)
        hidden = self.shared_net(x)

        next_state_pred = self.next_state_head(
            hidden
        )  # Shape: (batch_size, next_state_dim)
        reward_pred = self.reward_head(hidden)  # Shape: (batch_size, 1)
        return next_state_pred, reward_pred
