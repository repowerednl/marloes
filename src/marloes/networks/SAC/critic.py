import torch
import torch.nn as nn
from marloes.networks.SAC.base import SACBaseNetwork


class CriticNetwork(SACBaseNetwork):
    """
    Critic network (Q-function) for the Soft Actor-Critic (SAC) algorithm.

    This network estimates the Q-value for a given state-action pair. It takes the state
    and action as input, processes them through shared hidden layers, and outputs a single
    scalar Q-value.

    Attributes:
        output_layer (nn.Linear): Linear layer to produce the Q-value from the final hidden layer.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the CriticNetwork.

        Args:
            config (dict): Configuration dictionary containing:
                - "state_dim" (int): Dimension of the input state.
                - "action_dim" (int): Dimension of the input action.
                - Other keys inherited from SACBaseNetwork for hidden layers configuration.
        """
        # The input dimension needs to include both the state and action dimensions
        state_dim = config["state_dim"]
        action_dim = config["action_dim"]
        super(CriticNetwork, self).__init__(
            state_dim + action_dim, config.get("SAC", {})
        )
        self.output_layer = nn.Linear(self.hidden_dim, 1)  # Output layer for Q-value
        self.try_to_load_weights(config.get("uid", None))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            state (torch.Tensor): Input state tensor of shape (batch_size, state_dim).
            action (torch.Tensor): Input action tensor of shape (batch_size, action_dim).

        Returns:
            torch.Tensor: Q-value tensor of shape (batch_size, 1).
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        x = super().forward(x)  # The hidden layers
        q_value = self.output_layer(x)
        return q_value
