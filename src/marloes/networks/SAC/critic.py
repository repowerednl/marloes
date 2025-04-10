import torch
import torch.nn as nn
from marloes.networks.SAC.base import SACBaseNetwork


class CriticNetwork(SACBaseNetwork):
    """
    Critic network (Q-function) for the Soft Actor-Critic (SAC) algorithm.
    Input should be the state and action.
    The output is the Q-value.
    """

    def __init__(self, config: dict):
        # The input dimension needs to include both the state and action dimensions
        state_dim = config["state_dim"]
        action_dim = config["action_dim"]
        super(CriticNetwork, self).__init__(state_dim + action_dim, config)
        self.output_layer = nn.Linear(self.hidden_dim, 1)  # Output layer for Q-value

    def forward(self, state, action):
        """
        Forward pass through the network.
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        x = super().forward(x)  # The hidden layers
        q_value = self.output_layer(x)
        return q_value
