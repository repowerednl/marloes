import torch
import torch.nn as nn


class WorldDynamicsModel(nn.Module):
    """
    World dynamics model that predicts the next state and reward given the current state and action.
    """

    def __init__(self, world_model_config: dict, action_dim: int):
        super().__init__()
        world_enc_dim = world_model_config.get("world_enc_dim", 64)
        scalar_dim = world_model_config.get("scalar_dim", 5)
        num_agents = action_dim  # Every agent has its own action
        # Only predict the next state for each agent
        # We don't need to predict forecast
        # TODO: Explain this in the paper
        next_state_dim = scalar_dim * num_agents

        hidden_size = 128
        self.shared_net = nn.Sequential(
            nn.Linear(world_enc_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Head for next-state and one for reward
        self.next_state_head = nn.Linear(hidden_size, next_state_dim)
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, world_state, actions):
        """
        Forward pass through the world dynamics model.
        """
        x = torch.cat([world_state, actions], dim=-1)
        hidden = self.shared_net(x)

        next_state_pred = self.next_state_head(hidden)
        reward_pred = self.reward_head(hidden)
        return next_state_pred, reward_pred
