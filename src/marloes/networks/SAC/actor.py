import torch
import torch.nn as nn
from marloes.networks.SAC.base import SACBaseNetwork


class ActorNetwork(SACBaseNetwork):
    """
    Actor network (policy) for the Soft Actor-Critic (SAC) algorithm.
    Input should be the state.
    The output is the action for each agent, through a Gaussian distribution: mean and log_std.
    """

    def __init__(self, config: dict):
        state_dim = config["state_dim"]
        action_dim = config["action_dim"]
        super(ActorNetwork, self).__init__(state_dim, config)

        # Separate heads for mean and log_std
        self.mean_layer = nn.Linear(self.hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(self.hidden_dim, action_dim)

        # Clamping range to prevent the policy from becoming too deterministic or too random
        self.log_std_min = config.get("log_std_min", -20)
        self.log_std_max = config.get("log_std_max", 2)

    def forward(self, state):
        """
        Forward in this network only returns the mean and log_std of the Gaussian.
        """
        x = super().forward(state)  # The hidden layers
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)

        # Clamp log_std to prevent extreme values
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state):
        """
        Sample actions from the policy given the state.
        Using the reparameterization trick, as described in the original SAC paper.
        """
        # We are using rsample, which uses the reparameterization trick, so conversion is needed
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()

        # We want actions to be in range [-1, 1], so we use tanh to squash the output
        actions = torch.tanh(x)

        ## We need to adjust the log probability for the squashing
        # 1: Compute the log probability of x under the original Gaussian.
        log_prob_x = normal.log_prob(x).sum(dim=-1, keepdim=True)

        # 2: Compute the correction term for the tanh squashing
        # Plus, add a small constant (1e-6) for numerical stability
        log_prob_correction = torch.log(1 - actions.pow(2) + 1e-6).sum(
            dim=-1, keepdim=True
        )

        # 3: Subtract the correction term from the original log probability.
        log_probs = log_prob_x - log_prob_correction

        return actions, log_probs
