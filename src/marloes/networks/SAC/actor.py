import torch
import torch.nn as nn

from marloes.networks.SAC.base import SACBaseNetwork


class ActorNetwork(SACBaseNetwork):
    """
    Actor network (policy) for the Soft Actor-Critic (SAC) algorithm.

    This network is used for generating actions for the agents based on the input state.
    It outputs the mean and log standard deviation (log_std) of a Gaussian distribution,
    which is used to sample actions stochastically.

    Attributes:
        mean_layer (nn.Linear): Layer to output the mean of the Gaussian distribution.
        log_std_layer (nn.Linear): Layer to output the log standard deviation of the Gaussian distribution.
        log_std_min (float): Minimum value for log standard deviation.
        log_std_max (float): Maximum value for log standard deviation.
    """

    def __init__(self, config: dict):
        """
        Initializes the ActorNetwork with the given configuration.

        Args:
            config (dict): Configuration dictionary, containing the following keys:
                - "state_dim" (int): Dimension of the input state.
                - "action_dim" (int): Dimension of the output action.
                - "SAC" (dict): SAC-specific configuration, including:
                    - "log_std_min" (float, optional): Minimum value for log_std (default: -20).
                    - "log_std_max" (float, optional): Maximum value for log_std (default: 2).
        """
        state_dim = config["state_dim"]
        if config["dyna"].get("sCTCE", False):
            action_dim = 1
        else:
            action_dim = config["action_dim"]
        SAC_config = config.get("SAC", {})
        super(ActorNetwork, self).__init__(
            state_dim, SAC_config, hidden_dim=SAC_config.get("actor_hidden_dim", None)
        )

        # Separate heads for mean and log_std
        self.mean_layer = nn.Linear(self.hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(self.hidden_dim, action_dim)

        # Clamping range to prevent the policy from becoming too deterministic or too random
        # Extremely high or low log_std values mean either very wide or very narrow Gaussian distributions
        # So either almost deterministic (log_std -> -inf) or almost uniform (log_std -> inf)
        self.log_std_min = SAC_config.get("log_std_min", -20)
        self.log_std_max = SAC_config.get("log_std_max", 2)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the network.

        Args:
            state (torch.Tensor): Input state tensor of shape (batch_size, state_dim).

        Returns:
            tuple: A tuple containing:
                - mean (torch.Tensor): Mean of the Gaussian distribution for actions.
                - log_std (torch.Tensor): Log standard deviation of the Gaussian distribution for actions.
        """
        x = super().forward(state)  # The hidden layers
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy given the state.
        Using the reparameterization trick, as described in the original SAC paper.

        Reparameterization trick allows gradients to flow through the sampling process.
        Or intuitively: By sampling by extracting the randomness into a separate term (ε),
        we keep the action stochastic, but make it possible to learn the mean and std via gradients.

        Args:
            state (torch.Tensor): Input state tensor of shape (batch_size, state_dim).

        Returns:
            tuple: A tuple containing:
                - actions (torch.Tensor): Sampled actions from the policy.
                - log_probs (torch.Tensor): Log probabilities of the sampled actions.

        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)  # Standard deviation is the exponent of log_std
        normal = torch.distributions.Normal(mean, std)
        # rsample uses the reparameterization trick to sample from the distribution
        x = normal.rsample()

        # We want actions to be in range [-1, 1], so we use tanh to squash the output
        actions = torch.tanh(x)

        # Also adjust the log probability for the squashing
        # Log of the derivative of the tanh function tells us how to adjust the log probability
        log_prob_x = normal.log_prob(x).sum(dim=-1, keepdim=True)
        log_prob_correction = torch.log(1 - actions.pow(2) + 1e-6).sum(
            dim=-1, keepdim=True
        )  # 1e-6 is a small constant to prevent log(0)
        log_probs = log_prob_x - log_prob_correction

        return actions, log_probs
