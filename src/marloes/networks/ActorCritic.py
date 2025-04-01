import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from marloes.networks.util import compute_lambda_returns


class ActorCritic:
    """
    This ActorCritic module is based on the DreamerV3 architecture.
    The Actor and Critic networks are combined here, and learn from abstract trajectories or representations (latent states) predicted by the WorldModel.
    - The Actor and Critic operate on model states, s_t = {h_t, z_t}.
    - The Actor aims to maximize return with gamma-discounted rewards (gamma = 0.997).
    - The Critic aims to predict the value of the current state.
    """

    def __init__(self, input: int, output: int, hidden_size: int = 64):
        """
        Initializes the ActorCritic network.
        """
        self.actor = Actor(input, output, hidden_size)
        self.critic = Critic(input, hidden_size)

        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = 0.997
        self.lmbda = 0.95
        self.entropy_coef = 0.005
        self.beta_weights = {"val": 1.0, "repval": 0.3}

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns the actions predicted by the Actor network.
        """
        return self.actor(obs).sample()

    def learn(self, trajectories: dict) -> dict[str, torch.Tensor]:
        """
        Learning step for the ActorCritic network.
        """
        # Unpack the trajectories
        states = trajectories["states"]
        actions = trajectories["actions"]
        rewards = trajectories["rewards"]

        # Critic Evaluation
        values = self.critic(states).squeeze(-1)

        # Compute the advantages (lambda-returns)
        returns, advantages = self._compute_advantages(rewards, values)

        # Compute the actor and critic losses
        actor_loss = self._compute_actor_loss(states, actions, advantages)
        critic_loss = self._compute_critic_loss(values, returns)

        # Backpropagate the losses
        total_loss = actor_loss + critic_loss

        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        total_loss.backward()
        self.actor_optim.step()
        self.critic_optim.step()

        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "total_loss": total_loss,
        }

    def _compute_actor_loss(self, states, actions, advantages) -> torch.Tensor:
        """
        Computes the actor loss.
        """
        dist = self.actor(states)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().mean()
        actor_loss = -(log_probs * advantages).mean() - self.entropy_coef * entropy
        return actor_loss

    def _compute_critic_loss(
        self, values: torch.Tensor, returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the critic loss.
        """
        critic_loss = F.mse_loss(values, returns.detach())
        return critic_loss

    def _compute_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """
        Uses the lambda-returns to compute the advantages.
        """
        returns = compute_lambda_returns(rewards, values, self.gamma, self.lmbda)
        advantages = returns - values.detach()
        return returns, advantages


class Actor(nn.Module):
    """
    Actor class, MLP network with hidden layers, predicts the 'continuous' actions per agent. TODO: Discrete actions.
    Produces Gaussian policy over continuous actions.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 256):
        """
        Initializes the Actor network.
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, output_size)
        self.log_std = nn.Parameter(torch.zeros(output_size))
        # initialize the weights of log_std with a small negative value to encourage exploration
        nn.init.constant_(self.log_std, -0.5)

    def forward(self, x):
        """
        Forward pass through the Actor network.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Apply tanh bound to keep actions in [-1, 1]
        mu = torch.tanh(self.fc_mean(x))
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)


class Critic(nn.Module):
    """
    Critic class, MLP network with hidden layers, predicts the value of the current state.
    """

    def __init__(self, input_size: int, hidden_size: int = 256):
        """
        Initializes the Critic network.
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass through the Critic network.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_value(x)
