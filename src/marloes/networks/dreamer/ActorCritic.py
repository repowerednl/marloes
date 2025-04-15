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
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-4)

        self.gamma = 0.997
        self.lmbda = 0.95
        self.entropy_coef = 0.0003
        self.beta_weights = {"val": 1.0, "repval": 0.3}

        # Store losses
        self.actor_loss = []
        self.critic_loss = []

    def act(self, model_state: torch.Tensor) -> torch.Tensor:
        """
        Returns the actions predicted by the Actor network.
        """
        return self.actor(model_state).sample()

    def learn(self, trajectories: list) -> dict[str, torch.Tensor]:
        """
        Learning step for the ActorCritic network.
        Trajectories is a 'batch' of trajectories, each containing:
        - states
        - actions
        - rewards
        """
        # Unpack the states into a batched tensor for the critic
        states = torch.stack([t["states"] for t in trajectories], dim=0)
        # Unpack the actions into a batched tensor for the loss computing
        actions = torch.stack([t["actions"] for t in trajectories], dim=0)
        # Unpack the rewards into a batched tensor for the loss computing
        rewards = torch.stack([t["rewards"] for t in trajectories], dim=0)

        # Critic Evaluation
        values = self.critic(states)

        # Compute the advantages (lambda-returns)
        returns, advantages = self._compute_advantages(rewards, values)

        # Compute the actor and critic losses
        actor_loss = self._compute_actor_loss(states, actions, advantages, returns)
        critic_loss = self._compute_critic_loss(values, returns)

        # Backpropagate the losses
        total_loss = actor_loss + critic_loss

        # Actor loss
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.actor_loss.append(actor_loss.item())

        # Critic loss
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.critic_loss.append(critic_loss.item())

        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "total_loss": total_loss,
        }

    def _compute_actor_loss(self, states, actions, advantages, returns) -> torch.Tensor:
        """
        Computes the actor loss.
        Term 1: policy gradient: the negative log probability of the taken actions, weighted by the advantage.
        Term 2: entropy bonus: encourages exploration by adding a small penalty to the log probability of the actions.
        """
        policy_dist = self.actor(states)
        log_probs = policy_dist.log_prob(actions)

        entropy = policy_dist.entropy().mean()

        # Scale factor S=EMA( Per(returns, 95) - Per(returns, 5) )
        # where Per(x, p) is the p-th percentile of x (detaching to prevent gradient flow)
        flat_returns = returns.detach().view(-1)
        # Compute the 95th and 5th percentiles
        quantile_95 = torch.quantile(flat_returns, 0.95)
        quantile_5 = torch.quantile(flat_returns, 0.05)
        S = torch.clamp(
            quantile_95 - quantile_5,
            min=0.99,
        )
        # TODO: EMA is the exponential moving average.

        actor_loss = (
            -((advantages.detach() / S) * log_probs).mean()
            - self.entropy_coef * entropy
        )
        return actor_loss

    def _compute_critic_loss(
        self, values: torch.Tensor, returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the critic loss.
        First implementation: simple MSE loss.
        """
        critic_loss = F.mse_loss(values, returns.detach())
        return critic_loss

    def _compute_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Uses the lambda-returns to compute the advantages.

        Expects:
        - rewards: Tensor of shape (B, T, 1)
        - values: Tensor of shape (B, T, 1)

        Returns:
        - returns: λ-returns, tensor of shape (B, T, 1)
        - advantages: returns - values.detach(), tensor of shape (B, T, 1)
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
        mu = self.fc_mean(x)
        print("\nmu:", mu)
        mu = torch.tanh(mu)
        std = torch.exp(self.log_std)
        print("mu:", mu)
        print("std:", std)
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
