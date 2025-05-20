import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from marloes.networks.util import compute_lambda_returns
import copy


class ActorCritic:
    """
    Combines the Actor and Critic networks for learning from abstract trajectories.
    """

    def __init__(
        self, input: int, output: int, hidden_size: int = 64, config: dict = {}
    ):
        """
        Initializes the ActorCritic module.

        Args:
            input (int): Dimension of the input (model state).
            output (int): Dimension of the output (action space).
            hidden_size (int, optional): Dimension of the hidden layers. Defaults to 64.
        """
        self.actor = Actor(input, output, hidden_size)
        self.critic = Critic(input, hidden_size)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_ema_decay = config.get("ema_decay", 0.98)  # DreamerV3 uses 0.98
        for param in self.critic_target.parameters():
            param.requires_grad = False  # Freeze the target network

        self.actor_optim = Adam(
            self.actor.parameters(),
            lr=config.get("actor_lr", 1e-4),
            weight_decay=config.get("actor_weight_decay", 0.0),
        )
        self.critic_optim = Adam(
            self.critic.parameters(),
            lr=config.get("critic_lr", 1e-4),
            weight_decay=config.get("critic_weight_decay", 0.0),
        )

        self.gamma = config.get("gamma", 0.99)  # Discount factor
        self.lmbda = config.get("lambda", 0.95)  # GAE lambda
        self.entropy_coef = config.get("entropy_coef", 0.01)  # Entropy coefficient
        self.beta_weights = {"val": 1.0, "repval": 0.3}

        # Store losses
        self.reset_losses()

    def reset_losses(self):
        """
        Resets the actor and critic losses.
        """
        self.actor_loss = []
        self.critic_loss = []

    def act(self, model_state: torch.Tensor) -> torch.Tensor:
        """
        Predicts actions using the Actor network.

        Args:
            model_state (torch.Tensor): Current model state.

        Returns:
            torch.Tensor: Predicted actions.
        """
        return self.actor(model_state).sample()

    def learn(self, trajectories: list) -> dict[str, torch.Tensor]:
        """
        Performs a learning step for the ActorCritic module.

        Args:
            trajectories (list): Batch of trajectories.

        Returns:
            dict: Dictionary containing actor, critic, and total losses.
        """
        # TODO: not ideal to squeeze() afterwards, but stack adds a dimension.
        # Unpack the states into a batched tensor for the critic
        states = torch.stack([t["states"] for t in trajectories], dim=0).squeeze()
        # Unpack the actions into a batched tensor for the loss computing
        actions = torch.stack([t["actions"] for t in trajectories], dim=0).squeeze()
        # Unpack the rewards into a batched tensor for the loss computing
        rewards = torch.stack([t["rewards"] for t in trajectories], dim=0).squeeze(-1)

        # Obtaining the target values from the frozen critic target (v3)
        with torch.no_grad():
            # Use the critic target to compute the values
            target_values = self.critic_target(states)

            # Compute the advantages (lambda-returns)
            returns, advantages = self._compute_advantages(rewards, target_values)

        # Compute the values using the critic
        values = self.critic(states)

        # Compute the actor and critic losses
        actor_loss = self._compute_actor_loss(states, actions, advantages, returns)
        critic_loss = self._compute_critic_loss(values, returns)

        # Actor loss
        self.actor_optim.zero_grad()
        actor_loss.backward()
        # add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optim.step()

        self.actor_loss.append(actor_loss.item())

        # Critic loss
        self.critic_optim.zero_grad()
        critic_loss.backward()
        # add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()

        # Update the network with EMA
        self.update_critic_target()

        self.critic_loss.append(critic_loss.item())

    def _compute_actor_loss(self, states, actions, advantages, returns) -> torch.Tensor:
        """
        Computes the actor loss.

        Term 1: policy gradient: the negative log probability of the taken actions, weighted by the advantage.
        Term 2: entropy bonus: encourages exploration by adding a small penalty to the log probability of the actions.

        Args:
            states (torch.Tensor): States tensor.
            actions (torch.Tensor): Actions tensor.
            advantages (torch.Tensor): Advantages tensor.
            returns (torch.Tensor): Returns tensor.

        Returns:
            torch.Tensor: Computed actor loss.
        """
        policy_dist = self.actor(states)
        log_probs = policy_dist.log_prob(actions)

        entropy = policy_dist.entropy().mean()

        # Scale factor S=EMA( Per(returns, 95) - Per(returns, 5) )
        # where Per(x, p) is the p-th percentile of x (detaching to prevent gradient flow)
        flat_returns = returns.detach().view(-1)

        # # Standardize returns
        flat_returns = (flat_returns - flat_returns.mean()) / (
            flat_returns.std() + 1e-8
        )

        # Compute the 95th and 5th percentiles
        quantile_95 = torch.quantile(flat_returns, 0.95)
        quantile_5 = torch.quantile(flat_returns, 0.05)
        S = torch.clamp(
            quantile_95 - quantile_5,
            min=0.99,
        )
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

        Args:
            values (torch.Tensor): Predicted values tensor.
            returns (torch.Tensor): Returns tensor.

        Returns:
            torch.Tensor: Computed critic loss.
        """
        critic_loss = F.mse_loss(values, returns)
        return critic_loss

    def _compute_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Uses the lambda-returns to compute the advantages.

        Expects:
            rewards (torch.Tensor): Tensor of shape (B, T, 1).
            values (torch.Tensor): Tensor of shape (B, T, 1).

        Returns:
            tuple: λ-returns tensor of shape (B, T, 1) and advantages tensor of shape (B, T, 1).
        """
        returns = compute_lambda_returns(rewards, values, self.gamma, self.lmbda)
        advantages = returns - values
        return returns, advantages

    @torch.no_grad()
    def update_critic_target(self):
        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.mul_(self.critic_ema_decay).add_(
                param.data * (1.0 - self.critic_ema_decay)
            )


class Actor(nn.Module):
    """
    Actor network for predicting continuous actions.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 256):
        """
        Initializes the Actor network.

        Args:
            input_size (int): Dimension of the input.
            output_size (int): Dimension of the output (action space).
            hidden_size (int, optional): Dimension of the hidden layers. Defaults to 256.
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

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.distributions.Normal: Gaussian policy distribution.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Apply tanh bound to keep actions in [-1, 1]
        mu = self.fc_mean(x)
        mu = torch.tanh(mu)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)


class Critic(nn.Module):
    """
    Critic network for predicting the value of the current state.
    """

    def __init__(self, input_size: int, hidden_size: int = 256):
        """
        Initializes the Critic network.

        Args:
            input_size (int): Dimension of the input.
            hidden_size (int, optional): Dimension of the hidden layers. Defaults to 256.
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass through the Critic network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted value of the state.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_value(x)
