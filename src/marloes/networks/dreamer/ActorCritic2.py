import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from marloes.networks.util import compute_lambda_returns, symlog
import copy


class ActorCritic2(nn.Module):
    """
    Combines the Actor and Critic networks for learning from abstract trajectories.
    """

    def __init__(
        self,
        input: int,
        output: int,
        hidden_size: int = 64,
        config: dict = {},
        deterministic: bool = False,
    ):
        """
        Initializes the ActorCritic module.

        Args:
            input (int): Dimension of the input (model state).
            output (int): Dimension of the output (action space).
            hidden_size (int, optional): Dimension of the hidden layers. Defaults to 64.
        """
        super(ActorCritic2, self).__init__()
        self.actor = Actor(input, output, config.get("actor_hidden_size", hidden_size))
        self.n_bins = config.get("n_bins", 50)  # Number of bins for the critic
        self.critic = Critic(
            input, config.get("critic_hidden_size", hidden_size), self.n_bins
        )
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_ema_update = config.get("ema_update", 0.98)  # DreamerV3 uses 0.98
        for param in self.critic_target.parameters():
            param.requires_grad = False  # Freeze the target network

        self.actor_optim = AdamW(
            self.actor.parameters(),
            lr=config.get("actor_lr", 1e-4),
            weight_decay=config.get("actor_weight_decay", 0.0),
        )
        self.critic_optim = AdamW(
            self.critic.parameters(),
            lr=config.get("critic_lr", 1e-4),
            weight_decay=config.get("critic_weight_decay", 0.0),
        )

        self.gamma = config.get("gamma", 0.99)  # Discount factor
        self.lmbda = config.get("lambda", 0.95)  # GAE lambda
        self.entropy_coef = config.get("entropy_coef", 0.01)  # Entropy coefficient
        self.actor_clip_grad = config.get("actor_clip_grad", 0.5)
        self.critic_clip_grad = config.get("critic_clip_grad", 0.5)
        self.beta_weights = {"val": 1.0, "repval": 0.3}

        self.deterministic = (
            deterministic  # Whether to use deterministic actions during training
        )

        # EMA for S
        self.register_buffer("s_ema", torch.zeros(1))  # Initialize S as a zero tensor
        self.s_ema_alpha = config.get("s_ema_alpha", 0.98)  # EMA decay for S

        # Store losses
        self.reset_losses()

    def reset_losses(self):
        """
        Resets the actor and critic losses.
        """
        if not hasattr(self, "i"):
            self.i = 0
        else:
            self.i += 1
        if self.i % 5 == 0:
            self.actor_loss = []

        self.critic_loss = []

    def act(
        self, model_state: torch.Tensor, deterministic: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts actions using the Actor network.

        Args:
            model_state (torch.Tensor): Current model state.

        Returns:
            torch.Tensor: Predicted actions.
        """
        return self.actor.sample(model_state, deterministic)

    def learn(self, trajectories: list) -> dict[str, torch.Tensor]:
        # Unpack batched tensors
        states = torch.stack([t["state"] for t in trajectories], dim=0).squeeze()
        actions = torch.stack([t["actions"] for t in trajectories], dim=0).squeeze(2)
        rewards = torch.stack([t["rewards"] for t in trajectories], dim=0).squeeze(-1)

        # Compute targets with frozen critic
        with torch.no_grad():
            _, target_values = self.critic_target(states)
            returns, advantages = self._compute_advantages(rewards, target_values)

        # Actor update only when i % 5 == 0
        # if self.i % 5 == 0 and self.i > 0:
        actor_loss = self._compute_actor_loss(states, actions, advantages, returns)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_clip_grad)
        self.actor_optim.step()
        self.actor_loss.append(actor_loss.item())
        # if self.i == 0:
        #     self.actor_loss = [0]  # Reset actor loss on first iteration

        # Critic update using two-hot encoded cross-entropy
        B, T, D = states.shape
        states_flat = states.view(B * T, D)
        returns_flat = returns.view(B * T)
        logits_flat, _ = self.critic(states_flat)
        critic_loss = self._compute_critic_loss(logits_flat, returns_flat)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_clip_grad)
        self.critic_optim.step()
        self.critic_loss.append(critic_loss.item())

        # EMA target update
        # if self.i % 5 == 0 and self.i > 0:
        self.update_critic_target()

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
        mu, log_std = self.actor.forward(states)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        # invert tanh to get pre-tanh values
        eps = 1e-6  # small epsilon to avoid numerical issues
        actions = torch.clamp(actions, -1 + eps, 1 - eps).detach()
        # invert tanh to get pre-tanh values
        pre_tanh = 0.5 * (torch.log1p(actions) - torch.log1p(-actions))
        log_prob = dist.log_prob(pre_tanh).sum(
            dim=-1, keepdim=True
        )  # log probability of actions
        # correction
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        entropy = dist.entropy().sum(
            dim=-1, keepdim=True
        )  # entropy of the distribution

        # Scale factor S=EMA( Per(returns, 95) - Per(returns, 5) )
        # where Per(x, p) is the p-th percentile of x (detaching to prevent gradient flow)
        flat_returns = returns.detach().view(-1)
        # Standardize returns
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

        # Initialize EMA on first nonzero raw_S
        if self.s_ema.item() == 0.0:
            # fill_ keeps it as a buffer
            self.s_ema.fill_(S)
        else:
            # in-place EMA update: preserves buffer registration
            self.s_ema.mul_(self.s_ema_alpha).add_((1 - self.s_ema_alpha) * S)

        # Reduce entropy coefficient over time
        if not hasattr(self, "entropy_decay"):
            self.entropy_decay = self.entropy_coef  # Initialize entropy decay
        self.entropy_decay *= 0.999  # Decay factor (adjust as needed)

        scale_factor = torch.maximum(
            self.s_ema, torch.tensor(1.0, device=self.s_ema.device)
        )
        weighted_advantages = (advantages.detach() / scale_factor) * log_prob
        entropy_term = entropy * self.entropy_decay
        actor_loss = -(weighted_advantages.mean() + entropy_term.mean())
        return actor_loss

    def _compute_critic_loss(
        self, logits: torch.Tensor, target_returns: torch.Tensor
    ) -> torch.Tensor:
        """
        Critic loss using two-hot cross-entropy between predicted logits and soft targets.

        Args:
            logits (torch.Tensor): [N, n_bins] raw outputs from critic.
            target_returns (torch.Tensor): [N] continuous returns.

        Returns:
            torch.Tensor: scalar loss.
        """
        twohot = self._twohot_encode(target_returns, self.critic.bins)
        logp = F.log_softmax(logits, dim=-1)
        loss = -(twohot * logp).sum(dim=-1).mean()
        return loss

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
            target_param.data.mul_(self.critic_ema_update).add_(
                param.data * (1.0 - self.critic_ema_update)
            )

    def _twohot_encode(self, target: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
        """
        Converts a tensor of target values into a two-hot encoded tensor based on the provided bins.

        Args:
            target (torch.Tensor): Tensor containing continuous target values. [B,1]
            bins (torch.Tensor): Tensor containing the bins (sorted ascending)

        Returns:
            torch.Tensor: Tensor of shape [B, bins.size(0)] with two weights
        """
        # flatten
        y = target.view(-1).contiguous()  # make sure y is contiguous in memory
        # For each y, find the bin indices k where bins[k] <= y < bins[k+1]
        idx = torch.bucketize(y, bins)  # gives insertion index in [1..n_bins]
        idx_lo = torch.clamp(idx - 1, 0, bins.size(0) - 1)
        idx_hi = torch.clamp(idx, 0, bins.size(0) - 1)

        # Gather bin values
        b_lo = bins[idx_lo]
        b_hi = bins[idx_hi]

        # Compute weights
        # if b_hi == b_lo (edge cases), put all mass on lo
        denom = (b_hi - b_lo).clamp(min=1e-8)
        w_hi = ((y - b_lo) / denom).clamp(0.0, 1.0)
        w_lo = 1.0 - w_hi

        twohot = torch.zeros(
            y.size(0), bins.size(0), device=y.device, dtype=torch.float32
        )
        twohot.scatter_(1, idx_lo.unsqueeze(1), w_lo.unsqueeze(1))
        twohot.scatter_(1, idx_hi.unsqueeze(1), w_hi.unsqueeze(1))
        return twohot


class Actor(nn.Module):
    """
    Actor network for predicting continuous actions.

    Now uses two shared layers, then `output_size` separate heads:
    each head has one hidden layer (`hidden_size`) and a single output node.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # dropout layer
        self.dropout = nn.Dropout(p=0.1)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.output_size = output_size
        self.hidden_size = hidden_size

        # one head-per-action for the mean
        self.heads_mean_hidden = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(output_size)]
        )
        self.heads_mean_out = nn.ModuleList(
            [nn.Linear(hidden_size, 1) for _ in range(output_size)]
        )

        # one head-per-action for the log_std
        self.heads_log_std_hidden = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(output_size)]
        )
        self.heads_log_std_out = nn.ModuleList(
            [nn.Linear(hidden_size, 1) for _ in range(output_size)]
        )

        # initialize the weights of log_std output layers to encourage exploration
        for out in self.heads_log_std_out:
            out.weight.data.fill_(-0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass:
         - two shared layers
         - per-head hidden layer + output node
        Returns:
         - mu:      [batch, output_size]
         - log_std: [batch, output_size]
        """
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.dropout(x)

        mus = []
        log_stds = []
        for i in range(self.output_size):
            h_mu = F.relu(self.heads_mean_hidden[i](x))
            mu_i = self.heads_mean_out[i](h_mu)
            mus.append(mu_i)

            h_ls = F.relu(self.heads_log_std_hidden[i](x))
            ls_i = self.heads_log_std_out[i](h_ls)
            log_stds.append(ls_i)

        mu = torch.cat(mus, dim=-1)
        log_std = torch.cat(log_stds, dim=-1)
        log_std = torch.clamp(log_std, -2, 2)
        # Replace NaN and inf values with specific values
        mu = torch.nan_to_num(mu, nan=0.0, posinf=5.0, neginf=-5.0)
        log_std = torch.nan_to_num(log_std, nan=-1.0, posinf=2.0, neginf=-2.0)

        return mu, log_std

    def sample(
        self, x, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples actions using the reparameterization trick,
        now using the per-head outputs.
        """
        mu, log_std = self.forward(x)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)

        if deterministic:
            pre_tanh = mu
            actions = torch.tanh(mu)
        else:
            pre_tanh = dist.rsample()
            actions = torch.tanh(pre_tanh)

        log_prob = dist.log_prob(pre_tanh).sum(-1, keepdim=True)
        correction = torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        log_prob = log_prob - correction

        return actions, log_prob


class Critic(nn.Module):
    """
    Critic network for predicting the value of the current state.
    (unchanged)
    """

    def __init__(self, input_size: int, hidden_size: int = 256, n_bins: int = 50):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, n_bins)
        self.register_buffer("bins", torch.linspace(-20, 20, n_bins))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_out(x)
        probs = F.softmax(logits, dim=-1)
        value = (probs * self.bins).sum(dim=-1, keepdim=True)
        return logits, value
