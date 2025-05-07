import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from marloes.networks.SAC.actor import ActorNetwork
from marloes.networks.SAC.critic import CriticNetwork
from marloes.networks.SAC.value import ValueNetwork
from marloes.util import timethis


class SAC:
    """
    Soft Actor-Critic (SAC) algorithm for reinforcement learning.
    """

    def __init__(self, config: dict, device: str):
        # Hyperparameters
        self.SAC_config = config.get("SAC", {})
        self.gamma = self.SAC_config.get("gamma", 0.99)  # Discount factor
        self.tau = self.SAC_config.get("tau", 0.005)  # Target network update rate

        # We introduce learnable alpha (Temperature parameter for entropy)
        action_dim = config["action_dim"]
        self.target_entropy = -action_dim  # Target entropy is -action_dim for now
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)

        self.value_network = ValueNetwork(config).to(device)  # Parameterized by psi
        self.target_value_network = ValueNetwork(config).to(
            device
        )  # Parameterized by psi'
        self.critic_1_network = CriticNetwork(config).to(device)
        self.critic_2_network = CriticNetwork(config).to(device)
        self.actor_network = ActorNetwork(config).to(device)  # Parameterized by phi

        # Initialize optimizers
        self._init_optimizers()
        self.target_value_network.load_state_dict(self.value_network.state_dict())

        # Initialize losses
        self._init_losses(config.get("model_updates_per_step", 10))

    def _init_losses(self, model_updates_per_step):
        self.i = 0
        self.loss_value = np.zeros(model_updates_per_step)
        self.loss_critic_1 = np.zeros(model_updates_per_step)
        self.loss_critic_2 = np.zeros(model_updates_per_step)
        self.loss_actor = np.zeros(model_updates_per_step)
        self.alphas = np.zeros(model_updates_per_step)

    def _init_optimizers(self):
        """
        Initialize the optimizers for the networks.
        """
        # Create optimizers here
        actor_lr = self.SAC_config.get("actor_lr", 3e-4)
        critic_lr = self.SAC_config.get("critic_lr", 3e-4)
        value_lr = self.SAC_config.get("value_lr", 3e-4)
        alpha_lr = self.SAC_config.get("alpha_lr", 3e-4)

        eps = self.SAC_config.get("eps", 1e-7)
        weight_decay = self.SAC_config.get("weight_decay", 0.0)

        self.value_optimizer = Adam(
            self.value_network.parameters(),
            lr=value_lr,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.critic1_optimizer = Adam(
            self.critic_1_network.parameters(),
            lr=critic_lr,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.critic2_optimizer = Adam(
            self.critic_2_network.parameters(),
            lr=critic_lr,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.actor_optimizer = Adam(
            self.actor_network.parameters(),
            lr=actor_lr,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)

    def act(self, state):
        """
        Selects an action based on the current state using the actor network.
        """
        self.actor_network.eval()  # Set to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            actions, _ = self.actor_network.sample(state)
        return actions

    def update(self, batch):
        """
        Updates the networks using a batch of experiences.
        """
        # 1. Update the value network
        self._update_value_network(batch)

        # 2. Update the critic networks (using the critic:actor ratio)
        for _ in range(self.SAC_config.get("critic_actor_update_ratio", 2)):
            self._update_critic_networks(batch)

        # 3. Update the actor network (and with it the alpha parameter)
        self.actor_network.train()  # Set back to training mode
        self._update_actor_network(batch)

        # 4. Update the target value network
        self._update_target_value_network()

        self.i += 1

    def _update_value_network(self, batch):
        """
        Updates the value network using the given batch of experiences.
        """
        # Get value estimate from the value network
        V = self.value_network(batch["state"])

        # Sample actions and log probabilities from the actor network
        # To use in the target value: as we are estimating the value under the current policy
        actions, log_pi = self.actor_network.sample(batch["state"])

        # Use minimum of the two critics
        Q_1 = self.critic_1_network(batch["state"], actions)
        Q_2 = self.critic_2_network(batch["state"], actions)
        Q_min = torch.min(Q_1, Q_2)

        # Calculate the target value
        alpha = self.log_alpha.exp()
        target_value = Q_min - alpha * log_pi

        # Calculate the value loss (0.5 scaling from the paper)
        value_loss = 0.5 * F.mse_loss(V, target_value.detach())

        # Back propagate the value loss and update the value network parameters
        self.value_optimizer.zero_grad()
        value_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1.0)
        self.value_optimizer.step()

        self.loss_value[self.i] = value_loss.item()

    def _update_critic_networks(self, batch):
        """
        Update the critic networks (apart from each other) given the batch.
        """
        for i, critic_network in enumerate(
            [self.critic_1_network, self.critic_2_network]
        ):
            # Get the Q value from the critic network
            Q = critic_network(batch["state"], batch["actions"])

            # Assemble the target value
            V_next = self.target_value_network(batch["next_state"])
            target_value = (
                batch["rewards"] + self.gamma * V_next
            )  # No need to add dones as the environment is non-terminal

            # Calculate the critic loss (again, 0.5 scaling from the paper)
            # Use huber loss for stability
            # critic_loss = 0.5 * F.smooth_l1_loss(Q, target_value.detach())
            critic_loss = 0.5 * F.mse_loss(Q, target_value.detach())

            # Back propagate the critic loss and update the critic network parameters
            if i == 0:
                self.critic1_optimizer.zero_grad()
                critic_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.critic_1_network.parameters(), 1.0)
                self.critic1_optimizer.step()
            else:
                self.critic2_optimizer.zero_grad()
                critic_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.critic_2_network.parameters(), 1.0)
                self.critic2_optimizer.step()

            # Store the critic loss
            if i == 0:
                self.loss_critic_1[self.i] = critic_loss.item()
            else:
                self.loss_critic_2[self.i] = critic_loss.item()

    def _update_actor_network(self, batch):
        """
        Update the actor network given the batch.
        """
        # Sample actions and log probabilities from the actor network
        actions, log_pi = self.actor_network.sample(batch["state"])

        # Use minimum of the two critics
        Q_1 = self.critic_1_network(batch["state"], actions)
        Q_2 = self.critic_2_network(batch["state"], actions)
        Q_min = torch.min(Q_1, Q_2)

        # Calculate the actor loss
        alpha = self.log_alpha.exp()
        actor_loss = (alpha * log_pi - Q_min).mean()

        # Back propagate the loss and update parameters
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 0.5)
        self.actor_optimizer.step()

        # Update the alpha parameter; entropy -> target entropy
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.loss_actor[self.i] = actor_loss.item()
        self.alphas[self.i] = self.log_alpha.exp().item()

    def _update_target_value_network(self):
        """
        Update the target value network (polyak averaging).
        """
        for target_param, param in zip(
            self.target_value_network.parameters(), self.value_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
