import torch
import torch.nn.functional as F
from torch.optim import Adam

from marloes.networks.SAC.actor import ActorNetwork
from marloes.networks.SAC.critic import CriticNetwork
from marloes.networks.SAC.value import ValueNetwork


class SAC:
    """
    Soft Actor-Critic (SAC) algorithm for reinforcement learning.
    """

    def __init__(self, config: dict):
        self.config = config
        self.value_network = ValueNetwork(config)  # Parameterized by psi
        self.target_value_network = ValueNetwork(config)  # Parameterized by psi'
        self.critic_1_network = CriticNetwork(config)
        self.critic_2_network = CriticNetwork(config)
        self.actor_network = ActorNetwork(config)  # Parameterized by phi

        # Initialize optimizers
        self._init_optimizers()
        self.target_value_network.load_state_dict(self.value_network.state_dict())

        # Store losses
        self.loss_values = []
        self.loss_critic_1 = []
        self.loss_critic_2 = []
        self.loss_actor = []

        self.gamma = config.get("gamma", 0.99)  # Discount factor
        self.alpha = config.get("alpha", 0.2)  # Temperature parameter for entropy
        self.tau = config.get("tau", 0.005)  # Target network update rate

    def _init_optimizers(self):
        """
        Initialize the optimizers for the networks.
        """
        # Create optimizers here
        learning_rate = self.config.get("learning_rate", 3e-4)
        eps = self.config.get("eps", 1e-7)
        weight_decay = self.config.get("weight_decay", 0.0)

        self.value_optimizer = Adam(
            self.value_network.parameters(),
            lr=learning_rate,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.critic1_optimizer = Adam(
            self.critic_1_network.parameters(),
            lr=learning_rate,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.critic2_optimizer = Adam(
            self.critic_2_network.parameters(),
            lr=learning_rate,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.actor_optimizer = Adam(
            self.actor_network.parameters(),
            lr=learning_rate,
            eps=eps,
            weight_decay=weight_decay,
        )

    def act(self, state):
        """
        Selects an action based on the current state using the actor network.
        """
        self.actor_network.eval()  # Set to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            action, _ = self.actor_network.sample(state)
        return action

    def update(self, batch):
        """
        Updates the networks using a batch of experiences.
        """
        # TODO: convert batch to tensor (batch_size, state_dim)

        # 1. Update the value network
        self._update_value_network(batch)

        # 2. Update the critic networks
        self._update_critic_networks(batch)

        # 3. Update the actor network
        self._update_actor_network(batch)

        # 4. Update the target value network
        self._update_target_value_network()

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
        target_value = Q_min - self.alpha * log_pi

        # Calculate the value loss (0.5 scaling from the paper)
        value_loss = 0.5 * F.mse_loss(V, target_value.detach())

        # Back propagate the value loss and update the value network parameters
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.loss_values.append(value_loss.item())

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
            critic_loss = 0.5 * F.mse_loss(Q, target_value.detach())

            # Back propagate the critic loss and update the critic network parameters
            if i == 0:
                self.critic1_optimizer.zero_grad()
                critic_loss.backward()
                self.critic1_optimizer.step()
            else:
                self.critic2_optimizer.zero_grad()
                critic_loss.backward()
                self.critic2_optimizer.step()

            # Store the critic loss
            if i == 0:
                self.loss_critic_1.append(critic_loss.item())
            else:
                self.loss_critic_2.append(critic_loss.item())

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
        actor_loss = (self.alpha * log_pi - Q_min).mean()

        # Back propagate the loss and update parameters
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.loss_actor.append(actor_loss.item())

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
