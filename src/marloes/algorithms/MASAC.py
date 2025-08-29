import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from marloes.networks.SAC.actor import ActorNetwork
from marloes.networks.SAC.critic import CriticNetwork
from marloes.networks.SAC.value import ValueNetwork
from marloes.util import timethis


class MultiAgentSAC:
    """
    SAC with one ActorNetwork per agent, but shared central critic + value.
    Adhering to the sCTCE (sequential centralized training with centralized execution) paradigm.
    This is a multi-agent version of the Soft Actor-Critic (SAC) algorithm.
    """

    def __init__(self, config: dict, device: str):
        self.n = config[
            "action_dim"
        ]  # Number of agents is equal to action_dim, because continuous
        self.action_dim = config["action_dim"]
        self.device = device

        # Hyperparameters
        self.SAC_config = config.get("SAC", {})
        self.gamma = self.SAC_config.get("gamma", 0.99)  # Discount factor
        self.tau = self.SAC_config.get("tau", 0.005)  # Target network update rate

        # We introduce learnable alpha (Temperature parameter for entropy)
        joint_action_dim = self.action_dim * self.n
        self.target_entropy = -joint_action_dim  # Target entropy is -action_dim for now
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)

        self.value_network = ValueNetwork(config).to(device)  # Parameterized by psi
        self.target_value_network = ValueNetwork(config).to(
            device
        )  # Parameterized by psi'
        self.critic_1_network = CriticNetwork(config).to(device)
        self.critic_2_network = CriticNetwork(config).to(device)

        # NB: Separate actor networks for each agent
        self.actors = torch.nn.ModuleList(
            [ActorNetwork(config).to(device) for _ in range(self.n)]
        )

        # Initialize optimizers
        self._init_optimizers()
        self.target_value_network.load_state_dict(self.value_network.state_dict())

        # Initialize losses
        self._init_losses(config.get("model_updates_per_step", 10))

        self._try_to_load_weights(config.get("uid", None))

    def _init_losses(self, model_updates_per_step):
        self.i = 0
        self.loss_value = np.zeros(model_updates_per_step)
        self.loss_critic_1 = np.zeros(model_updates_per_step)
        self.loss_critic_2 = np.zeros(model_updates_per_step)
        self.loss_actor = np.zeros((model_updates_per_step, self.n))
        self.alphas = np.zeros(model_updates_per_step)
        self.mean_q = np.zeros(model_updates_per_step)

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
        self.critic_1_optimizer = Adam(
            self.critic_1_network.parameters(),
            lr=critic_lr,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.critic_2_optimizer = Adam(
            self.critic_2_network.parameters(),
            lr=critic_lr,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.actor_optimizers = [
            Adam(actor.parameters(), lr=actor_lr, eps=eps, weight_decay=weight_decay)
            for actor in self.actors
        ]

        self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)

    def act(self, state, deterministic=False):
        """
        Selects an action based on the current state using the actor network.
        """
        with torch.no_grad():
            if deterministic:
                means = [actor(state)[0] for actor in self.actors]
                actions = [torch.tanh(mean) for mean in means]
            else:
                actions = [actor.sample(state)[0] for actor in self.actors]
        return torch.cat(actions, dim=-1)

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
        self._update_actor_networks(batch)

        # 4. Update alpha
        self._update_alpha(batch)

        # 5. Update the target value network
        self._update_target_value_network()

        self.i += 1

    def _update_value_network(self, batch):
        """
        Updates the value network using the given batch of experiences.
        """
        # Get value estimate from the value network
        V = self.value_network(batch["state"])

        # Sample joint actions and log probabilities from the actor networks
        with torch.no_grad():
            actions, log_pi = self._sample_joint_action(batch["state"])

        # Use minimum of the two critics
        Q_1 = self.critic_1_network(batch["state"], actions)
        Q_2 = self.critic_2_network(batch["state"], actions)
        Q_min = torch.min(Q_1, Q_2)
        self.mean_q[self.i] = Q_min.mean().item()

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
                self.critic_1_optimizer.zero_grad()
                critic_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.critic_1_network.parameters(), 1.0)
                self.critic_1_optimizer.step()
            else:
                self.critic_2_optimizer.zero_grad()
                critic_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.critic_2_network.parameters(), 1.0)
                self.critic_2_optimizer.step()

            # Store the critic loss
            if i == 0:
                self.loss_critic_1[self.i] = critic_loss.item()
            else:
                self.loss_critic_2[self.i] = critic_loss.item()

    def _update_actor_networks(self, batch):
        """
        Update the actor networks given the batch.
        """
        # Loop over all actors while keeping other actors fixed
        for i, (actor, optimizer) in enumerate(zip(self.actors, self.actor_optimizers)):
            # Sample this agent with gradient so that we can backpropagate
            a_i, log_pi_i = actor.sample(batch["state"])

            # Sample actions (no gradient) from the other actors
            actions, _ = self._sample_joint_action(batch["state"])

            # Construct the joint action, using the fixed actions of the other agents
            joint_action = torch.cat(
                [
                    actions[:, :i].detach(),
                    a_i,
                    actions[:, (i + 1) :].detach(),
                ],
                dim=-1,
            )

            # Use minimum of the two critics
            Q_1 = self.critic_1_network(batch["state"], joint_action)
            Q_2 = self.critic_2_network(batch["state"], joint_action)
            Q_min = torch.min(Q_1, Q_2)

            # Calculate the actor loss
            alpha = self.log_alpha.exp()
            actor_loss = (alpha * log_pi_i - Q_min).mean()

            # Back propagate the loss and update parameters
            optimizer.zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 0.5)
            optimizer.step()

            # Save the actor loss
            self.loss_actor[self.i, i] = actor_loss.item()

    def _update_alpha(self, batch):
        """
        Update the alpha parameter (temperature) using the batch.
        """
        _, log_pi = self._sample_joint_action(batch["state"])
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

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

    def _sample_joint_action(self, state):
        """
        Sample a joint action from the actor networks.
        This is extra functionality required for the multi-agent setting.
        It combines the actions from all actors into a single joint action.
        """
        actions, log_pis = [], []
        for actor in self.actors:
            with torch.no_grad():
                a, log_pi = actor.sample(state)
                actions.append(a)
                log_pis.append(log_pi)

        # Concatenate actions and sum log probabilities from all actors
        joint_action = torch.cat(actions, dim=-1)
        total_log_pi = torch.stack(log_pis, 0).sum(0)

        return joint_action, total_log_pi

    def _try_to_load_weights(self, uid: int = None) -> None:
        """
        Load the network weights from a folder if the uid is provided.

        Args:
            uid (int): Unique identifier for the network weights.
        """
        if not uid:
            print("No UID provided. Skipping loading of weights.")
            return

        try:
            checkpoint = torch.load(
                f"results/models/{uid}.pth",
                map_location=self.device,
                weights_only=False,
            )
        except FileNotFoundError:
            print(f"No saved model found for UID {uid}. Starting with random weights.")
            return

        # Load model weights
        for i, actor in enumerate(self.actors):
            if f"actor_{i}_network" not in checkpoint:
                print(f"Actor {i} network not found in checkpoint. Skipping.")
                continue
            actor.load_state_dict(checkpoint[f"actor_{i}_network"])
        self.critic_1_network.load_state_dict(checkpoint["critic_1_network"])
        self.critic_2_network.load_state_dict(checkpoint["critic_2_network"])
        self.value_network.load_state_dict(checkpoint["value_network"])
        self.target_value_network.load_state_dict(checkpoint["target_value_network"])

        # Load log_alpha and ensure it requires grad
        self.log_alpha = checkpoint["log_alpha"].to(self.device)
        self.log_alpha.requires_grad_(True)

        # Load optimizers
        for i, optimizer in enumerate(self.actor_optimizers):
            if f"actor_{i}_optimizer" not in checkpoint:
                print(f"Actor {i} optimizer not found in checkpoint. Skipping.")
                continue
            optimizer.load_state_dict(checkpoint[f"actor_{i}_optimizer"])
        self.critic_1_optimizer.load_state_dict(checkpoint["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(checkpoint["critic_2_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
