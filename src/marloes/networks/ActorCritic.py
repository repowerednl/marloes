import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class ActorCritic:
    """
    This ActorCritic module is based on the DreamerV3 architecture.
    The Actor and Critic networks are combined here, and learn from abstract trajectories or representations (latent states) predicted by the WorldModel.
    - The Actor and Critic operate on model states, s_t = {h_t, z_t}.
    - The Actor aims to maximize return with gamma-discounted rewards (gamma = 0.997).
    - The Critic aims to predict the value of the current state.
    """

    def __init__(self, input: int, output: int, hidden_size: int = 256):
        """
        Initializes the ActorCritic network.
        """
        self.actor = Actor(input, output, hidden_size)
        self.critic = Critic(input, hidden_size)

        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)
        self.beta_weights = {"val": 1.0, "repval": 0.3}

    def act(self, obs: torch.Tensor):
        """
        Returns the actions predicted by the Actor network.
        """
        return self.actor(obs)

    def learn(self):
        """
        Learning step for the ActorCritic network.
        """
        pass


class Actor(nn.Module):
    """
    Actor class, MLP network with hidden layers, predicts the 'continuous' actions per agent. TODO: Discrete actions.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 256):
        """
        Initializes the Actor network.
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the Actor network.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


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
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass through the Critic network.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
