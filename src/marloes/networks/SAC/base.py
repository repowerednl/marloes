import torch.nn as nn
import torch.nn.functional as F


class SACBaseNetwork(nn.Module):
    """
    Modular BaseNetwork class for Soft Actor-Critic (SAC) algorithm.
    It builds the hidden layers for the networks, as those are the same for all networks.
    """

    def __init__(self, input_dim, config, activation=nn.ReLU()):
        """
        Initializes the SACBaseNetwork with the given input and output dimensions and hidden layer dimensions.
        All provided defaults are based on the original SAC paper.
        """
        super(SACBaseNetwork, self).__init__()

        # Get the parameters from the config
        self.hidden_dim = config.get("hidden_dim", 256)
        num_layers = config.get("num_layers", 2)

        # Build the layers
        layers = []
        dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(dim, self.hidden_dim))
            layers.append(activation)
            dim = self.hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Basic forward pass through the network.
        """
        return self.hidden_layers(x)
