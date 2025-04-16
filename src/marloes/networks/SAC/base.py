import torch
import torch.nn as nn


class SACBaseNetwork(nn.Module):
    """
    Modular BaseNetwork class for Soft Actor-Critic (SAC) algorithm.

    It builds the hidden layers for the networks, as those are the same for the actor, critic, and value networks.

    Attributes:
        hidden_dim (int): Dimension of the hidden layers.
        hidden_layers (nn.Sequential): Sequential container for the hidden layers.
    """

    def __init__(self, input_dim: int, sac_config: dict, activation=nn.ReLU()) -> None:
        """
        Initializes the SACBaseNetwork with the given input and output dimensions and hidden layer dimensions.
        All provided defaults are based on the original SAC paper.

        Args:
            input_dim (int): Dimension of the input features.
            config (dict): Configuration dictionary containing:
                - "hidden_dim" (int, optional): Dimension of the hidden layers (default: 256).
                - "num_layers" (int, optional): Number of hidden layers (default: 2).
            activation (nn.Module, optional): Activation function to use between layers (default: nn.ReLU()).
        """
        super(SACBaseNetwork, self).__init__()

        # Get the parameters from the config
        self.hidden_dim = sac_config.get("hidden_dim", 256)
        num_layers = sac_config.get("num_layers", 2)

        # Build the layers
        layers = []
        dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(dim, self.hidden_dim))
            layers.append(activation)
            dim = self.hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Basic forward pass through the network hidden layers.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor after passing through the hidden layers.
        """
        return self.hidden_layers(x)
