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

    _id_counters = {}

    def __init__(
        self,
        input_dim: int,
        sac_config: dict,
        activation=nn.ReLU(),
        hidden_dim: int = None,
    ) -> None:
        """
        Initializes the SACBaseNetwork with the given input and output dimensions and hidden layer dimensions.
        All provided defaults are based on the original SAC paper.

        Args:
            input_dim (int): Dimension of the input features.
            config (dict): Configuration dictionary containing:
                - "hidden_dim" (int, optional): Dimension of the hidden layers (default: 256).
                - "num_layers" (int, optional): Number of hidden layers (default: 2).
            activation (nn.Module, optional): Activation function to use between layers (default: nn.ReLU()).
            hidden_dim (int, optional): Dimension of the hidden layers (default: None).
        """
        super(SACBaseNetwork, self).__init__()

        # Set name
        cls_name = self.__class__.__name__
        if cls_name not in SACBaseNetwork._id_counters:
            SACBaseNetwork._id_counters[cls_name] = 0

        self.name = f"{cls_name}_{SACBaseNetwork._id_counters[cls_name]}"
        SACBaseNetwork._id_counters[cls_name] += 1

        # Get the parameters from the config
        if hidden_dim is not None:
            self.hidden_dim = hidden_dim
        else:
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

    def try_to_load_weights(self, uid: int) -> None:
        """
        Load the network weights from a folder if the uid is provided.

        Args:
            uid (int): Unique identifier for the network weights.
        """
        self.was_loaded = False
        try:
            self.load_state_dict(torch.load(f"results/models/{self.name}/{uid}"))
            self.was_loaded = True
        except FileNotFoundError:
            print(
                f"Model weights for {self.name} with uid {uid} not found. Initializing with random weights."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Basic forward pass through the network hidden layers.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor after passing through the hidden layers.
        """
        return self.hidden_layers(x)
