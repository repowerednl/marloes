import torch
import torch.nn as nn
from marloes.networks.SAC.base import SACBaseNetwork


class ValueNetwork(SACBaseNetwork):
    """
    Value network (V-function) for the Soft Actor-Critic (SAC) algorithm.

    This network estimates the value of a given state. It takes the state as input,
    processes it through shared hidden layers, and outputs a single scalar V-value.

    Attributes:
        output_layer (nn.Linear): Linear layer to produce the V-value from the final hidden layer.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the ValueNetwork.

        Args:
            config (dict): Configuration dictionary containing:
                - "state_dim" (int): Dimension of the input state.
                - Other keys inherited from SACBaseNetwork for hidden layers configuration.
        """
        super(ValueNetwork, self).__init__(config["state_dim"], config["SAC"])
        self.output_layer = nn.Linear(self.hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            state (torch.Tensor): Input state tensor of shape (batch_size, state_dim).

        Returns:
            torch.Tensor: V-value tensor of shape (batch_size, 1).
        """
        x = super().forward(state)  # The hidden layers
        v_value = self.output_layer(x)
        return v_value
