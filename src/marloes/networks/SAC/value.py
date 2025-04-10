import torch.nn as nn
from marloes.networks.SAC.base import SACBaseNetwork


class ValueNetwork(SACBaseNetwork):
    """
    Value network (V-function) for the Soft Actor-Critic (SAC) algorithm.
    Input should be the state.
    The output is the V-value.
    """

    def __init__(self, config: dict):
        super(ValueNetwork, self).__init__(config["state_dim"], config)
        self.output_layer = nn.Linear(self.hidden_dim, 1)

    def forward(self, state):
        x = super().forward(state)  # The hidden layers
        v_value = self.output_layer(x)
        return v_value
