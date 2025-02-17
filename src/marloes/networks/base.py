from torch.optim import Adam
from torch.nn import Module, MSELoss
import torch
from dataclasses import dataclass


@dataclass
class LayerDetails:
    """
    Class to store details of the layers in a network, input expects one layer, hidden expects a list of layers and output expects one layer.
    Example:
    input = {
        "details": {"in_features": 10, "out_features": 20},
        "activation": "ReLU"
    }
    hidden = {
        "layer_1": {
            "details": {"in_features": 20, "out_features": 30},
            "activation": "ReLu"
        },
        "dropout": {
            "details": {"p": 0.5}
        },
        "layer_2": {
            "details": {"in_features": 30, "out_features": 40},
            "activation": "ReLu"}
        },
    }
    output = {
        "details": {"in_features": 40, "out_features": 1},
        "activation": "Sigmoid"
    }
    """

    input: dict
    hidden: dict
    output: dict
    random_init: bool = False


@dataclass
class HyperParams:
    """
    Class to store hyperparameters for a network.
    """

    lr: float
    weight_decay: float


class BaseNetwork(Module):
    """
    Base class for all networks.
    """

    def __init__(
        self,
        params: dict = None,
        layer_details: LayerDetails = None,
        hyper_params: HyperParams = None,
    ):
        if not params and not layer_details:
            raise ValueError(
                "Either params or input_dim and output_dim must be provided."
            )
        self._validate_layerdetails(layer_details)
        super(BaseNetwork, self).__init__()
        self.initialize(params, layer_details)
        self.optimizer = Adam(
            self.parameters(),
            lr=hyper_params.lr,
            weight_decay=hyper_params.weight_decay,
        )
        self.loss = MSELoss()

    def initialize(self, params: dict, layer_details: LayerDetails):
        """
        Method to initialize the network. Should be implemented by the child class.
        """
        self._initialize_layers(layer_details)
        if params:
            self._load_from_params(params)
        elif layer_details.random_init:
            self._initialize_random_params()

    def _initialize_layers(self, layer_details: LayerDetails):
        """
        Method to initialize the layers of the network.
        """
        # initialize input layer
        self.input = torch.nn.Sequential(
            torch.nn.Linear(**layer_details.input["details"]),
            self._select_activation(layer_details.input["activation"]),
        )
        # initialize hidden layers
        self.hidden = torch.nn.ModuleList()
        for layer in layer_details.hidden.values():
            if "dropout" in layer:
                self.hidden.append(torch.nn.Dropout(**layer["details"]))
            else:
                self.hidden.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(**layer["details"]),
                        self._select_activation(layer["activation"]),
                    )
                )
        # initialize output layer
        self.output = torch.nn.Sequential(
            torch.nn.Linear(**layer_details.output),
            self._select_activation(layer_details.output["activation"]),
        )

    @staticmethod
    def _select_activation(activation):
        """
        Method to select the activation function.
        """
        if activation == "ReLU":
            return torch.nn.ReLU()
        elif activation == "Sigmoid":
            return torch.nn.Sigmoid()
        elif activation == "Tanh":
            return torch.nn.Tanh()
        elif activation == "Softmax":
            return torch.nn.Softmax(dim=1)
        else:
            raise ValueError("Activation function not supported")

    def forward(self, x):
        """
        Forward pass of the network.
        """
        x = self.input(x)
        for layer in self.hidden:
            x = layer(x)
        x = self.output(x)
        return x

    def backward(self):
        """
        Backward pass of the network.
        """
        raise NotImplementedError("Backward method not implemented.")

    def _load_from_params(self, params):
        """
        Method to load the network parameters.
        """
        self.load_state_dict(params)

    def _initialize_random_params(self):
        """
        Initialize random parameters for the network.
        """
        for param in self.parameters():
            param.data.uniform_(-0.1, 0.1)

    def save(self, path):
        """
        Method to save the network parameters to a file.
        """
        torch.save(self.state_dict(), path)

    @staticmethod
    def _validate_layerdetails(layerdetails: LayerDetails):
        """
        Method to validate the layer details.
        """
        if not isinstance(layerdetails, LayerDetails):
            raise ValueError("Layer details must be an instance of LayerDetails.")
        if not isinstance(layerdetails.input, dict):
            raise ValueError("Input layer details must be a dictionary.")
        if not isinstance(layerdetails.hidden, dict):
            raise ValueError("Hidden layer details must be a dictionary.")
        if not isinstance(layerdetails.output, dict):
            raise ValueError("Output layer details must be a dictionary.")
        if not isinstance(layerdetails.random_init, bool):
            raise ValueError("Random init must be a boolean.")
        # now we need to check if the output matches the next input for input > hidden > output
        if (
            layerdetails.input["details"]["out_features"]
            != layerdetails.hidden[0]["details"]["in_features"]
        ):
            raise ValueError(
                "Output of input layer must match input of first hidden layer."
            )
        for i in range(len(layerdetails.hidden) - 1):
            if (
                layerdetails.hidden[i]["details"]["out_features"]
                != layerdetails.hidden[i + 1]["details"]["in_features"]
            ):
                raise ValueError(
                    "Output of hidden layer must match input of next hidden layer."
                )
        if (
            layerdetails.hidden[-1]["details"]["out_features"]
            != layerdetails.output["details"]["in_features"]
        ):
            raise ValueError(
                "Output of last hidden layer must match input of output layer."
            )
        # if the output uses a sigmoid activation, the output features must be 1
        if (
            layerdetails.output["activation"] == "Sigmoid"
            and layerdetails.output["details"]["out_features"] != 1
        ):
            raise ValueError(
                "Output layer with sigmoid activation must have 1 output feature."
            )
        # if the output uses a softmax activation, the output features must be > 1
        if (
            layerdetails.output["activation"] == "Softmax"
            and layerdetails.output["details"]["out_features"] <= 1
        ):
            raise ValueError(
                "Output layer with softmax activation must have > 1 output feature."
            )
