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
            "activation": "ReLu"
        },
        "recurrent": {
            "details": {
                "input_size": 40,
                "hidden_size": 40,
                "num_layers": 1,
                "nonlinearity": "tanh", # The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``.
                "bias": True, # If False, then the layer does not use bias weights `b_ih` and `b_hh`.
                "batch_first": False, # input and output are (seq, batch, feature)
                "dropout": 0, # dropout probability
                "bidirectional": False
            },
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

    def validate(self):
        self.validate_input()
        self.validate_hidden()
        self.validate_output()

    def validate_input(self):
        if not isinstance(self.input, dict):
            raise ValueError("Input layer details must be a dictionary.")
        if "details" not in self.input or "activation" not in self.input:
            raise ValueError("Input layer details must have details and activation.")
        if not all(
            key in self.input["details"] for key in ["in_features", "out_features"]
        ):
            raise ValueError(
                "Input layer details must have in_features and out_features."
            )

    def validate_hidden(self):
        if not isinstance(self.hidden, dict):
            raise ValueError("Hidden layer details must be a dictionary.")
        for key, layer in self.hidden.items():
            if "details" not in layer:
                raise ValueError("Hidden layer details must have details.")
            if "dropout" in key:
                if "p" not in layer["details"]:
                    raise ValueError("Dropout layer details must have p.")
                if "activation" in layer:
                    raise ValueError(
                        "Hidden layer cannot have both dropout and activation."
                    )
            elif "recurrent" in key:
                required_keys = [
                    "input_size",
                    "hidden_size",
                    "num_layers",
                    "bias",
                    "batch_first",
                    "dropout",
                    "bidirectional",
                ]
                if not all(elem in layer["details"] for elem in required_keys):
                    raise ValueError(
                        "Recurrent layer details must have input_size, hidden_size, num_layers, nonlinearity, bias, batch_first, dropout, and bidirectional."
                    )
                if not isinstance(layer["details"]["bias"], bool):
                    raise ValueError("Bias must be a boolean.")
                if not isinstance(layer["details"]["batch_first"], bool):
                    raise ValueError("Batch_first must be a boolean.")
                if not isinstance(layer["details"]["dropout"], float):
                    raise ValueError("Dropout must be a float.")
                if not isinstance(layer["details"]["bidirectional"], bool):
                    raise ValueError("Bidirectional must be a boolean.")
            else:
                if not all(
                    elem in layer["details"] for elem in ["in_features", "out_features"]
                ):
                    raise ValueError(
                        "Hidden layer details must have in_features and out_features."
                    )
                if "activation" not in layer:
                    raise ValueError("Hidden layer must have activation.")

        def _get_first_layer(self):
            """
            Returns the first layer that is not a dropout layer, if it exists, otherwise raises an error
            """
            for key, layer in self.hidden.items():
                if "dropout" not in key:
                    return layer
            raise ValueError("No hidden layer found that is not a dropout layer.")

        if self.input["details"]["out_features"] != _get_first_layer(self)[
            "details"
        ].get("in_features", _get_first_layer(self)["details"].get("input_size", 0)):
            raise ValueError(
                f"Output of input layer must match input of first hidden layer. {self.input['details']['out_features']} != {_get_first_layer(self)['details'].get('in_features', _get_first_layer(self)['details'].get('input_size', 0))}"
            )
        hidden_layers = [
            value for key, value in self.hidden.items() if "dropout" not in key
        ]
        for i in range(len(hidden_layers) - 1):
            output = hidden_layers[i]["details"].get(
                "out_features", hidden_layers[i]["details"].get("hidden_size", 0)
            )  # never returns 0 as we established either out_features or hidden_size exist
            output = (
                output * 2
                if hidden_layers[i]["details"].get("bidirectional", False)
                else output
            )  # double output if bidirectional
            input = hidden_layers[i + 1]["details"].get(
                "in_features", hidden_layers[i + 1]["details"].get("input_size", 0)
            )  # never returns 0 as we established either in_features or input_size exist
            if output != input:
                raise ValueError(
                    f"Output of hidden layer must match input of next hidden layer. If an RNN is bidirectional, output of RNN is doubled. {output} != {input}"
                )

    def validate_output(self):
        if not isinstance(self.output, dict):
            raise ValueError("Output layer details must be a dictionary.")
        if "details" not in self.output or "activation" not in self.output:
            raise ValueError("Output layer details must have details and activation.")
        if not all(
            key in self.output["details"] for key in ["in_features", "out_features"]
        ):
            raise ValueError(
                "Output layer details must have in_features and out_features."
            )

        def _get_last_layer(self):
            """
            Returns the last layer that is not a dropout layer, if it exists, otherwise raises an error
            """
            for key, layer in reversed(self.hidden.items()):
                if "dropout" not in key:
                    return layer
            raise ValueError("No hidden layer found that is not a dropout layer.")

        last_layer = _get_last_layer(self)
        last_layer_output = last_layer["details"].get(
            "out_features", last_layer["details"].get("hidden_size", 0)
        )
        if last_layer["details"].get("bidirectional", False):
            last_layer_output *= 2  # double output if bidirectional

        if self.output["details"]["in_features"] != last_layer_output:
            raise ValueError(
                f"Output of last hidden layer must match input of output layer. {self.output['details']['in_features']} != {last_layer_output}"
            )
        if (
            self.output["activation"] == "Sigmoid"
            and self.output["details"]["out_features"] != 1
        ):
            raise ValueError(
                "Output layer with sigmoid activation must have 1 output feature."
            )
        if (
            self.output["activation"] == "Softmax"
            and self.output["details"]["out_features"] <= 1
        ):
            raise ValueError(
                "Output layer with softmax activation must have > 1 output feature."
            )


@dataclass
class HyperParams:
    """
    Class to store hyperparameters for a network.
    """

    lr: float = 0.001
    weight_decay: float = 0.01
    loss_fn = MSELoss()


class BaseNetwork(Module):
    """
    Base class for all networks.
    """

    def __init__(self):
        super().__init__()

    def _initialize_layers_from_layer_details(self, layer_details: LayerDetails):
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
        for key, layer in layer_details.hidden.items():
            if "dropout" in key:
                self.hidden.append(torch.nn.Dropout(**layer["details"]))
            elif "recurrent" in key:
                self.hidden.append(self._recurrent_layer(layer["details"]))
            else:
                self.hidden.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(**layer["details"]),
                        self._select_activation(layer["activation"]),
                    )
                )
        # initialize output layer
        self.output = torch.nn.Sequential(
            torch.nn.Linear(**layer_details.output["details"]),
            self._select_activation(layer_details.output["activation"]),
        )

    @staticmethod
    def _select_activation(activation: str):
        """
        Method to select the activation function.
        """
        activation = activation.lower()
        if activation == "relu":
            return torch.nn.ReLU()
        elif activation == "sigmoid":
            return torch.nn.Sigmoid()
        elif activation == "tanh":
            return torch.nn.Tanh()
        elif activation == "softmax":
            return torch.nn.Softmax(dim=1)
        else:
            raise ValueError("Activation function not supported")

    @staticmethod
    def _recurrent_layer(details):
        """
        Method to select the recurrent layer.
        # TODO: pop the type from details and select the wanted layer, might be different with parameters, would require changes to validation
        """
        return torch.nn.GRU(**details)

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
        # add .pth extension if not present
        if not path.endswith(".pth"):
            path += ".pth"
        torch.save(self.state_dict(), path)
