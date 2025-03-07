import torch

from marloes.networks.base import LayerDetails, BaseNetwork
from unittest import TestCase


def create_layer_details(input_details, hidden_details, output_details):
    return LayerDetails(input_details, hidden_details, output_details)


class TestLayerDetailsValidation(TestCase):
    def setUp(self):
        """
        Initialize different possible layer details for testing.
        """
        # initialize correct layer details
        self.correct_input = {
            "details": {"in_features": 5, "out_features": 10},
            "activation": "ReLU",
        }
        self.correct_hidden = {
            "layer_1": {
                "details": {"in_features": 10, "out_features": 20},
                "activation": "ReLU",
            },
            "dropout": {"details": {"p": 0.5}},
            "layer_2": {
                "details": {"in_features": 20, "out_features": 30},
                "activation": "ReLU",
            },
        }
        self.correct_recurrent = {
            "recurrent": {
                "details": {
                    "input_size": 10,
                    "hidden_size": 30,
                    "num_layers": 1,
                    "nonlinearity": "tanh",
                    "bias": True,
                    "batch_first": False,
                    "dropout": 0.0,
                    "bidirectional": False,
                }
            }
        }
        self.correct_output = {
            "details": {"in_features": 30, "out_features": 1},
            "activation": "Sigmoid",
        }

    def test_validate_layer_details_correct(self):
        """
        Test if the layer details are created correctly.
        """
        layer_details = create_layer_details(
            self.correct_input, self.correct_hidden, self.correct_output
        )
        self.assertIsNone(layer_details.validate())

    def test_validate_layer_details_wrong_input(self):
        """
        Test if the input layer details are validated correctly.
        """
        layer_details = create_layer_details(
            None, self.correct_hidden, self.correct_output
        )
        with self.assertRaises(ValueError):
            layer_details.validate_input()
        input_without_activation = {"details": self.correct_input["details"]}
        layer_details = create_layer_details(
            input_without_activation, self.correct_hidden, self.correct_output
        )
        with self.assertRaises(ValueError):
            layer_details.validate_input()

    def test_validate_layer_details_wrong_hidden(self):
        """
        Test if the hidden layer details are validated correctly.
        """
        layer_details = create_layer_details(
            self.correct_input, None, self.correct_output
        )
        with self.assertRaises(ValueError):
            layer_details.validate_hidden()
        hidden_without_layer_1 = {
            "dropout": self.correct_hidden["dropout"],
            "layer_2": self.correct_hidden["layer_2"],
        }
        layer_details = create_layer_details(
            self.correct_input, hidden_without_layer_1, self.correct_output
        )
        with self.assertRaises(ValueError):
            layer_details.validate_hidden()
        hidden_without_out_features = {
            "layer_1": self.correct_hidden["layer_1"],
            "dropout": self.correct_hidden["dropout"],
            "layer_2": {"details": {"in_features": 20}, "activation": "ReLU"},
        }
        layer_details = create_layer_details(
            self.correct_input, hidden_without_out_features, self.correct_output
        )
        with self.assertRaises(ValueError):
            layer_details.validate_hidden()

    def test_validate_layer_details_wrong_output(self):
        """
        Test if the output layer details are validated correctly.
        """
        output_without_activation = {"details": self.correct_output["details"]}
        layer_details = create_layer_details(
            self.correct_input, self.correct_hidden, output_without_activation
        )
        with self.assertRaises(ValueError):
            layer_details.validate_output()
        output_with_wrong_features = {
            "details": {"in_features": 30, "out_features": 2},
            "activation": "Sigmoid",
        }
        layer_details = create_layer_details(
            self.correct_input, self.correct_hidden, output_with_wrong_features
        )
        with self.assertRaises(ValueError):
            layer_details.validate_output()

    def test_validate_layer_details_missing_input_details(self):
        """
        Test if the input layer details are validated correctly when details are missing.
        """
        input_missing_details = {"activation": "ReLU"}
        layer_details = create_layer_details(
            input_missing_details, self.correct_hidden, self.correct_output
        )
        with self.assertRaises(ValueError):
            layer_details.validate_input()

    def test_validate_layer_details_missing_hidden_activation(self):
        """
        Test if the hidden layer details are validated correctly when activation is missing.
        """
        hidden_missing_activation = {
            "layer_1": {"details": {"in_features": 10, "out_features": 20}},
            "dropout": {"details": {"p": 0.5}},
            "layer_2": {
                "details": {"in_features": 20, "out_features": 30},
                "activation": "ReLU",
            },
        }
        layer_details = create_layer_details(
            self.correct_input, hidden_missing_activation, self.correct_output
        )
        with self.assertRaises(ValueError):
            layer_details.validate_hidden()

    def test_validate_layer_details_missing_output_details(self):
        """
        Test if the output layer details are validated correctly when details are missing.
        """
        output_missing_details = {"activation": "Sigmoid"}
        layer_details = create_layer_details(
            self.correct_input, self.correct_hidden, output_missing_details
        )
        with self.assertRaises(ValueError):
            layer_details.validate_output()

    def test_validate_layer_details_softmax_output(self):
        """
        Test if the output layer details are validated correctly for softmax activation.
        """
        output_with_softmax = {
            "details": {"in_features": 30, "out_features": 1},
            "activation": "Softmax",
        }
        layer_details = create_layer_details(
            self.correct_input, self.correct_hidden, output_with_softmax
        )
        with self.assertRaises(ValueError):
            layer_details.validate_output()

    def test_validate_layer_details_with_dropout_before_output(self):
        """
        Makes sure the validation is correct when dropout is before the output layer.
        """
        # add a dropout at the end of the hidden layers
        self.correct_hidden["dropout_2"] = {"details": {"p": 0.5}}
        layer_details = create_layer_details(
            self.correct_input, self.correct_hidden, self.correct_output
        )
        self.assertIsNone(layer_details.validate())

    def test_validate_layer_details_with_valid_recurrent(self):
        """
        Makes sure the validation is correct when LayerDetails contain a valid recurrent element.
        """
        layer_details = create_layer_details(
            self.correct_input, self.correct_recurrent, self.correct_output
        )
        self.assertIsNone(layer_details.validate())

    def test_validate_layer_details_with_valid_bidirectional_recurrent(self):
        """
        Makes sure the validation is correct when LayerDetails contain a valid bidirectional recurrent element.
        """
        # additional test case with a bidirectional RNN (input of output should be double hidden_size)
        self.correct_recurrent["recurrent"]["details"]["bidirectional"] = True
        self.correct_output["details"]["in_features"] = 60
        layer_details = create_layer_details(
            self.correct_input, self.correct_recurrent, self.correct_output
        )
        self.assertIsNone(layer_details.validate())

        # additional test case with another hidden layer
        self.correct_recurrent["layer_1"] = {
            "details": {"in_features": 60, "out_features": 30},
            "activation": "ReLU",
        }
        # change the output layer back
        self.correct_output["details"]["in_features"] = 30
        layer_details = create_layer_details(
            self.correct_input, self.correct_recurrent, self.correct_output
        )
        self.assertIsNone(layer_details.validate())

        # remove layer_1 and change bidirectional back to False; to avoid confusion in future testing
        self.correct_recurrent.pop("layer_1")
        self.correct_recurrent["recurrent"]["details"]["bidirectional"] = False

    def test_validate_layer_details_with_invalid_recurrent(self):
        """
        Makes sure the validation is correct when LayerDetails contain an invalid recurrent element.
        """
        invalid_recurrent_hidden = {
            "recurrent": {
                "details": {
                    "input_size": 10,
                    "hidden_size": 20,
                    "num_layers": 1,
                    "nonlinearity": "invalid_nonlinearity",  # Invalid nonlinearity
                    "bias": True,
                    "batch_first": False,
                    "dropout": 0.0,
                    "bidirectional": False,
                }
            }
        }
        layer_details = create_layer_details(
            self.correct_input, invalid_recurrent_hidden, self.correct_output
        )
        with self.assertRaises(ValueError):
            layer_details.validate_hidden()

    def test_actual_network_creation(self):
        """
        Test if the network is created correctly.
        """
        layer_details = create_layer_details(
            self.correct_input, self.correct_hidden, self.correct_output
        )
        network = BaseNetwork(layer_details=layer_details)
        # check if the network is created correctly, should have input (Sequential), hidden (ModuleList), and output (Sequential)
        self.assertIsInstance(network.input, torch.nn.Sequential)
        self.assertIsInstance(network.hidden, torch.nn.ModuleList)
        self.assertIsInstance(network.output, torch.nn.Sequential)
        # loss should be MSELoss()
        self.assertIsInstance(network.loss, torch.nn.MSELoss)

    def test_actual_network_creation_with_recurrent(self):
        """
        Test if the network is created correctly with a recurrent layer.
        """
        layer_details = create_layer_details(
            self.correct_input, self.correct_recurrent, self.correct_output
        )
        network = BaseNetwork(layer_details=layer_details)
        # check if the network is created correctly, should have input (Sequential), hidden (ModuleList), and output (Sequential)
        self.assertIsInstance(network.input, torch.nn.Sequential)
        self.assertIsInstance(network.hidden, torch.nn.ModuleList)
        # hidden should contain the recurrent layer
        self.assertIsInstance(network.hidden[0], torch.nn.RNN)
        self.assertIsInstance(network.output, torch.nn.Sequential)
        # loss should be MSELoss()
        self.assertIsInstance(network.loss, torch.nn.MSELoss)
