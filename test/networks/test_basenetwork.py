from unittest import TestCase
from marloes.networks.base import BaseNetwork, LayerDetails
from marloes.networks.configuration import NetworkConfig
import torch

from test.util import get_valid_layerdetails


class TestBaseNetwork(TestCase):
    """
    Class for tests tot test functionality of the BaseNetwork.
    """

    @classmethod
    def setUp(cls):
        # create
        cls.layer_details = LayerDetails(*get_valid_layerdetails())
        cls.layer_details.validate()
        cls.base = BaseNetwork(layer_details=cls.layer_details)

    def test_basenetwork_initialization(self):
        """
        Test if the BaseNetwork is created correctly, should have input, hidden and output.
        """
        # input should be a torch.nn.Sequential() layer with input: 10, output: 2, and ReLU activation
        self.assertIsInstance(self.base.input, torch.nn.Sequential)
        self.assertEqual(self.base.input[0].in_features, 10)
        self.assertEqual(self.base.input[0].out_features, 2)
        self.assertEqual(self.base.input[1].__class__.__name__, "ReLU")
        # hidden should be a ModuleList() with 4 Sequential elements, a layer, a dropout, a layer and a dropout
        self.assertIsInstance(self.base.hidden, torch.nn.ModuleList)
        self.assertEqual(len(self.base.hidden), 4)
        self.assertIsInstance(self.base.hidden[0], torch.nn.Sequential)
        self.assertIsInstance(self.base.hidden[1], torch.nn.Dropout)
        self.assertIsInstance(self.base.hidden[2], torch.nn.Sequential)
        self.assertIsInstance(self.base.hidden[3], torch.nn.Dropout)
        # output should be a torch.nn.Sequential() layer with input: 4, output: 1, and Sigmoid activation
        self.assertIsInstance(self.base.output, torch.nn.Sequential)
        self.assertEqual(self.base.output[0].in_features, 4)
        self.assertEqual(self.base.output[0].out_features, 1)
        self.assertEqual(self.base.output[1].__class__.__name__, "Sigmoid")

    def test_no_params_given(self):
        """
        If no parameters are given a default lr and weight decay are used.
        """
        # make sure optimizer is Adam
        self.assertEqual(self.base.optimizer.__class__.__name__, "Adam")
        self.assertEqual(self.base.optimizer.defaults["lr"], 0.001)
        self.assertEqual(self.base.optimizer.defaults["weight_decay"], 0.01)

    def test_forward(self):
        """
        Test the forward pass of the BaseNetwork.
        """
        # create a dummy input tensor
        input_tensor = torch.randn(10)
        # make sure forward pass works
        output = self.base(input_tensor)
        # output should be one value
        self.assertEqual(len(output), 1)

    def test_initialization_based_on_params(self):
        """
        Test if the model can be created if the parameters are saved correctly.
        """
        # save the parameters
        params = self.base.state_dict()
        # create a new model with the saved parameters
        new_model = BaseNetwork(params=params, layer_details=self.layer_details)
        # make sure the new model has the same parameters
        for key in params:
            self.assertTrue(torch.equal(params[key], new_model.state_dict()[key]))

    def test_saving_and_loading(self):
        """
        Test if the model can be saved and loaded correctly.
        """
        save_path = "params.pth"
        # save the model
        self.base.save(save_path)
        # load the model into BaseNetwork
        new_model = BaseNetwork(
            params=torch.load(save_path), layer_details=self.layer_details
        )  # loading as done in NetworkConfig
        # make sure the new model has the same parameters
        for key in self.base.state_dict():
            self.assertTrue(
                torch.equal(self.base.state_dict()[key], new_model.state_dict()[key])
            )
        # delete the saved model
        import os

        os.remove(save_path)
        self.assertFalse(os.path.exists(save_path))
