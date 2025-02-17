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
        layer_details = LayerDetails(*get_valid_layerdetails())
        print(layer_details)
        layer_details.validate()
        cls.base = BaseNetwork(layer_details=layer_details)

    def test_basenetwork_creation(self):
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
