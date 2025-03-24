from unittest import TestCase, mock
import numpy as np
import torch

from marloes.networks.WorldModel import Encoder, Decoder, RewardPredictor
from marloes.networks.details import RSSM_LD
from marloes.networks.util import dict_to_tens


class EncoderDecoderTestCase(TestCase):
    """
    Test Case for the Encoder and Decoder, to test if creation is flexible and forward pass goes through all parts.
    """

    @classmethod
    def setUpClass(cls):
        state = {"nom": 1, "test": 2}
        state_2 = {"nom": 3, "test": 4, "extra": 5}
        cls.observation = {"agent1": state, "agent2": state_2}
        cls.tensor = dict_to_tens(cls.observation)

        cls.z_t_size = RSSM_LD.hidden["dense"]["out_features"]
        cls.encoder = Encoder(cls.tensor.shape[0], cls.z_t_size)
        cls.z_t = torch.randn(cls.z_t_size)
        cls.decoder = Decoder(cls.z_t_size, cls.tensor.shape[0])

    def test_encoder_creation(self):
        """
        Test if the Encoder is created correctly.
        """
        self.assertEqual(self.encoder.fc1.in_features, self.tensor.shape[0])
        self.assertEqual(self.encoder.fc_mu.out_features, self.z_t_size)
        self.assertEqual(self.encoder.fc_logvar.out_features, self.z_t_size)

    def test_encoder_forward(self):
        """
        Test if the forward pass goes through the Encoder.
        """
        with mock.patch(
            "torch.nn.functional.relu", wraps=torch.nn.functional.relu
        ) as mock_relu:
            z_t, details = self.encoder(self.tensor)
            self.assertIsInstance(z_t, torch.Tensor)
            self.assertEqual(z_t.shape[0], self.z_t_size)
            mock_relu.assert_called_once()

    def test_decoder_creation(self):
        """
        Test if the Decoder is created correctly.
        """
        self.assertEqual(self.decoder.fc1.in_features, self.z_t_size)
        self.assertEqual(self.decoder.fc2.out_features, self.tensor.shape[0])

    def test_decoder_forward(self):
        """
        Test if the forward pass goes through the Decoder.
        """
        with mock.patch(
            "torch.nn.functional.relu", wraps=torch.nn.functional.relu
        ) as mock_relu:
            x_hat_t = self.decoder(self.z_t)
            self.assertIsInstance(x_hat_t, torch.Tensor)
            self.assertEqual(x_hat_t.shape[0], self.tensor.shape[0])
            mock_relu.assert_called_once()

    def test_latent_state_size(self):
        """
        Test if the latent state size is the same for Encoder and Decoder.
        """
        self.assertEqual(self.encoder.fc_mu.out_features, self.decoder.fc1.in_features)
        self.assertEqual(
            self.encoder.fc_logvar.out_features, self.decoder.fc1.in_features
        )
