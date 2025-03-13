from unittest import TestCase, mock
import numpy as np
import torch

from marloes.networks.WorldModel import Encoder, Decoder, RewardPredictor
from marloes.networks.details import RSSM_LD
from marloes.networks.util import observation_to_tensor


class EncoderTestCase(TestCase):
    """
    Test Case for the Encoder, to test if creation is flexible and forward pass goes through all parts.
    """

    @classmethod
    def setUpClass(cls):
        state = {"nom": 1, "test": 2}
        state_2 = {"nom": 3, "test": 4, "extra": 5}
        cls.observation = {"agent1": state, "agent2": state_2}
        cls.tensor = observation_to_tensor(cls.observation)
        cls.hidden_dim = RSSM_LD.hidden["recurrent"][
            "hidden_size"
        ]  # in WorldModel.py encoder creation is done with self.rssm.rnn.hidden_size
        # create the Encoder
        cls.encoder = Encoder(cls.tensor.shape, latent_dim=cls.hidden_dim)

    def test_encoder_creation(self):
        """
        Test if the Encoder is created correctly.
        """
        self.assertEqual(self.encoder.fc1.in_features, self.tensor.shape[0])
        self.assertEqual(self.encoder.fc2.out_features, self.hidden_dim)

    def test_encoder_forward(self):
        """
        Test if the forward pass goes through the Encoder.
        """
        with mock.patch(
            "torch.nn.functional.relu", wraps=torch.nn.functional.relu
        ) as mock_relu:
            z_t = self.encoder(self.observation)
            # result should be a tensor with shape (hidden_dim,)
            self.assertIsInstance(z_t, torch.Tensor)
            self.assertEqual(z_t.shape[0], self.hidden_dim)
            mock_relu.assert_called_once()
