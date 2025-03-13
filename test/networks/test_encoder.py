from unittest import TestCase
import numpy as np

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
        observations = {"agent1": state, "agent2": state_2}
        tensor = observation_to_tensor(observations)
        cls.encoder = Encoder(tensor.shape, latent_dim=3)

    def test_encoder_creation(self):
        """
        Test if the Encoder is created correctly.
        """
        self.assertEqual(self.encoder.fc1.in_features, self.observation_shape[0])
        self.assertEqual(self.encoder.fc2.out_features, self.latent_dim)
