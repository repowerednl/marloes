from unittest import TestCase, mock
import torch
import numpy as np

from marloes.networks.WorldModel import RewardPredictor, ContinuePredictor


class PredictorsTestCase(TestCase):
    """
    Test case to test the (initialization and forward pass of) RewardPredictor and ContinuePredictor.
    """

    @classmethod
    def setUpClass(cls):
        cls.hidden_dim = 256
        cls.latent_dim = 64
        cls.reward_predictor = RewardPredictor(
            hidden_dim=cls.hidden_dim, latent_dim=cls.latent_dim
        )
        cls.continue_predictor = ContinuePredictor(
            hidden_dim=cls.hidden_dim, latent_dim=cls.latent_dim
        )

    def test_reward_predictor_initialization(self):
        """
        Testing predictor initialization. Should have layer fc (Linear).
        """
        self.assertEqual(
            self.reward_predictor.fc.in_features, self.hidden_dim + self.latent_dim
        )
        self.assertEqual(self.reward_predictor.fc.out_features, 1)

    def test_reward_predictor_forward(self):
        """
        Testing predictor forward pass. Takes a tensor of hidden_dim, and a tensor of latent_dim, should call fc once, and return one value.
        """
        hidden_state = torch.randn(1, self.hidden_dim)
        latent_state = torch.randn(1, self.latent_dim)
        with mock.patch.object(
            self.reward_predictor.fc, "forward", wraps=self.reward_predictor.fc.forward
        ) as mock_forward:
            output = self.reward_predictor(hidden_state, latent_state)
            mock_forward.assert_called_once()
            self.assertEqual(output.shape, torch.Size([1, 1]))

    def test_continue_predictor_initialization(self):
        """
        Testing predictor initialization. Should have layer fc (Linear).
        """
        self.assertEqual(
            self.continue_predictor.fc.in_features, self.hidden_dim + self.latent_dim
        )
        self.assertEqual(self.continue_predictor.fc.out_features, 1)

    def test_continue_predictor_forward(self):
        """
        Testing predictor forward pass. Takes a tensor of hidden_dim, and a tensor of latent_dim, should call fc once, and return one value.
        """
        hidden_state = torch.randn(1, self.hidden_dim)
        latent_state = torch.randn(1, self.latent_dim)
        with mock.patch.object(
            self.continue_predictor.fc,
            "forward",
            wraps=self.continue_predictor.fc.forward,
        ) as mock_forward:
            output = self.continue_predictor(hidden_state, latent_state)
            mock_forward.assert_called_once()
            self.assertEqual(output.shape, torch.Size([1, 1]))
