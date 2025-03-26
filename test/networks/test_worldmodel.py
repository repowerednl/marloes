from unittest import TestCase, mock
import numpy as np
import torch

from marloes.networks.WorldModel import (
    WorldModel,
    Encoder,
    Decoder,
    RewardPredictor,
    ContinuePredictor,
)
from marloes.networks.RSSM import RSSM
from marloes.networks.details import RSSM_LD


class WorldModelTestCase(TestCase):
    """
    TestCase for the bigger (end-to-end) WorldModel network.
    Testing the creation, and the bigger functions.
    """

    @classmethod
    def setUpClass(cls):
        """
        SetUp a WorldModel with the observation and action shape.
        """
        # Observation Space: torch.Size([2888])
        # For Solar, Battery and Demand:
        #   - 2*1440, forecast of 24 hours (horizon) per asset with a forecast:
        #   - plus 3* power, (sol, dem, bat)
        #   - plus 2* nomination, (sol, dem)
        #   - plus state_of_charge, (bat)
        #   - plus degradation, (bat)
        # = 2880 + 3 + 2 + 1 + 1 = 2888
        # Any additional info:
        # Action Space: torch.Size([3])

    def test_normal_creation(self):
        """
        Tests if the existed WorldModel is as expected.
        """
        world_model = WorldModel(self.observation_shape, self.action_shape)
        self.assertIsInstance(world_model.rssm, RSSM)
        self.assertIsInstance(world_model.encoder, Encoder)
        self.assertIsInstance(world_model.decoder, Decoder)
        self.assertIsInstance(world_model.reward_predictor, RewardPredictor)
        self.assertIsInstance(world_model.continue_predictor, ContinuePredictor)

    def test_learn(self):
        """
        Test if the WorldModel learn function passes through and calls all necessary functions.
        """
        world_model = WorldModel(self.observation_shape, self.action_shape)
        obs = torch.randn(1, 1, RSSM_LD.hidden["dense"]["out_features"])
        actions = torch.randn(6)
        rewards = torch.randn(10)
        dones = torch.tensor([False])
        # Make a batch with size 5 : of observations, actions, rewards, dones
        obs = torch.cat([obs] * 5, dim=0)
        actions = torch.cat([actions] * 5, dim=0)
        rewards = torch.cat([rewards] * 5, dim=0)
        dones = torch.cat([dones] * 5, dim=0)

        d_loss, r_loss = world_model.learn(obs, actions, rewards, dones)
        self.assertIsInstance(d_loss, torch.Tensor)
        self.assertIsInstance(r_loss, torch.Tensor)
