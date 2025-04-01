from unittest import TestCase, mock
import numpy as np
import torch
import torch.nn.functional as F

from marloes.networks.WorldModel import (
    WorldModel,
    Encoder,
    Decoder,
    RewardPredictor,
    ContinuePredictor,
)
from marloes.networks.ActorCritic import Actor
from marloes.networks.RSSM import RSSM
from marloes.networks.details import RSSM_LD
from marloes.algorithms.replaybuffer import ReplayBuffer


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
        # For Solar, Battery and Demand, and 3 more batteries:
        #   - 2*1440, forecast of 24 hours (horizon) per asset with a forecast:
        #   - plus 6* power, (sol, dem, bat)
        #   - plus 2* nomination, (sol, dem)
        #   - plus 4* state_of_charge, (bat)
        #   - plus 4* degradation, (bat)
        # = 2880 + 6 + 2 + 4 + 4 = 2896
        # Any additional info...
        cls.observation_shape = (2896,)
        # Action Space: 6 agents, 6 actions
        cls.action_shape = (6,)
        # initialize a replay buffer to get the observations, actions, rewards and dones
        cls.replay_buffer = ReplayBuffer(100)
        obs = torch.tensor(np.ones(cls.observation_shape))
        action = torch.tensor(np.ones(cls.action_shape))
        reward = torch.tensor(1.0)
        for _ in range(100):
            cls.replay_buffer.push(obs=obs, action=action, reward=reward)

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

    def test_replay_buffer_sample(self):
        """
        Test if the replay buffer returns samples of the correct shape.
        """
        sample_size = 10
        sample = self.replay_buffer.sample(sample_size)
        self.assertEqual(sample["obs"].shape[0], sample_size)
        self.assertEqual(sample["action"].shape[0], sample_size)
        self.assertEqual(sample["reward"].shape[0], sample_size)

    def test_imagine_end_to_end(self):
        """
        The imagine() function takes an Actor and returns predicted observations and rewards for a given horizon.
        initial: torch.Tensor with shape (batch, obs_size)
        """
        world_model = WorldModel(self.observation_shape, self.action_shape)
        initial = torch.tensor(np.ones((1, self.observation_shape[0])))
        # Actor input size is h_t + z_t (RSSM hidden state + latent state)
        input_size = world_model.rssm.hidden_size + world_model.rssm.latent_state_size
        actor = Actor(input_size, self.action_shape[0])
        horizon = 16  # from Dreamer
        imagined = world_model.imagine(initial, actor, horizon)
        self.assertIsInstance(imagined["states"], torch.Tensor)
        self.assertIsInstance(imagined["rewards"], torch.Tensor)
        self.assertIsInstance(imagined["actions"], torch.Tensor)
        self.assertEqual(imagined["states"].shape[0], horizon + 1)  # also initial state
        self.assertEqual(imagined["rewards"].shape[0], horizon)
        self.assertEqual(imagined["actions"].shape[0], horizon)

    def test_learn_end_to_end(self):
        """
        No errors during learn() returning the losses.
        """
        world_model = WorldModel(self.observation_shape, self.action_shape)
        # obtain a sample (5) from the replay buffer
        sample = self.replay_buffer.sample(5)
        dones = torch.ones(5)
        # last done should not be continuation (1) but 0
        dones[-1] = 0

        d_loss, r_loss, p_loss, total_loss = world_model.learn(
            sample["obs"], sample["action"], sample["reward"], dones
        )
        self.assertIsInstance(d_loss, torch.Tensor)
        self.assertIsInstance(r_loss, torch.Tensor)
        self.assertIsInstance(p_loss, torch.Tensor)
        self.assertIsInstance(total_loss, torch.Tensor)
