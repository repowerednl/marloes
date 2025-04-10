from unittest import TestCase, mock
import numpy as np
import torch
import torch.nn.functional as F

from marloes.networks.dreamer.WorldModel import (
    WorldModel,
    Decoder,
    RewardPredictor,
    ContinuePredictor,
)
from marloes.networks.dreamer.ActorCritic import Actor
from marloes.networks.dreamer.RSSM import RSSM, Encoder
from marloes.networks.details import RSSM_LD
from marloes.data.replaybuffer import ReplayBuffer


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

        [
            cls.replay_buffer.push(
                *cls.sample_transition(i, cls.observation_shape, cls.action_shape)
            )
            for i in range(100)
        ]

    @staticmethod
    def sample_transition(i, obs, act):
        state = {"value": np.full(obs, i)}
        actions = {"value": np.full(act, i + 0.1)}
        rewards = {"value": i + 0.2}
        next_state = {"value": np.full(obs, i + 1)}
        return state, actions, rewards, next_state

    def test_normal_creation(self):
        """
        Tests if the existed WorldModel is as expected.
        """
        world_model = WorldModel(self.observation_shape, self.action_shape)
        self.assertIsInstance(world_model.rssm, RSSM)
        self.assertIsInstance(world_model.rssm.encoder, Encoder)
        self.assertIsInstance(world_model.decoder, Decoder)
        self.assertIsInstance(world_model.reward_predictor, RewardPredictor)
        self.assertIsInstance(world_model.continue_predictor, ContinuePredictor)

    def test_imagine_end_to_end(self):
        """
        The imagine() function takes an Actor and returns predicted observations and rewards for a given horizon.
        initial: torch.Tensor with shape (batch, obs_size)
        """
        world_model = WorldModel(self.observation_shape, self.action_shape)
        # sample 5 random starting points from the replay buffer
        sample = self.replay_buffer.sample(batch_size=5)
        # Initialize the actor
        input_size = world_model.rssm.hidden_size + world_model.rssm.latent_state_size
        actor = Actor(input_size, self.action_shape[0])
        horizon = 16  # from Dreamer
        imagined_batch = world_model.imagine(
            starting_points=sample["state"], actor=actor, horizon=horizon
        )
        # should contain length(batch) elements
        self.assertEqual(len(imagined_batch), len(sample))
        # should contain dictionaries
        self.assertTrue(all(isinstance(i, dict) for i in imagined_batch))
        # each element should have states/actions/rewards with # of elements = horizon
        for i in imagined_batch:
            self.assertIn("states", i)
            self.assertIn("actions", i)
            self.assertIn("rewards", i)
            self.assertEqual(i["states"].shape[0], horizon)
            self.assertEqual(i["actions"].shape[0], horizon)
            self.assertEqual(i["rewards"].shape[0], horizon)

    def test_learn_end_to_end(self):
        """
        No errors during learn() returning the losses.
        """
        world_model = WorldModel(self.observation_shape, self.action_shape)
        # obtain a sample (5) from the replay buffer
        sample = self.replay_buffer.sample(batch_size=5, sequence=10)

        losses = world_model.learn(sample)
        self.assertIsInstance(losses["dynamics_loss"], torch.Tensor)
        self.assertIsInstance(losses["representation_loss"], torch.Tensor)
        self.assertIsInstance(losses["prediction_loss"], torch.Tensor)
        self.assertIsInstance(losses["total_loss"], torch.Tensor)
