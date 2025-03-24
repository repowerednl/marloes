from unittest import TestCase
import torch

from marloes.networks.util import obs_to_tens, rew_to_tens


class TestUtil(TestCase):
    """
    Test utility functions for network.
    """

    def setUp(self):
        self.observation = {
            "agent1": {"nom": 1, "test": 2},
            "agent2": {"nom": 3, "test": 4, "extra": 5},
        }
        self.reward = {
            "agent1": 1,
            "agent2": 2,
            "agent3": 3,
        }

    def test_observation_to_tensor(self):
        """
        Test if the observation is converted to tensor correctly.
        """
        tensor = obs_to_tens(observation=self.observation, concatenate_all=True)
        # should be a tensor object
        self.assertIsInstance(tensor, torch.Tensor)
        # should have shape (5,)
        self.assertEqual(tensor.shape, torch.Size([5]))
        # should be [1, 2, 3, 4, 5]
        self.assertTrue(torch.equal(tensor, torch.tensor([1, 2, 3, 4, 5])))

    def test_observation_to_tensor_separate(self):
        """
        Test the observation to tensor function with concatenate_all=False which should return a list of tensorts.
        """
        tensor = obs_to_tens(observation=self.observation, concatenate_all=False)
        # should be a tensor object
        self.assertIsInstance(tensor, list)
        # first element should have shape (2,)
        self.assertEqual(tensor[0].shape, torch.Size([2]))
        # second element should have shape (3,)
        self.assertEqual(tensor[1].shape, torch.Size([3]))
        # first element should be [1, 2]
        self.assertTrue(torch.equal(tensor[0], torch.tensor([1, 2])))
        # second element should be [3, 4, 5]
        self.assertTrue(torch.equal(tensor[1], torch.tensor([3, 4, 5])))
        # number of agents should be 2
        self.assertEqual(len(tensor), 2)

    def test_reward_to_tensor(self):
        """
        Test if the reward is converted to tensor correctly.
        """
        tensor = rew_to_tens(rewards=self.reward, single_reward=True)
        # should be a tensor object
        self.assertIsInstance(tensor, torch.Tensor)
        # should have shape ()
        self.assertEqual(tensor.shape, torch.Size([]))
        # should be 6
        self.assertTrue(torch.equal(tensor, torch.tensor(6)))

    def test_reward_to_tensor_separate(self):
        """
        Test the reward to tensor function with single_reward=False which should return a tensor with rewards for each agent.
        """
        tensor = rew_to_tens(rewards=self.reward, single_reward=False)
        # should be a tensor object
        self.assertIsInstance(tensor, torch.Tensor)
        # should have shape (3,)
        self.assertEqual(tensor.shape, torch.Size([3]))
        # should be [1, 2, 3]
        self.assertTrue(torch.equal(tensor, torch.tensor([1, 2, 3])))
