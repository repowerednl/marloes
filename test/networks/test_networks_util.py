from unittest import TestCase
import torch

from marloes.networks.util import observation_to_tensor


class TestUtil(TestCase):
    """
    Test utility functions for network.
    """

    def setUp(self):
        self.observation = {
            "agent1": {"nom": 1, "test": 2},
            "agent2": {"nom": 3, "test": 4, "extra": 5},
        }

    def test_observation_to_tensor(self):
        """
        Test if the observation is converted to tensor correctly.
        """
        tensor = observation_to_tensor(
            observation=self.observation, concatenate_all=True
        )
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
        tensor = observation_to_tensor(
            observation=self.observation, concatenate_all=False
        )
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
