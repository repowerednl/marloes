from unittest import TestCase, mock
import numpy as np
import torch

from marloes.networks.ActorCritic import ActorCritic


class ActorCriticTestCase(TestCase):
    """
    Test case to test initialization and forward pass of the ActorCritic network.
    """

    @classmethod
    def setUpClass(cls):
        cls.actor_critic = ActorCritic(input=10, output=5)

    def test_init(self):
        """
        Test if the ActorCritic network initialized correctly.
        """
        self.assertTrue(hasattr(self.actor_critic, "actor"))
        self.assertTrue(hasattr(self.actor_critic, "critic"))
        self.assertTrue(hasattr(self.actor_critic, "actor_optim"))
        self.assertTrue(hasattr(self.actor_critic, "critic_optim"))
        self.assertEqual(self.actor_critic.actor_optim.__class__.__name__, "Adam")
        self.assertEqual(self.actor_critic.critic_optim.__class__.__name__, "Adam")

    def test_actor(self):
        """
        Test the forward pass of the Actor network.
        """
        obs = torch.tensor(np.random.rand(10)).float()
        actions = self.actor_critic.actor(obs)
        self.assertEqual(actions.shape, (5,))

    def test_critics(self):
        """
        Test the forward pass of the Critic network.
        """
        obs = torch.tensor(np.random.rand(10)).float()
        values = self.actor_critic.critic(obs)
        self.assertEqual(values.shape, (1,))

    def test_act(self):
        """
        Test the act method of the ActorCritic network.
        """
        obs = torch.tensor(np.random.rand(10)).float()
        actions = self.actor_critic.act(obs)
        self.assertEqual(actions.shape, (5,))

    def test_learn(self):
        """
        Test the learn method of the ActorCritic network.
        """
        pass
