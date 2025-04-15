from unittest import TestCase
from unittest.mock import patch
import pytest

import pandas as pd
import numpy as np
import torch

from marloes.algorithms.dreamer import Dreamer
from marloes.agents.base import Agent
from marloes.networks.dreamer.ActorCritic import ActorCritic
from marloes.networks.dreamer.WorldModel import WorldModel
from marloes.networks.util import dict_to_tens


def get_new_config() -> dict:
    return {
        "algorithm": "dreamer",
        "epochs": 10,
        "agents": [
            {
                "type": "demand",
                "scale": 1.5,
                "profile": "Farm",
            },
            {
                "type": "solar",
                "AC": 900,
                "DC": 1000,
                "orientation": "EW",
            },
            {
                "type": "battery",
                "energy_capacity": 1000,
                "efficiency": 0.9,
                "power": 100,
            },
        ],
    }


@pytest.mark.slow
class DreamerTestCase(TestCase):
    """
    Dreamer TestCase to check creation of an actual algorithm (and WorldModel in it).
    Not mocking anything to get the actual observations and actions.
    """

    @classmethod
    def setUpClass(cls) -> None:
        Agent._id_counters = {}
        with patch(
            "marloes.results.saver.Saver._update_simulation_number", return_value=0
        ):
            cls.alg = Dreamer(config=get_new_config())

    def test_init(self):
        self.assertEqual(len(self.alg.environment.agents), 3)
        # environment should have observation_space torch.Size([2888])
        self.assertEqual(self.alg.environment.observation_space, (2888,))
        # environment should have action_space torch.Size([3])
        self.assertEqual(self.alg.environment.action_space, (3,))
        # there should be a WorldModel and an ActorCritic
        self.assertIsInstance(self.alg.world_model, WorldModel)
        self.assertIsInstance(self.alg.actor_critic, ActorCritic)

    def test_get_actions(self):
        """
        Testing the actions (calling ActorCritic.act) and returning a dictionary.
        """
        obs, _ = self.alg.environment.reset()
        obs = dict_to_tens(obs, concatenate_all=True)
        actions = self.alg.get_actions(
            obs
        )  # Calls actor(model_state).sample() which returns shape: (#agents,)
        self.assertIsInstance(actions, dict)
        # one action for each agent
        self.assertEqual(len(actions), 3)

    def test_perform_training_steps(self):
        """
        Testing the training steps.
        """
        obs, _ = self.alg.environment.reset()
        self.alg.update_interval = 5
        # it should not call self.world_model.learn or self.actor_critic.learn for step 0-4 and return None
        for step in range(5):
            self.alg.perform_training_steps(step)
            self.assertIsNone(self.alg.world_model.learn)
            self.assertIsNone(self.alg.actor_critic.learn)
        # Mocking the learn methods to return a dictionary and make sure if step % update_interval == 0 returns a dict
        with patch.object(self.alg.world_model, "learn", return_value={}):
            with patch.object(self.alg.actor_critic, "learn", return_value={}):
                loss = self.alg.perform_training_steps(5)
                self.assertIsInstance(loss, dict)
