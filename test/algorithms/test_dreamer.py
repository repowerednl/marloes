from unittest import TestCase
from unittest.mock import patch
import pytest

import pandas as pd
import numpy as np
import torch

from marloes.algorithms.dreamer import Dreamer
from marloes.handlers.base import Handler
from marloes.networks.dreamer.ActorCritic import ActorCritic
from marloes.networks.dreamer.WorldModel import WorldModel
from marloes.data.replaybuffer import ReplayBuffer


def get_new_config() -> dict:
    return {
        "algorithm": "dreamer",
        "epochs": 10,
        "handlers": [
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
            {
                "type": "wind",
                "AC": 900,
                "power": 1000,
                "location": "Offshore",
            },
            {
                "type": "demand",
                "scale": 1.5,
                "profile": "Farm",
            },
            {
                "type": "demand",
                "scale": 1.5,
                "profile": "RealEstate",
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
        Handler._id_counters = {}
        with patch(
            "marloes.results.saver.Saver._update_simulation_number", return_value=0
        ):
            cls.alg = Dreamer(config=get_new_config())

    def test_init(self):
        self.assertEqual(len(self.alg.environment.handlers), 6)
        # environment should have state dim
        self.assertEqual(self.alg.environment.state_dim, (7215,))
        # environment should have action_space torch.Size([3])
        self.assertEqual(self.alg.environment.action_dim, (6,))
        # there should be a WorldModel and an ActorCritic
        self.assertIsInstance(self.alg.world_model, WorldModel)
        self.assertIsInstance(self.alg.actor_critic, ActorCritic)

    def test_get_actions(self):
        """
        Testing the actions (calling ActorCritic.act) and returning a dictionary.
        """
        obs, _ = self.alg.environment.reset()
        obs = ReplayBuffer.dict_to_tens(obs, concatenate_all=True)
        actions = self.alg.get_actions(
            obs
        )  # Calls actor(model_state).sample() which returns shape: (#handlers,)
        self.assertIsInstance(actions, dict)
        # one action for each handler
        self.assertEqual(len(actions), 6)

    def test_perform_training_steps(self):
        """
        Testing the training steps.
        """
        actions = {
            handler_id: 0.5 for handler_id in self.alg.environment.handler_dict.keys()
        }
        # fill the replaybuffer with 100 elements
        for i in range(1000):
            obs, rew, _, _ = self.alg.environment.step(actions)
            self.alg.real_RB.push(obs, actions, rew, obs)

        obs, _ = self.alg.environment.reset()
        self.alg.update_interval = 5
        # it should not call self.world_model.learn or self.actor_critic.learn for step 0-4 and return None
        for step in range(5):
            self.alg.perform_training_steps(step)
            self.assertIsNone(self.alg.world_model.learn())
            self.assertIsNone(self.alg.actor_critic.learn())
        # Mocking the learn methods to return a dictionary and make sure if step % update_interval == 0 returns a dict
        with patch.object(self.alg.world_model, "learn", return_value={}):
            with patch.object(self.alg.actor_critic, "learn", return_value={}):
                loss = self.alg.perform_training_steps(5)
                self.assertIsInstance(loss, dict)
