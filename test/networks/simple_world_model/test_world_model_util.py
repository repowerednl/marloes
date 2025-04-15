import torch
import numpy as np
import pytest
from marloes.networks.simple_world_model.util import (
    parse_state,
    parse_actions,
    parse_rewards,
    parse_batch,
)


@pytest.fixture
def example_state():
    return [
        {
            "SolarAgent 0": {
                "power": 0.0,
                "available_power": 43.21193509915177,
                "forecast": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "nomination": 10.510129678863084,
            },
            "SolarAgent 1": {
                "power": 0.0,
                "available_power": 0.0,
                "forecast": np.array([4.0, 5.0, 6.0], dtype=np.float32),
                "nomination": 8.267004187266336,
            },
            "WindAgent 0": {"power": 0.0, "available_power": 7.765112675967633},
            "DemandAgent 0": {
                "power": 0.0,
                "forecast": np.array([7.0, 8.0, 9.0], dtype=np.float32),
                "nomination": -30.37188596955736,
            },
            "DemandAgent 1": {
                "power": 0.0,
                "forecast": np.array([10.0, 11.0, 12.0], dtype=np.float32),
                "nomination": -5.02699144260975,
            },
            "BatteryAgent 0": {"power": 0.0, "state_of_charge": 0.5, "degradation": 0},
            "BatteryAgent 1": {"power": 0.0, "state_of_charge": 0.5, "degradation": 0},
            "additional_info": {},
        }
    ]


def test_parse_state(example_state):
    parsed = parse_state(example_state)
    assert "agents" in parsed
    assert "SolarAgent 0" in parsed["agents"]
    assert parsed["agents"]["SolarAgent 0"]["scalars"].shape == (1, 3)
    assert parsed["agents"]["SolarAgent 0"]["forecast"].shape == (1, 3, 1)


def test_parse_actions():
    action_list = [{"SolarAgent 0": 1.0, "BatteryAgent 0": -0.5}]
    parsed = parse_actions(action_list)
    assert isinstance(parsed, torch.Tensor)
    assert parsed.shape == (1, 2)  # 1 batch, 2 actions
    assert torch.allclose(parsed[0], torch.tensor([1.0, -0.5], dtype=torch.float32))


def test_parse_rewards():
    rewards = [0.5, 1.2, -0.7]
    parsed = parse_rewards(rewards)
    assert torch.allclose(parsed, torch.tensor([0.5, 1.2, -0.7], dtype=torch.float32))


class DummyTransition:
    def __init__(self, state, actions, rewards, next_state):
        self.state = state
        self.actions = actions
        self.rewards = rewards
        self.next_state = next_state


def test_parse_batch(example_state):
    sample = [
        DummyTransition(
            state=example_state[0],
            actions={"SolarAgent 0": 1.0, "BatteryAgent 0": -0.5},
            rewards=1.0,
            next_state=example_state[0],
        )
    ]
    parsed = parse_batch(sample)
    assert "state" in parsed
    assert "actions" in parsed
    assert "rewards" in parsed
    assert "next_state" in parsed
