import pytest
import torch

from marloes.networks.SAC.actor import ActorNetwork


@pytest.fixture
def dummy_config():
    return {
        "state_dim": 10,
        "action_dim": 4,
        "SAC": {
            "log_std_min": -20,
            "log_std_max": 2,
        },
        "hidden_dim": 64,
    }


def test_actor_sample(dummy_config):
    actor = ActorNetwork(dummy_config)

    # Create a batch of (flattened) states
    batch_size = 32
    dummy_states = torch.randn(batch_size, dummy_config["state_dim"])

    actions, log_probs = actor.sample(dummy_states)

    # Check action shape
    assert actions.shape == (batch_size, dummy_config["action_dim"])

    # Check log_probs shape
    assert log_probs.shape == (batch_size, 1)

    # Check that the actions are squashed right
    assert torch.all(actions <= 1.0) and torch.all(actions >= -1.0)
