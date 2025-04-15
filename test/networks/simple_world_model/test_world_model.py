import torch
import numpy as np
import pytest

from marloes.networks.simple_world_model.world_model import (
    WorldModel,
    flatten_state,
    unflatten_state,
)


## Some fixture set-up for the tests
@pytest.fixture
def dummy_config():
    return {
        "num_agents": 2,
        "action_dim": 2,
        "global_dim": 0,
        "WorldModel": {
            "forecast_hidden_size": 8,
            "forecast_num_layers": 1,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "scalar_dim": 5,
        },
        "agents_scalar_dim": [3, 3],
        "forecasts": [True, False],
    }


@pytest.fixture
def dummy_forward_input(dummy_config):
    # Create dummy inputs for the forward pass.
    num_agents = dummy_config["num_agents"]
    batch_size = 1
    scalar_dim = 3
    forecast_seq_len = 5
    forecast_input_dim = 1
    scalars = [torch.randn(batch_size, scalar_dim) for _ in range(num_agents)]
    forecasts = [
        torch.randn(batch_size, forecast_seq_len, forecast_input_dim)
        for _ in range(num_agents)
    ]
    global_context = torch.randn(batch_size, dummy_config["global_dim"])
    actions = torch.randn(batch_size, dummy_config["action_dim"])
    return scalars, forecasts, global_context, actions


# Need this for update testing
class DummyTransition:
    def __init__(self, state, actions, rewards, next_state):
        self.state = state
        self.actions = actions
        self.rewards = rewards
        self.next_state = next_state


@pytest.fixture
def dummy_transition():
    # This is simplified version, but should match our state.
    state = {
        "Agent 0": {
            "power": 0.0,
            "available_power": 50.0,
            "forecast": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "nomination": 10.0,
        },
        "Agent 1": {
            "power": 0.0,
            "available_power": 20.0,
            "nomination": 5.0,
        },
        "global_context": {},
    }
    actions = {"Agent 0": 0.5, "Agent 1": -0.3}
    reward = 1.0
    return DummyTransition(state, actions, reward, state)


def test_world_model_initialization(dummy_config):
    # Simple init testing
    model = WorldModel(dummy_config)
    assert isinstance(model, WorldModel)
    assert len(model.agent_state_encoders) == dummy_config["num_agents"]


def test_world_model_forward_dimensions(dummy_config, dummy_forward_input):
    # Test for dimensions of the output of forward (with submodules)
    model = WorldModel(dummy_config)
    scalars, forecasts, global_context, actions = dummy_forward_input
    next_state, reward = model.forward(scalars, forecasts, global_context, actions)
    # Main check for batch dimensions (assume the rest will be correct)
    assert next_state.shape[0] == 1
    assert reward.shape[0] == 1


def test_world_model_update_executes(dummy_config, dummy_transition):
    # Just test for errors
    model = WorldModel(dummy_config)
    sample_batch = [dummy_transition]
    model.update(sample_batch)


def test_flatten_unflatten_consistency(dummy_transition):
    """
    Test for flatten and unflatten -> should lead to the original state.
    """
    state = dummy_transition.state
    flat = flatten_state(state)
    unflat = unflatten_state(state, iter(flat))
    assert state["Agent 0"]["power"] == unflat["Agent 0"]["power"]
    assert state["Agent 0"]["available_power"] == unflat["Agent 0"]["available_power"]


def test_world_model_predict_executes(dummy_config, dummy_transition):
    # Again just test for errors
    model = WorldModel(dummy_config)
    state = dummy_transition.state
    action = dummy_transition.actions
    # Should be able to predict for multiple states/actions
    states = [state, state]
    # Create corresponding dummy actions as a list of dicts.
    actions = [action, action]
    next_state, rewards_list = model.predict(states, actions)
    assert len(next_state) == 2
    assert len(rewards_list) == 2
    # Check dimensions
    assert isinstance(next_state[0], dict)
    assert isinstance(rewards_list[0], float)
