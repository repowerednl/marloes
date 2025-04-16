import torch
import numpy as np


def parse_state(
    state_list: list[dict], device: str = "cpu"
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """
    Divide state into scalars and forecast per agent and convert to tensors.

    Args:
        state_list (list[dict[str, Any]]): list of state dictionaries, where each dictionary contains
            agent-specific data including scalars and forecasts.
        device (str, optional): Device to place the tensors on (default: "cpu").

    Returns:
        dict[str, dict[str, dict[str, torch.Tensor]]]: A dictionary containing tensors for each agent's
            scalars and forecasts.
    """
    agent_data = {}
    SCALAR_KEYS = [
        "power",
        "available_power",
        "nomination",
        "state_of_charge",
        "degradation",
    ]

    for state_dict in state_list:
        for agent_name, agent_info in state_dict.items():
            if agent_name not in agent_data:
                agent_data[agent_name] = {"scalars": [], "forecast": []}
            scalars = []
            forecast_array = None

            for key in SCALAR_KEYS:
                # If the agent has this key, convert it; else add default 0
                if key in agent_info:
                    scalars.append(float(agent_info[key]))

            if "forecast" in agent_info:
                # GRU expects (.., seq_length, num_features), not (.., num_features, seq_length)
                # So we reshape
                forecast_array = np.array(
                    agent_info["forecast"], dtype=np.float32
                ).reshape(-1, 1)

            agent_data[agent_name]["scalars"].append(scalars)
            agent_data[agent_name]["forecast"].append(forecast_array)

    # Now convert tensors; stack along batch dim
    final_dict = {"agents": {}}

    for agent_name, data in agent_data.items():
        scalar = np.array(data["scalars"], dtype=np.float32)
        scalar_tensor = torch.from_numpy(scalar).to(device)

        # Check if we actually have forecast data
        forecast_list = data["forecast"]
        if any(f is None for f in forecast_list):
            forecast_tensor = None
        else:
            forecast = np.stack(forecast_list, axis=0)
            forecast_tensor = torch.from_numpy(forecast).to(device)

        final_dict["agents"][agent_name] = {
            "scalars": scalar_tensor,
            "forecast": forecast_tensor,
        }

    # TODO: add global context

    return final_dict


def parse_actions(
    action_list: list[dict[str, float]], device: str = "cpu"
) -> torch.Tensor:
    """
    Turn actions into a tensor.

    Args:
        action_list (list[dict[str, float]]): list of dictionaries containing actions for each agent.
        device (str, optional): Device to place the tensor on (default: "cpu").

    Returns:
        torch.Tensor: Tensor of shape (batch_size, num_agents) containing the actions.
    """
    # Sort to ensure consistency
    all_agents = list(action_list[0].keys())
    action_tensors = []

    for agent in all_agents:
        actions = [float(actions_dict[agent]) for actions_dict in action_list]
        # Also reshape; network requires (batch_size, feature_dim)
        # Here, feature_dim is 1 since we have a single action per agent
        agent_array = np.array(actions, dtype=np.float32).reshape(-1, 1)
        agent_tensor = torch.from_numpy(agent_array).to(device)
        action_tensors.append(agent_tensor)

    # Actions can be a single tensor with shape (batch_size, num_agents)
    final_actions = torch.cat(action_tensors, dim=-1)
    return final_actions


def parse_rewards(reward_list: list[float], device: str = "cpu") -> torch.Tensor:
    """
    Convert rewards to a tensor.

    Args:
        reward_list (list[float]): list of rewards for each sample in the batch.
        device (str, optional): Device to place the tensor on (default: "cpu").

    Returns:
        torch.Tensor: Tensor of shape (batch_size,) containing the rewards.
    """
    rewards = np.array(reward_list, dtype=np.float32)
    return torch.from_numpy(rewards).to(device)


def parse_batch(sample: list, device: str = "cpu") -> dict:
    """
    Convert a batch of transitions into a workable dictionary of tensors.

    Args:
        sample (list[Any]): list of transition objects, where each transition contains
            state, actions, rewards, and next_state.
        device (str, optional): Device to place the tensors on (default: "cpu").

    Returns:
        dict[str, Any]: A dictionary containing tensors for states, actions, rewards, and next states.
    """
    state_list = [t.state for t in sample]
    actions_list = [t.actions for t in sample]
    rewards_list = [t.rewards for t in sample]
    next_state_list = [t.next_state for t in sample]

    states = parse_state(state_list, device=device)
    actions = parse_actions(actions_list, device=device)
    rewards = parse_rewards(rewards_list, device=device)
    next_states = parse_state(next_state_list, device=device)

    sample_dict = {
        "state": states,
        "actions": actions,
        "rewards": rewards,
        "next_state": next_states,
    }
    return sample_dict
