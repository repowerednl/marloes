import torch
import numpy as np


def parse_state(state_list, device="cpu"):
    """
    Divide state into scalars and forecast per agent and convert to tensors.
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
                else:
                    scalars.append(0.0)

            if "forecast" in agent_info:
                forecast_array = np.array(agent_info["forecast"], dtype=np.float32)

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


def parse_actions(action_list, device="cpu"):
    """
    Turn actions into a tensor.
    """
    all_agents = list(action_list[0].keys())
    agent_actions = {agent: [] for agent in all_agents}

    # Accumulate actions for each agent
    for actions_dict in action_list:
        for agent in all_agents:
            value = float(actions_dict[agent])
            agent_actions[agent].append(value)

    # Convert to torch
    final_actions = {}
    for agent in all_agents:
        actions = np.array(agent_actions[agent], dtype=np.float32)
        final_actions[agent] = torch.from_numpy(actions).to(device)

    return final_actions


def parse_rewards(reward_list, device="cpu"):
    """
    Convert rewards to a tensor.
    """
    rewards = np.array(reward_list, dtype=np.float32)
    return torch.from_numpy(rewards).to(device)


def parse_batch(sample, device="cpu"):
    """
    Since the WorldModel requires separating several parts of the transition,
    this converts state to a workable dict of tensors.
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
