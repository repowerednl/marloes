import torch


def dist(mu, logvar):
    """
    Reparametrization trick to create a stochastic latent state, using mu and logvar for the distribution.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def obs_to_tens(
    observation: dict, concatenate_all: bool = True
) -> torch.Tensor | list[torch.Tensor]:
    """
    Transforms an observation dictionary into a tensor.

    Args:
        observation (dict): A dictionary where keys are agent identifiers, and values are dictionaries containing state information.
        concatenate_all (bool): If True, concatenates all agent states into a single tensor.
                               If False, returns a list of tensors, one per agent.

    Returns:
        torch.Tensor or list of torch.Tensor: The transformed observation tensor(s).
    """
    agent_tensors = []

    for agent in observation:
        agent_state = observation[agent]
        agent_tensor = torch.tensor(list(agent_state.values()), dtype=torch.float32)
        agent_tensors.append(agent_tensor)

    if concatenate_all:
        return torch.cat(agent_tensors) if agent_tensors else torch.tensor([])
    else:
        return agent_tensors


def rew_to_tens(rewards: dict, concatenate_all: bool = True) -> torch.Tensor:
    """
    Transforms a dictionary of rewards into a tensor.
    Either returns a tensor with the reward for each agent (len(rewards)), or sums the rewards into a scalar tensor.
    """
    if concatenate_all:
        return torch.tensor(sum(rewards.values()), dtype=torch.float32)
    else:
        return torch.tensor(list(rewards.values()), dtype=torch.float32)


def dict_to_tens(
    data: dict, concatenate_all: bool = True
) -> torch.Tensor | list[torch.Tensor]:
    """
    Transforms a dictionary into a tensor.
    If the value of the dictionary is also a dictionary, extracts the values to a tensor.
    Either concatenates everything into a single tensor, or returns a list of tensors.

    Args:
        data (dict): A dictionary where keys are identifiers, and values are either dictionaries or other values.
        concatenate_all (bool): If True, concatenates all values into a single tensor.
                                If False, returns a list of tensors, one per key.

    Returns:
        torch.Tensor or list of torch.Tensor: The transformed tensor(s).
    """
    tensors = []

    for key in data:
        value = data[key]
        if isinstance(value, dict):
            tensor = torch.tensor(list(value.values()), dtype=torch.float32)
        else:
            tensor = torch.tensor(value, dtype=torch.float32)
        tensors.append(tensor)
    print("\n", tensors)
    if concatenate_all:
        # unsqueeze if dimension == 0
        tensors = [t.unsqueeze(0) if t.dim() == 0 else t for t in tensors]
        tensors = torch.cat(tensors) if tensors else torch.tensor([])
    return tensors
