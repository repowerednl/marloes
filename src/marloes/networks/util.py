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
