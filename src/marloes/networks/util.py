import torch
import numpy as np


def dist(mu, logvar):
    """
    Reparametrization trick to create a stochastic latent state, using mu and logvar for the distribution.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


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

    def recursive_tensor_extraction(value):
        """
        Recursively extracts tensors from a dictionary.
        """
        if isinstance(value, dict):
            tensors = [recursive_tensor_extraction(v) for v in value.values()]
            tensors = [t.unsqueeze(0) if t.dim() == 0 else t for t in tensors]
            return torch.cat(tensors) if tensors else torch.tensor([])
        else:
            return torch.tensor(value, dtype=torch.float32)

    tensors = [recursive_tensor_extraction(value) for value in data.values()]

    if concatenate_all:
        tensors = [t.unsqueeze(0) if t.dim() == 0 else t for t in tensors]
        tensors = torch.cat(tensors) if tensors else torch.tensor([])

    return tensors


def symlog_squared_loss(x, y):
    """
    Symlog squared loss function for Prediction loss in the World Model.
    """

    def symlog(x):
        return torch.sign(x) * torch.log1p(torch.abs(x))

    return torch.nn.functional.mse_loss(symlog(x), symlog(y))


def compute_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.997,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """
    Compute λ-returns (bootstrapped n-step returns) from rewards and value estimates.

    Args:
        rewards: Tensor of shape (T, B) with rewards at each time step.
        values: Tensor of shape (T, B) with critic value estimates.
        gamma: Discount factor.
        lambda_: Mixing parameter for multi-step bootstrapping.

    Returns:
        Tensor of shape (T, B) with computed λ-returns.
    """
    T = rewards.size(0)
    returns = torch.zeros_like(rewards)
    # Bootstrap from the final value.
    returns[-1] = values[-1]
    # Compute returns recursively from the end of the trajectory.
    for t in reversed(range(T - 1)):
        # The lambda-return is: r_t + gamma * [(1 - lambda_) * v_{t+1} + lambda_ * R_{t+1}]
        returns[t] = rewards[t] + gamma * (
            (1 - lambda_) * values[t + 1] + lambda_ * returns[t + 1]
        )
    return returns


def gaussian_kl_divergence(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the KL divergence between two Gaussians with parameters (mu, logvar).
    All should be tensors of shape.
    """
    kl = 0.5 * (
        logvar_p
        - logvar_q
        + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
        - 1
    )
    return kl  # shape (batch, latent_dim)


def kl_free_bits(kl: torch.Tensor, free_bits: float = 1.0) -> torch.Tensor:
    """
    Adjusts the kl-divergence penalizing KL values above the threshold.
    """
    adjusted = torch.clamp(kl - free_bits, min=0.0)
    return adjusted.sum(dim=-1).mean()
