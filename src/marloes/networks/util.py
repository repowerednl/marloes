import torch
import numpy as np


def dist(mu, raw_logvar):
    # Apply softplus to ensure positivity and avoid extreme small values.
    logvar = torch.nn.functional.softplus(raw_logvar) + 1e-6

    # Clamp logvar to ensure it remains within a reasonable range for numerical stability.
    logvar = torch.clamp(logvar, min=-5.0, max=5.0)
    # Compute standard deviation (std) from the logvar.
    std = torch.exp(0.5 * logvar)

    # Generate random noise from a normal distribution.
    eps = torch.randn_like(std)
    # Return the sampled values.
    return mu + eps * std


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

    Expects rewards and values of shape (B, T, 1), where B is the batch size and T is the trajectory length.

    Returns:
        Tensor of shape (B, T, 1) with computed λ-returns.
    """
    B, T, _ = rewards.shape
    # Initialize returns with the same shape as rewards.
    returns = torch.zeros_like(rewards)
    # Bootstrap from the final value.
    returns[:, -1, :] = values[:, -1, :]
    # Compute returns recursively from the second-last time step down to t=0.
    for t in range(T - 2, -1, -1):
        # The lambda-return is computed per batch element:
        # returns[t] = rewards[t] + gamma * ((1 - lambda_) * values[t+1] + lambda_ * returns[t+1])
        returns[:, t, :] = rewards[:, t, :] + gamma * (
            (1 - lambda_) * values[:, t + 1, :] + lambda_ * returns[:, t + 1, :]
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
