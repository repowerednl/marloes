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
