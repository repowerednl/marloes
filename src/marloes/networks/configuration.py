import os
import torch
from collections import defaultdict
from .base import BaseNetwork


class NetworkConfig:
    """
    A class to manage network configurations, allowing it to be used as a dictionary
    and providing functionality to load and save configurations from/to a file.
    Attributes:
        networks (dict): A dictionary to store network configurations.
    Methods:
        __getitem__(key):
            Returns the value associated with the given key in the networks dictionary.
        __setitem__(key, value):
            Sets the value for the given key in the networks dictionary.
        __delitem__(key):
            Deletes the key-value pair associated with the given key from the networks dictionary.
        __iter__():
            Returns an iterator over the keys of the networks dictionary.
        __len__():
            Returns the number of key-value pairs in the networks dictionary.
        load(uid):
            Loads the network configuration from a file identified by the given UID and config type.
        save(uid):
            Saves the current network configuration to a file identified by the given UID and config type.
    """

    def __init__(self):
        self.networks = defaultdict()

    """
    Below are the load and save functions that allow the configuration to be saved and loaded from a file.
    """

    def load(self, uid):
        """
        Loading available configurations for a given UID. If UID is not found, it will return an empty dictionary.
        """
        if not os.path.exists(f"configs/{uid}"):
            return {}
        for config_type in os.listdir(f"configs/{uid}"):
            path = f"configs/{uid}/{config_type}"
            self.networks[config_type] = BaseNetwork(params=torch.load(path))

    def save(self, uid):
        """
        Saving the current configuration to separate files identified by the given UID.
        """
        for config_type in self.networks:
            path = f"configs/{uid}/{config_type}"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.networks[config_type].state_dict(), path)

    """
    Below is functionality that allows the NetworkConfig class to be used as a dictionary.
    """

    def __getitem__(self, key):
        return self.networks[key]

    def __setitem__(self, key, value):
        self.networks[key] = value

    def __delitem__(self, key):
        del self.networks[key]

    def __iter__(self):
        return iter(self.networks)

    def __len__(self):
        return len(self.networks)
