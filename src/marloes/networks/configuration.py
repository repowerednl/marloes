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

    def _validation(self):
        """
        Validation of the network configuration. We expect every instance to be [str, BaseNetwork].
        """
        for key, value in self.networks.items():
            if not isinstance(key, str):
                raise ValueError(f"Key {key} is not a string.")
            if not isinstance(value, BaseNetwork):
                raise ValueError(f"Value {value} is not a BaseNetwork.")

    """
    Below are the load and save functions that allow the configuration to be saved and loaded from a file.
    """

    def load(self, uid):
        """
        Loading available configurations for a given UID. If UID is not found, it will return an empty dictionary.
        It requires some specifics about the network architecture to be passed as well (layer details), this should be defined in the algorithm to match the saved configuration.
        """
        self._validation()
        if not os.path.exists(f"configs/{uid}"):
            raise ValueError(f"Configuration with UID {uid} not found.")
        for network in os.listdir(f"configs/{uid}"):
            if network not in self.networks:
                raise ValueError(f"Network type {network} not found in configuration.")
            path = f"configs/{uid}/{network}.pth"
            self.networks[network].load(path)

    def save(self, uid):
        """
        Saving the current configuration to separate files identified by the given UID.
        """
        self._validation()
        for network in self.networks:
            path = f"configs/{uid}/{network}.pth"
            # make sure configs/uid exists
            os.makedirs(f"configs/{uid}", exist_ok=True)
            self.networks[network].save(path)

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
