import os
import yaml


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
            Loads the network configuration from a file identified by the given UID.
        save(uid):
            Saves the current network configuration to a file identified by the given UID.
    """

    def __init__(self):
        self.networks = {}

    """
    Below are the load and save functions that allow the configuration to be saved and loaded from a file.
    """

    def load(self, uid):
        path = f"results/network/config/{uid}.yaml"
        if os.path.exists(path):
            with open(path, "r") as f:
                self.networks = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"No configuration found for UID: {uid}")

    def save(self, uid):
        path = f"results/network/config/{uid}.yaml"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.networks, f)

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
