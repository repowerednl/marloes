import unittest
import os
import torch
from marloes.networks.configuration import NetworkConfig
from marloes.networks.base import BaseNetwork


def get_state_dict():
    # create a dummy model and return state dict
    dummy = BaseNetwork()
    return dummy.state_dict()


class TestNetworkConfig(unittest.TestCase):
    def setUp(self):
        self.config = NetworkConfig()
        self.test_uid = "test_uid"
        self.test_path = f"configs/{self.test_uid}"
        self.test_data = {
            "network1": BaseNetwork(params=get_state_dict()),
            "network2": BaseNetwork(params=get_state_dict()),
        }

    def tearDown(self):
        if os.path.exists(self.test_path):
            for file in os.listdir(self.test_path):
                os.remove(os.path.join(self.test_path, file))
            os.rmdir(self.test_path)

    def test_set_and_get_item(self):
        self.config["network1"] = self.test_data["network1"]
        self.assertEqual(
            self.config["network1"].params, self.test_data["network1"].params
        )

    def test_del_item(self):
        self.config["network1"] = self.test_data["network1"]
        del self.config["network1"]
        with self.assertRaises(KeyError):
            _ = self.config["network1"]

    def test_len(self):
        self.config["network1"] = self.test_data["network1"]
        self.config["network2"] = self.test_data["network2"]
        self.assertEqual(len(self.config), 2)

    def test_load(self):
        os.makedirs(self.test_path, exist_ok=True)
        for key, network in self.test_data.items():
            torch.save(network.state_dict(), os.path.join(self.test_path, key))
        self.config.load(self.test_uid)
        for key in self.test_data:
            self.assertEqual(
                self.config.networks[key].params, self.test_data[key].params
            )

    def test_save(self):
        self.config.networks = self.test_data
        self.config.save(self.test_uid)
        for key in self.test_data:
            loaded_params = torch.load(os.path.join(self.test_path, key))
            self.assertEqual(loaded_params, self.test_data[key].state_dict())
