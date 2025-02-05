import unittest
import os
import yaml
from marloes.networks.configuration import NetworkConfig


class TestNetworkConfig(unittest.TestCase):
    def setUp(self):
        self.config = NetworkConfig()
        self.test_uid = "test_uid"
        self.test_path = f"results/network/config/{self.test_uid}.yaml"
        self.test_data = {"network1": "config1", "network2": "config2"}

    def tearDown(self):
        if os.path.exists(self.test_path):
            os.remove(self.test_path)

    def test_set_and_get_item(self):
        self.config["network1"] = "config1"
        self.assertEqual(self.config["network1"], "config1")

    def test_del_item(self):
        self.config["network1"] = "config1"
        del self.config["network1"]
        with self.assertRaises(KeyError):
            _ = self.config["network1"]

    def test_len(self):
        self.config["network1"] = "config1"
        self.config["network2"] = "config2"
        self.assertEqual(len(self.config), 2)

    def test_load(self):
        os.makedirs(os.path.dirname(self.test_path), exist_ok=True)
        with open(self.test_path, "w") as f:
            yaml.dump(self.test_data, f)
        self.config.load(self.test_uid)
        self.assertEqual(self.config.networks, self.test_data)

    def test_save(self):
        self.config.networks = self.test_data
        self.config.save(self.test_uid)
        with open(self.test_path, "r") as f:
            data = yaml.safe_load(f)
        self.assertEqual(data, self.test_data)
