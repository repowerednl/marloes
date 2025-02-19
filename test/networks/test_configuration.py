import unittest
import os
import torch
from marloes.networks.configuration import NetworkConfig
from marloes.networks.base import BaseNetwork, LayerDetails

from test.util import get_valid_basenetwork


class TestNetworkConfig(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.config = NetworkConfig()
        cls.test_uid = "test_uid"
        cls.test_path = f"configs/{cls.test_uid}"
        cls.test_data = {
            "network1": get_valid_basenetwork(),
            "network2": get_valid_basenetwork(),
        }

    def tearDown(self):
        if os.path.exists(self.test_path):
            for file in os.listdir(self.test_path):
                os.remove(os.path.join(self.test_path, file))
            os.rmdir(self.test_path)

    def test_set_and_get_item(self):
        self.config["network1"] = self.test_data["network1"]
        # check if the parameters of the network are the same (with torch)
        for key in self.config["network1"].state_dict():
            self.assertTrue(
                torch.equal(
                    self.config["network1"].state_dict()[key],
                    self.test_data["network1"].state_dict()[key],
                )
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
        """
        Testing load function, to be extended with correct layer details.
        """
        with self.assertRaises(ValueError):
            self.config.load(self.test_uid, None)

    def test_save(self):
        """
        Testing save function, to be extended with correct layer details.
        """
        self.config["network1"] = self.test_data["network1"]
        self.config["network2"] = self.test_data["network2"]
        self.config.save(self.test_uid)
        self.assertTrue(os.path.exists(self.test_path))
        for network in self.config:
            self.assertTrue(os.path.exists(f"{self.test_path}/{network}.pth"))

        # Clean up
        for file in os.listdir(self.test_path):
            os.remove(os.path.join(self.test_path, file))
        os.rmdir(self.test_path)
