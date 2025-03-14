from unittest import TestCase, mock
import numpy as np
import torch

from marloes.networks.base import HyperParams
from marloes.networks.RSSM import RSSM
from marloes.networks.util import observation_to_tensor
from marloes.networks.details import RSSM_LD


class RSSMTestCase(TestCase):
    """
    TestCase for the RSSM network, to test if creation is flexible and forward pass goes through all parts.
    """

    @classmethod
    def setUpClass(cls):
        # create a random (new) RSSM network
        cls.existing_rssm = RSSM()
        cls.loaded_params = cls.existing_rssm.state_dict()
        cls.hyper_params = HyperParams(
            lr=0.002,
        )

    def test_normal_creation(self):
        """
        Tests if the existed RSSM is as expected.
        """
        self.assertIsInstance(self.existing_rssm.rnn, torch.nn.GRU)
        self.assertIsInstance(self.existing_rssm.fc, torch.nn.Linear)
        self.assertEqual(
            self.existing_rssm.rnn.hidden_size, self.existing_rssm.fc.in_features
        )
        self.assertEqual(
            self.existing_rssm.fc.out_features, RSSM_LD.hidden["dense"]["out_features"]
        )
        # test parameters
        self.assertEqual(self.existing_rssm.optimizer.param_groups[0]["lr"], 0.001)
        self.assertIsInstance(self.existing_rssm.loss, torch.nn.MSELoss)

    def test_new_rssm_creation_with_hyperparams(self):
        """
        Test if the RSSM network is created correctly.
        """
        rssm = RSSM(hyper_params=self.hyper_params)
        self.assertIsInstance(rssm.rnn, torch.nn.GRU)
        self.assertIsInstance(rssm.fc, torch.nn.Linear)
        self.assertEqual(rssm.rnn.hidden_size, rssm.fc.in_features)
        self.assertEqual(rssm.fc.out_features, RSSM_LD.hidden["dense"]["out_features"])
        # test parameters
        self.assertEqual(rssm.optimizer.param_groups[0]["lr"], self.hyper_params.lr)
        self.assertIsInstance(rssm.loss, torch.nn.MSELoss)

    def test_creation_from_params(self):
        """
        Test if the RSSM network is created correctly from loaded params.
        """
        with mock.patch.object(
            RSSM, "_load_from_params", wraps=self.existing_rssm._load_from_params
        ) as mock_load:
            rssm = RSSM(params=self.loaded_params)
            self.assertIsInstance(rssm.rnn, torch.nn.GRU)
            self.assertIsInstance(rssm.fc, torch.nn.Linear)
            self.assertEqual(rssm.rnn.hidden_size, rssm.fc.in_features)
            self.assertEqual(
                rssm.fc.out_features, RSSM_LD.hidden["dense"]["out_features"]
            )
            # test parameters
            self.assertEqual(rssm.optimizer.param_groups[0]["lr"], 0.001)
            self.assertIsInstance(rssm.loss, torch.nn.MSELoss)
            # make sure _load_from_params is called
            mock_load.assert_called_once_with(self.loaded_params)

    def test_forward(self):
        """
        Test if the forward pass goes through the RSSM network without problems.
        """
        rssm = RSSM()
        # input should be of size RSSM_LD.hidden["recurrent"]["input_size"] (256 + 64 + 6 right now) = torch.cat(h_t, z_t, a_t)
        a_t = torch.randn(1, 1, 6)  # 6 agents
        h_t = torch.randn(1, 1, 256)
        z_t = torch.randn(1, 1, 64)

        # forward pass
        h_t, z_hat_t = rssm(h_t, z_t, a_t)
        # check output shapes
        self.assertEqual(h_t.shape, (1, 1, 256))
        self.assertEqual(z_hat_t.shape, (1, 1, 64))

    def test_forward_wrong_input(self):
        """
        Test if the forward pass raises an assertion error with incorrect input sizes.
        """
        rssm = RSSM()
        a_t = torch.randn(1, 1, 5)  # Incorrect size for a_t
        h_t = torch.randn(1, 1, 256)
        z_t = torch.randn(1, 1, 64)

        with self.assertRaises(AssertionError) as context:
            rssm(h_t, z_t, a_t)

        self.assertIn(
            "Combined input size does not match the RNN input size",
            str(context.exception),
        )
