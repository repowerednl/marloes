from unittest import TestCase
import torch
from marloes.algorithms.replaybuffer import ReplayBuffer


class ReplayBufferTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capacity = 100
        cls.RB_cpu = ReplayBuffer(capacity=cls.capacity, device="cpu")
        cls.RB_gpu = ReplayBuffer(capacity=cls.capacity, device="cuda")

    def test_initialization(self):
        self.assertEqual(len(self.RB_cpu), 0)
        self.assertEqual(len(self.RB_gpu), 0)
        self.assertEqual(self.RB_cpu.device, "cpu")
        self.assertEqual(self.RB_gpu.device, "cuda")

    def test_push_and_length(self):
        obs = torch.tensor([1.0])
        action = torch.tensor([0.0])
        reward = torch.tensor([1.0])
        next_obs = torch.tensor([2.0])
        done = torch.tensor([0.0])

        self.RB_cpu.push(obs, action, reward, next_obs, done)
        self.assertEqual(len(self.RB_cpu), 1)

        self.RB_gpu.push(obs, action, reward, next_obs, done)
        self.assertEqual(len(self.RB_gpu), 1)

    def test_sample(self):
        obs = torch.tensor([1.0])
        action = torch.tensor([0.0])
        reward = torch.tensor([1.0])
        next_obs = torch.tensor([2.0])
        done = torch.tensor([0.0])

        for _ in range(self.capacity):
            self.RB_cpu.push(obs, action, reward, next_obs, done)

        sample = self.RB_cpu.sample(10)
        self.assertEqual(len(sample["obs"]), 10)
        self.assertEqual(len(sample["action"]), 10)
        self.assertEqual(len(sample["reward"]), 10)
        self.assertEqual(len(sample["next_obs"]), 10)
        self.assertEqual(len(sample["done"]), 10)

    def test_clear(self):
        self.RB_cpu.clear()
        self.assertEqual(len(self.RB_cpu), 0)

        self.RB_gpu.clear()
        self.assertEqual(len(self.RB_gpu), 0)
