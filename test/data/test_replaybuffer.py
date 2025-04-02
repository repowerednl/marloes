from unittest import TestCase
import torch
from marloes.data.replaybuffer import ReplayBuffer


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

        self.RB_cpu.push(obs, action, reward)
        self.assertEqual(len(self.RB_cpu), 1)

        self.RB_gpu.push(obs, action, reward)
        self.assertEqual(len(self.RB_gpu), 1)

    def test_sample_random(self):
        obs = torch.tensor([1.0])
        action = torch.tensor([0.0])
        reward = torch.tensor([1.0])

        for _ in range(self.capacity):
            self.RB_cpu.push(obs, action, reward)

        sample = self.RB_cpu.sample(10, True)
        self.assertEqual(len(sample["obs"]), 10)
        self.assertEqual(len(sample["action"]), 10)
        self.assertEqual(len(sample["reward"]), 10)

    def test_sample_sequential(self):
        obs = torch.tensor([1.0])
        action = torch.tensor([0.0])
        reward = torch.tensor([0.0])

        for _ in range(self.capacity):
            self.RB_cpu.push(obs, action, reward)
            reward = torch.tensor([reward.item() + 1.0])

        size = 10

        sample = self.RB_cpu._sequential_sample(size, True)
        self.assertEqual(len(sample["obs"]), size)
        # also make sure the reward value is capacity - size, as most recent transitions are sampled
        self.assertEqual(sample["reward"][0].item(), self.capacity - size)

    def test_clear(self):
        self.RB_cpu.clear()
        self.assertEqual(len(self.RB_cpu), 0)

        self.RB_gpu.clear()
        self.assertEqual(len(self.RB_gpu), 0)
