import unittest

import torch
from marloes.data.replaybuffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.capacity = 5
        self.device = "cpu"
        self.buffer = ReplayBuffer(capacity=self.capacity, device=self.device)

    def sample_transition(self, i):
        state = {"value": i}
        actions = {"value": i + 0.1}
        rewards = {"value": i + 0.2}
        next_state = {"value": i + 1}
        return state, actions, rewards, next_state

    def test_push(self):
        self.assertEqual(len(self.buffer), 0)
        # Push one
        self.buffer.push(*self.sample_transition(0))
        self.assertEqual(len(self.buffer), 1)
        # Push a few more
        for i in range(1, 4):
            self.buffer.push(*self.sample_transition(i))
        self.assertEqual(len(self.buffer), 4)

    def test_random_sample(self):
        for i in range(5):
            self.buffer.push(*self.sample_transition(i))
        batch = self.buffer.sample(batch_size=3)

        for key in ["state", "actions", "rewards", "next_state"]:
            self.assertIn(key, batch)
            tensor = batch[key]
            # Check that the batch dimension is 3
            self.assertEqual(tensor.shape[0], 3)
            # Verify the tensor is on the correct device
            self.assertEqual(tensor.device.type, self.device)

    def test_sequential_sample(self):
        for i in range(6):
            self.buffer.push(*self.sample_transition(i))
        batch = self.buffer.sample(batch_size=2, sequence=2)
        self.assertEqual(len(batch), 2)
        for sequence in batch:
            # sequence is a dictionary
            self.assertIn("state", sequence)
            self.assertIn("actions", sequence)
            self.assertIn("rewards", sequence)
            self.assertIn("next_state", sequence)
            # check that the values have 2 elements
            self.assertEqual(sequence["state"].shape[0], 2)

    def test_clear_buffer(self):
        for i in range(3):
            self.buffer.push(*self.sample_transition(i))
        self.assertEqual(len(self.buffer), 3)
        self.buffer.clear()
        self.assertEqual(len(self.buffer), 0)

    def test_dict_to_tens(self):
        # Test dict_to_tens with a simple dict
        data = {"a": 1, "b": 2}
        tensor = ReplayBuffer.dict_to_tens(data, concatenate_all=True)
        expected = torch.tensor([1.0, 2.0])
        self.assertTrue(
            torch.equal(tensor, expected), f"Expected {expected}, got {tensor}"
        )
        integer_value_reward = 5
        tensor = ReplayBuffer.dict_to_tens(integer_value_reward)
        expected = torch.tensor([5.0])
