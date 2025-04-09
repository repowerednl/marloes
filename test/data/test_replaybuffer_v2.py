import unittest

import torch
from marloes.data.replaybuffer_v2 import ReplayBufferV2


class TestReplayBufferV2(unittest.TestCase):
    def setUp(self):
        self.capacity = 5
        self.device = "cpu"
        self.buffer = ReplayBufferV2(capacity=self.capacity, device=self.device)

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
        batch = self.buffer.sample(batch_size=3, random=True)

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
        batch = self.buffer.sample(batch_size=4, random=False, use_most_recent=True)
        state_tensor = batch["state"]
        state_values = state_tensor.squeeze().tolist()
        if not isinstance(state_values, list):
            state_values = [state_values]
        self.assertEqual(state_values, [2, 3, 4, 5])

    def test_clear_buffer(self):
        for i in range(3):
            self.buffer.push(*self.sample_transition(i))
        self.assertEqual(len(self.buffer), 3)
        self.buffer.clear()
        self.assertEqual(len(self.buffer), 0)

    def test_dict_to_tens(self):
        # Test dict_to_tens with a simple dict
        data = {"a": 1, "b": 2}
        tensor = ReplayBufferV2.dict_to_tens(data, concatenate_all=True)
        expected = torch.tensor([1.0, 2.0])
        self.assertTrue(
            torch.equal(tensor, expected), f"Expected {expected}, got {tensor}"
        )
