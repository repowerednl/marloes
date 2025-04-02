import random
from collections import deque, namedtuple
import torch

# Define a transition structure
Transition = namedtuple("Transition", ["obs", "action", "reward"])


class ReplayBuffer:
    def __init__(self, capacity: int = 10000, device="cpu"):
        """
        Initializes the replay buffer.

        Args:
            capacity (int): Maximum number of transitions to store.
            device (str): Device to move tensors to when sampling.
        """
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def __len__(self):
        return len(self.buffer)

    def push(self, obs, action, reward):
        """
        Stores a transition in the buffer.
        """
        self.buffer.append(Transition(obs, action, reward))

    def sample(
        self, batch_size: int, random: bool = False, use_most_recent: bool = True
    ):
        """
        Samples a random batch of transitions, uses the most recent transitions if not random, unless specified otherwise.
        """
        return (
            self._random_sample(batch_size)
            if random
            else self._sequential_sample(batch_size, use_most_recent)
        )

    def _random_sample(self, batch_size: int):
        """
        Random sampling.
        """
        batch = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*batch))  # Transpose batch to Transition of lists

        obs = torch.stack(batch.obs).to(self.device)
        action = torch.stack(batch.action).to(self.device)
        reward = torch.stack(batch.reward).to(self.device)

        return {
            "obs": obs,
            "action": action,
            "reward": reward,
        }

    def _sequential_sample(self, batch_size: int, use_most_recent: bool = True):
        """
        Retrieves consecutive transitions more efficiently.
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Not enough transitions to sample a sequential batch of length {batch_size}"
            )

        start_idx = (
            len(self.buffer) - batch_size
            if use_most_recent
            else random.randint(0, len(self.buffer) - batch_size)
        )

        # Use slicing to retrieve consecutive transitions
        sequence = list(self.buffer)[start_idx : start_idx + batch_size]

        # Transpose and convert to tensors in one step
        obs, action, reward = map(
            lambda x: torch.stack(x).to(self.device), zip(*sequence)
        )

        return {
            "obs": obs,
            "action": action,
            "reward": reward,
        }

    def clear(self):
        """
        Clears the entire buffer.
        """
        self.buffer.clear()
