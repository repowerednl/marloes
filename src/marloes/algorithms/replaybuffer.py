import random
from collections import deque, namedtuple
import torch

# Define a transition structure
Transition = namedtuple("Transition", ["obs", "action", "reward", "next_obs", "done"])


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

    def push(self, obs, action, reward, next_obs, done):
        """
        Stores a transition in the buffer.
        """
        self.buffer.append(Transition(obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        """
        Samples a random batch of transitions.
        Returns a dictionary of tensors.
        """
        batch = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*batch))  # Transpose batch to Transition of lists

        obs = torch.stack(batch.obs).to(self.device)
        action = torch.stack(batch.action).to(self.device)
        reward = torch.stack(batch.reward).to(self.device)
        next_obs = torch.stack(batch.next_obs).to(self.device)
        done = torch.stack(batch.done).to(self.device)

        return {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
        }

    def clear(self):
        """
        Clears the entire buffer.
        """
        self.buffer.clear()
