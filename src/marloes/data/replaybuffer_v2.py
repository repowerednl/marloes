import random
from collections import deque, namedtuple
import torch

# Define a transition structure
Transition = namedtuple("Transition", ["state", "actions", "rewards", "next_state"])


class ReplayBufferV2:
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

    def push(self, state, actions, rewards, next_state):
        """
        Stores a transition in the buffer.
        """
        self.buffer.append(Transition(state, actions, rewards, next_state))

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

        state = torch.stack(batch.state).to(self.device)
        actions = torch.stack(batch.actions).to(self.device)
        rewards = torch.stack(batch.rewards).to(self.device)
        next_state = torch.stack(batch.next_state).to(self.device)

        return {
            "state": state,
            "actions": actions,
            "rewards": rewards,
            "next_state": next_state,
        }

    def _sequential_sample(self, batch_size: int, use_most_recent: bool = True):
        """
        Retrieves consecutive transitions.
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Not enough transitions to sample a sequential batch of length {batch_size}"
            )

        if use_most_recent:
            start_idx = len(self.buffer) - batch_size
        else:
            # Pick a random start index where a consecutive sequence of length batch_size can be retrieved.
            start_idx = random.randint(0, len(self.buffer) - batch_size)

        # Retrieve consecutive transitions
        sequence = [self.buffer[i] for i in range(start_idx, start_idx + batch_size)]

        # Transpose the sequence of transitions into a Transition of lists.
        batch = Transition(*zip(*sequence))

        # Convert each component into a tensor and move to the proper device.
        state = torch.stack(batch.state).to(self.device)
        actions = torch.stack(batch.actions).to(self.device)
        rewards = torch.stack(batch.rewards).to(self.device)
        next_state = torch.stack(batch.next_state).to(self.device)

        return {
            "state": state,
            "actions": actions,
            "rewards": rewards,
            "next_state": next_state,
        }

    def clear(self):
        """
        Clears the entire buffer.
        """
        self.buffer.clear()
