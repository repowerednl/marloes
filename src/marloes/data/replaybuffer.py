import random
from collections import deque, namedtuple
import torch

# Define a transition structure
Transition = namedtuple("Transition", ["state", "actions", "rewards", "next_state"])


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

    def push(self, state, actions, rewards, next_state):
        """
        Stores a transition in the buffer.
        """
        self.buffer.append(Transition(state, actions, rewards, next_state))

    def sample(self, batch_size: int, sequence: int = 0):
        """
        Samples a random batch of transitions, uses the most recent transitions if not random, unless specified otherwise.
        Args:
            batch_size (int): Number of transitions to sample.
            sequence (int): If > 0, samples a sequence of transitions.
                            If 0, samples random transitions.
        """
        if sequence:
            # should sample [batch_size] sequences of size [sequence]
            return self._sequential_sample(batch_size, horizon=sequence)
        return self._random_sample(batch_size)

    def _random_sample(self, batch_size: int):
        if batch_size > len(self.buffer):
            raise ValueError("Not enough elements in buffer for random sample.")

        # Randomly sample
        transitions = random.sample(self.buffer, batch_size)

        # Convert
        return self._convert_to_tensors(transitions)

    def _sequential_sample(self, batch_size: int, horizon: int = 1) -> list[dict]:
        """
        Samples sequences from the buffer, returns a batch of size [batch_size] with each element being a sequence of size [horizon].
        """
        if batch_size > len(self.buffer):
            raise ValueError("Not enough elements in buffer for sequential sample.")
        if horizon > len(self.buffer):
            raise ValueError("Horizon is larger than buffer size.")
        if horizon == 0:
            raise ValueError("Horizon cannot be zero.")
        batch = []
        max_idx = len(self.buffer) - horizon
        for b in range(batch_size):
            # randomly select a starting index
            start_idx = random.randint(0, max_idx)
            sequence = [self.buffer[i] for i in range(start_idx, start_idx + horizon)]
            batch.append(self._convert_to_tensors(sequence))
        return batch

    def _convert_to_tensors(self, transitions):
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []

        for tr in transitions:
            state_list.append(self.dict_to_tens(tr.state))
            action_list.append(self.dict_to_tens(tr.actions))
            reward_list.append(self.dict_to_tens(tr.rewards))
            next_state_list.append(self.dict_to_tens(tr.next_state))

        state = torch.stack(state_list).to(self.device)
        action = torch.stack(action_list).to(self.device)
        reward = torch.stack(reward_list).to(self.device)
        next_state = torch.stack(next_state_list).to(self.device)

        return {
            "state": state,
            "actions": action,
            "rewards": reward,
            "next_state": next_state,
        }

    def clear(self):
        """
        Clears the entire buffer.
        """
        self.buffer.clear()

    @staticmethod
    def dict_to_tens(
        data: dict, concatenate_all: bool = True
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Transforms a dictionary into a tensor.
        If the value of the dictionary is also a dictionary, extracts the values to a tensor.
        Either concatenates everything into a single tensor, or returns a list of tensors.
        Args:
            data (dict): A dictionary where keys are identifiers, and values are either dictionaries or other values.
            concatenate_all (bool): If True, concatenates all values into a single tensor.
                                    If False, returns a list of tensors, one per key.
        Returns:
            torch.Tensor or list of torch.Tensor: The transformed tensor(s).
        """

        def recursive_tensor_extraction(value):
            """
            Recursively extracts tensors from a dictionary.
            """
            if isinstance(value, dict):
                tensors = [recursive_tensor_extraction(v) for v in value.values()]
                tensors = [t.unsqueeze(0) if t.dim() == 0 else t for t in tensors]
                return torch.cat(tensors) if tensors else torch.tensor([])
            else:
                return torch.tensor(value, dtype=torch.float32)

        tensors = [recursive_tensor_extraction(value) for value in data.values()]

        if concatenate_all:
            tensors = [t.unsqueeze(0) if t.dim() == 0 else t for t in tensors]
            tensors = torch.cat(tensors) if tensors else torch.tensor([])

        return tensors
