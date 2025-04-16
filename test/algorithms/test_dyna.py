import torch
from marloes.algorithms.dyna import Dyna


def test_combine_batches():
    real_batch = {
        "state": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "action": torch.tensor([[0.1], [0.2]]),
    }
    synthetic_batch = {
        "state": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        "action": torch.tensor([[0.3], [0.4]]),
    }

    combined_batch = Dyna._combine_batches(real_batch, synthetic_batch)

    # This is what the combined batch should look like
    assert torch.equal(
        combined_batch["state"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
    )
    assert torch.equal(
        combined_batch["action"], torch.tensor([[0.1], [0.2], [0.3], [0.4]])
    )
