import torch
import torch.nn as nn


class WorldStateEncoder(nn.Module):
    """
    World state encoder that combines the encodings of all handlers and the global context.

    This module takes the encodings of individual handlers and optionally a global context,
    processes them through a multi-layer perceptron (MLP), and outputs a fixed-size
    representation of the world state.

    Attributes:
        mlp (nn.Sequential): MLP for processing the concatenated handler encodings and global context.
    """

    def __init__(
        self, world_model_config: dict, num_handlers: int, global_dim: int
    ) -> None:
        """
        Initialize the WorldStateEncoder.

        Args:
            world_model_config (dict): Configuration dictionary containing:
                - "asset_enc_dim" (int, optional): Dimension of each handler's encoding (default: 16).
                - "world_enc_dim" (int, optional): Dimension of the output world state encoding (default: 64).
                - "world_hidden_size" (int, optional): Hidden size of the MLP (default: 128).
            num_handlers (int): Number of handlers in the environment.
            global_dim (int): Dimension of the global context.
        """
        super(WorldStateEncoder, self).__init__()
        asset_enc_dim = world_model_config.get("asset_enc_dim", 16)
        world_enc_dim = world_model_config.get("world_enc_dim", 64)
        hidden_size = world_model_config.get("world_hidden_size", 128)

        input_dim = num_handlers * asset_enc_dim + global_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, world_enc_dim),
            nn.ReLU(),
        )

    def forward(
        self, handler_encodings: list[torch.Tensor], global_context: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Perform a forward pass through the world state encoder.

        Args:
            handler_encodings (list[torch.Tensor]): List of tensors representing the encodings of individual handlers.
                Each tensor should have shape (batch_size, asset_enc_dim).
            global_context (torch.Tensor, optional): Tensor representing the global context, with shape
                (batch_size, global_dim). If None, only handler encodings are used.

        Returns:
            torch.Tensor: Encoded world state tensor of shape (batch_size, world_enc_dim).
        """
        # Concatenate handler encodings and global context
        all_handlers = torch.cat(handler_encodings, dim=-1)
        if global_context is not None:
            combined = torch.cat([all_handlers, global_context], dim=-1)
        else:
            combined = all_handlers

        world_emb = self.mlp(combined)
        return world_emb
