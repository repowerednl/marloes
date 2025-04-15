import torch
import torch.nn as nn


class WorldStateEncoder(nn.Module):
    """
    World state encoder that combines the encodings of all agents and the global context.
    """

    def __init__(self, world_model_config: dict, num_agents: int, global_dim: int):
        super(WorldStateEncoder, self).__init__()
        agent_enc_dim = world_model_config.get("agent_enc_dim", 16)
        world_enc_dim = world_model_config.get("world_enc_dim", 64)
        hidden_size = world_model_config.get("world_hidden_size", 128)

        input_dim = num_agents * agent_enc_dim + global_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, world_enc_dim),
            nn.ReLU(),
        )

    def forward(self, agent_encodings, global_context=None):
        """
        Forward pass through the world state encoder.
        """
        # Concatenate agent encodings and global context
        all_agents = torch.cat(agent_encodings, dim=-1)
        if global_context is not None:
            combined = torch.cat([all_agents, global_context], dim=-1)
        else:
            combined = all_agents

        world_emb = self.mlp(combined)
        return world_emb
