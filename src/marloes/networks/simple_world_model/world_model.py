import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from marloes.networks.simple_world_model.agent_state_encoder import AgentStateEncoder
from marloes.networks.simple_world_model.util import (
    parse_actions,
    parse_batch,
    parse_state,
)
from marloes.networks.simple_world_model.world_dynamics import WorldDynamicsModel
from marloes.networks.simple_world_model.world_state_encoder import WorldStateEncoder


class WorldModel(nn.Module):
    """
    World model that combines all components.
    """

    def __init__(self, config: dict):
        super(WorldModel, self).__init__()
        self.world_model_config = config.get("WorldModel", {})
        self.num_agents = config["num_agents"]

        self.agent_state_encoders = nn.ModuleList(
            [AgentStateEncoder(self.world_model_config) for _ in range(self.num_agents)]
        )
        self.world_state_encoder = WorldStateEncoder(
            self.world_model_config, self.num_agents, config.get("global_dim", 0)
        )
        self.world_dynamics_model = WorldDynamicsModel(
            self.world_model_config, config["action_dim"]
        )
        self.optimizer = Adam(
            self.parameters(),
            lr=self.world_model_config.get("lr", 1e-3),
            weight_decay=self.world_model_config.get("weight_decay", 0.0),
        )
        self.loss = []

    def forward(self, scalar_list, forecast_list, global_context, actions):
        # Encode each agent
        agent_encodings = []
        for i in range(self.num_agents):
            agent_encoding = self.agent_state_encoders[i](
                scalar_list[i], forecast_list[i]
            )
            agent_encodings.append(agent_encoding)

        # Aggregate in the world state encoder
        world_state = self.world_state_encoder(agent_encodings, global_context)

        # Predict next state and reward
        next_state, reward = self.world_dynamics_model(world_state, actions)

        return next_state, reward

    def predict(self, states: list[dict], actions: list[dict], device: str = "cpu"):
        """
        Given environment states and actions, predict the next state and reward.
        """
        # Convert original formats to separable tensors
        parsed_states = parse_state(states, device=device)
        parsed_actions = parse_actions(actions, device=device)

        # Extract scalar variables and forecast from states
        scalar_list = [agent["scalars"] for agent in parsed_states["agents"]]
        forecast_list = [agent["forecast"] for agent in parsed_states["agents"]]

        # Extract global context if available
        global_context = None
        if "global_context" in parsed_states:
            global_context = parsed_states["global_context"]

        # Forward pass through the model
        next_states, rewards = self.forward(
            scalar_list, forecast_list, global_context, parsed_actions
        )

        # Reconvert next state to original format
        reconstructed_next_states = [
            self._reconstruct_state(curr, pred)
            for curr, pred in zip(states, next_states)
        ]

        return reconstructed_next_states, rewards

    def update(self, transitions_batch):
        """
        Update the world model using a batch of real transitions.
        """
        # 1. Parse the batch
        parsed_batch = parse_batch(transitions_batch)
        scalar_list = [agent["scalars"] for agent in parsed_batch["state"]["agents"]]
        forecast_list = [agent["forecast"] for agent in parsed_batch["state"]["agents"]]
        global_context = parsed_batch["state"].get("global_context", None)
        actions = parsed_batch["actions"]
        rewards = parsed_batch["rewards"]
        next_state_target = parsed_batch["next_state"]

        # 2. Forward pass
        latent_next_state, reward_pred = self.forward(
            scalar_list, forecast_list, global_context, actions
        )

        # 3. Compute loss (for example, MSE on state and reward)
        state_loss = F.mse_loss(latent_next_state, next_state_target)
        reward_loss = F.mse_loss(reward_pred, rewards)
        loss = state_loss + reward_loss

        # 4. Backpropagate and update all parameters (all submodules are updated automatically)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 5. Store loss for logging
        self.loss.append(loss.item())

    def _reconstruct_state(self, state, next_state):
        new_state = {}

        # Process global context first
        if "global_context" in state:
            if "global_context" in next_state:
                new_state["global_context"] = next_state["global_context"]
            else:
                new_state["global_context"] = state["global_context"]

        # Process each agent
        for agent, agent_data in state.items():
            if agent == "global_context":
                continue  # already processed
            new_agent_data = {}
            for key, value in agent_data.items():
                if key == "forecast":
                    # Shift the forecast:
                    # Assume forecast is either a list or a numpy array.
                    if isinstance(value, list):
                        new_forecast = value[1:] + [0.0]
                    else:
                        import numpy as np

                        new_forecast = np.roll(value, -1)
                        new_forecast[-1] = 0.0
                    new_agent_data["forecast"] = new_forecast
                else:
                    # For non-forecast keys, use the predicted value if provided, else keep current.
                    if agent in next_state and key in next_state[agent]:
                        new_agent_data[key] = next_state[agent][key]
                    else:
                        new_agent_data[key] = value
            new_state[agent] = new_agent_data

        return new_state
