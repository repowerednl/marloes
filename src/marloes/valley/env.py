"""
Environment that holds all necessary information for the Simulation, called EnergyValley
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo
import numpy as np
import torch

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from simon.solver import Model

from marloes.agents import (
    Agent,
    BatteryAgent,
    CurtailmentAgent,
    DemandAgent,
    ElectrolyserAgent,
    GridAgent,
    SolarAgent,
    WindAgent,
)
from marloes.algorithms.util import get_net_forecasted_power
from marloes.data.util import encode_datetime
from marloes.results.extractor import ExtensiveExtractor, Extractor
from marloes.data.replaybuffer import ReplayBuffer
from marloes.valley.rewards.reward import Reward


class EnergyValley(MultiAgentEnv):
    """
    Environment that holds all necessary information for the simulation.
    """

    AGENT_TYPE_MAP = {
        "battery": BatteryAgent,
        "electrolyser": ElectrolyserAgent,
        "demand": DemandAgent,
        "solar": SolarAgent,
        "wind": WindAgent,
    }
    EXTRACTOR_MAP = {
        "default": Extractor,
        "extensive": ExtensiveExtractor,
    }

    def __init__(self, config: dict, algorithm_type: str):
        """
        Initializes the environment, agents, and solver model.
        """
        super().__init__()
        default_start_time = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
        self.start_time = config.get("simulation_start_time", default_start_time)
        self.time_stamp = self.start_time
        self.time_step = 60  # 1 minute in seconds

        self.agents: list[Agent] = []
        self.trainable_agents: list[Agent] = []
        self.grid: Optional[GridAgent] = None
        self.model: Optional[Model] = None
        self.extractor: Extractor = self.EXTRACTOR_MAP[
            config.pop("extractor_type", "default")
        ]()
        self.reward = Reward(config, actual=True, **config.get("subrewards", {}))

        self._initialize_agents(config, algorithm_type, config.get("data_config", {}))
        self._initialize_model(
            algorithm_type
        )  # Model has a graph (nx.DiGraph) with assets as nodes and edges as connections

        # For efficiency
        self.agent_dict = {agent.id: agent for agent in self.agents}
        self.trainable_agent_dict = {agent.id: agent for agent in self.trainable_agents}
        self._state_cache = {agent.id: None for agent in self.agents}
        # is the hub ever done? Is life ever done? Is life a simulation?
        self._dones_cache = {agent.id: False for agent in self.agents}
        self._infos_cache = {agent.id: {} for agent in self.agents}

        # Add dims to the environment
        self.state_dim = ReplayBuffer.dict_to_tens(self._get_full_observation()).shape
        self.action_dim = torch.Size([len(self.trainable_agents)])
        self.global_dim = ReplayBuffer.dict_to_tens(
            self._get_global_context(self._combine_states())
        ).shape
        self.agents_scalar_dim = [
            len(state) for state in self._combine_states(False).values()
        ]
        self.forecasts = [agent.forecast is not None for agent in self.agents]

    def _initialize_agents(
        self, config: dict, algorithm_type: str, data_config: dict
    ) -> None:
        """
        Function to initialize all agents with the given configuration.
        Requires config with "agents" key (list of dicts), and "grid" key (dict).
        """
        logging.info("Adding agents to the environment...")

        # Add the grid agent and curtailment agent to agents
        self.grid = GridAgent(config=config.get("grid", {}), start_time=self.start_time)
        self.agents.append(self.grid)
        if algorithm_type == "Priorities":
            self.agents.append(CurtailmentAgent({}, self.start_time))

        for agent_config in config.get("agents", []):
            self._add_agent(agent_config, data_config)

    def _add_agent(self, agent_config: dict, data_config: dict) -> None:
        """
        Adds an agent based on its type from the configuration.
        """
        agent_type = agent_config.pop("type", None)
        if agent_type not in self.AGENT_TYPE_MAP:
            raise ValueError(f"Unknown agent type: '{agent_type}'")

        agent_class = self.AGENT_TYPE_MAP[agent_type]
        trainable = agent_config.pop("trainable", True)
        self.agents.append(
            agent_class(agent_config, self.start_time, data_config=data_config)
        )
        if trainable:
            self.trainable_agents.append(self.agents[-1])

    def _initialize_model(self, algorithm_type: str) -> None:
        """
        Function to initialize the Model imported from Simon.
        It adds all agents to the model, and dynamically adds priorities to agent connections.
        """
        logging.info("Constructing the networkx model...")
        self.model = Model()
        # Add agents to the model, temporarily add the grid agent and curtailment if algorithm is priorities
        for agent in self.agents:
            self.model.add_asset(agent.asset, self._get_targets(agent))
        # Remove the grid agent and curtailment if algorithm is priorities
        self.agents.pop(0)
        if algorithm_type == "Priorities":
            self.agents.pop(0)

    def _get_targets(self, agent: Agent) -> list[tuple[Agent, int]]:
        """
        Get the targets for a Supply/Flexible agent, Demand/Flexible/Grid agents are targets.
        A list of Tuple(Asset, Priority) with:
            - Demand Agents of priority 3
            - Flexible Agents of priority 2
            - Grid Agent of priority -1
            - Curtailment Agent (only for Solar and Wind) of priority 0
        """

        def can_supply(a):
            return isinstance(
                a, (SolarAgent, WindAgent, BatteryAgent, ElectrolyserAgent, GridAgent)
            )

        def is_target(supplier, target):
            """
            - CurtailmentAgent is a valid target only for SolarAgent and WindAgent.
            """
            if isinstance(target, CurtailmentAgent):
                return isinstance(supplier, (SolarAgent, WindAgent))
            return isinstance(
                target,
                (
                    DemandAgent,
                    BatteryAgent,
                    ElectrolyserAgent,
                    GridAgent,
                ),
            )

        return [
            (
                other_agent.asset,
                self._get_priority(type(agent), type(other_agent)),
            )
            for other_agent in self.agents
            if other_agent != agent
            and is_target(agent, other_agent)
            and can_supply(agent)
        ]

    @staticmethod
    def _get_priority(agent_type: type, target_agent_type: type) -> float:
        """
        Get the right priority map for the right algorithm.
        """
        if agent_type == GridAgent:
            return -1
        elif (agent_type in [ElectrolyserAgent, BatteryAgent]) and (
            target_agent_type in [ElectrolyserAgent, BatteryAgent]
        ):
            return -2
        else:
            priority_map = {
                DemandAgent: 3,
                BatteryAgent: 2,
                ElectrolyserAgent: 2,
                CurtailmentAgent: 0,
                GridAgent: -1,
            }
            return priority_map[target_agent_type]

    def _combine_states(self, include_forecast: bool = True) -> dict:
        """Function to combine all agents states into one observation"""
        for agent in self.agents:
            full_state = agent.get_state(self.time_stamp)
            # time is also in state, and is_fcr for battery is not relevant for now.
            relevant_state = {
                key: value
                for key, value in full_state.items()
                if key not in ["time", "is_fcr"]
                and (include_forecast or key != "forecast")
            }
            self._state_cache[agent.id] = relevant_state
        return self._state_cache

    def _get_global_context(self, observations: dict, normalize: bool = True) -> dict:
        """Function to get additional global information (market prices, etc.)"""
        net_forecasted_power = get_net_forecasted_power(observations)
        if not normalize:
            current_month = self.time_stamp.month
            current_day = self.time_stamp.day
            current_hour = self.time_stamp.hour
            current_minute = self.time_stamp.minute
            return {
                "global_context": {
                    "net_forecasted_power": net_forecasted_power,
                    "month": current_month,
                    "day": current_day,
                    "hour": current_hour,
                    "minute": current_minute,
                }
            }
        else:
            # Use cyclical normalization for time
            return {
                "global_context": {"net_forecasted_power": net_forecasted_power}
                | encode_datetime(self.time_stamp)
            }

    def _get_full_observation(self, normalize: bool = True) -> dict:
        """Function to get the full observation (agent state + additional information)"""
        # TODO: Is the grid information added to the state?
        combined_states = self._combine_states()
        return combined_states | self._get_global_context(combined_states, normalize)

    def _calculate_reward(self):
        """Function to calculate the reward"""
        reward = self.reward.get(self.extractor, self.time_stamp)
        return reward

    def reset(self) -> tuple[dict, dict]:
        """
        Function should return the initial state.
        """
        for agent in self.agents:
            agent.asset.load_default_state(self.start_time)
        self.time_stamp = self.start_time
        return self._get_full_observation(), {agent.id: {} for agent in self.agents}

    def step(
        self, actions: dict, loss_dict: dict | None = None, normalize: bool = True
    ):
        """Function should return the observation, reward, done, info"""
        # Set setpoints for agents based on actions
        for agent_id, action in actions.items():
            self.agent_dict[agent_id].act(action, self.time_stamp)

        # Update the time_stamp and i
        self.time_stamp += timedelta(seconds=self.time_step)

        # Solve and step the model
        self.model.solve(self.time_step)
        self.model.step(self.time_step)

        # Update the electrolysers that have a slight loss of energy
        electrolysers = (
            agent for agent in self.agents if isinstance(agent, ElectrolyserAgent)
        )
        for electrolyser in electrolysers:
            electrolyser._loss_discharge()

        # Get full observation
        observations = self._get_full_observation(normalize)

        # Extract results and calculate next states
        self.extractor.from_model(self.model)
        self.extractor.from_observations(observations)
        if loss_dict is not None:
            self.extractor.store_loss(loss_dict)

        # All relevant information must be added to the extractor before this is called
        reward = self._calculate_reward()
        self.extractor.store_reward(reward)

        # Update the extractor
        self.extractor.update()

        # After the update, the ExtensiveExtractor needs the model again to save additional information
        self.extractor.add_additional_info_from_model(self.model)

        if normalize:
            observations = self._normalize_observations(observations)

        return observations, reward, self._dones_cache, self._infos_cache

    def _normalize_observations(self, observations: dict):
        """
        Normalize the agent observations to a range of 0 to max_power.
        """
        for agent_id, agent_dict in observations.items():
            if agent_id.startswith("global"):
                continue  # Skip global observations
            max_power = self.agent_dict[agent_id].asset.max_power_out
            if max_power == 0:
                # Demand agent; take max power in instead
                max_power = self.agent_dict[agent_id].asset.max_power_in

            for key, value in agent_dict.items():
                if key in ["degradation", "state_of_charge"]:
                    continue  # No need to normalize these values
                agent_dict[key] = value / max_power

        return observations
