"""
Environment that holds all necessary information for the Simulation, called EnergyValley
"""

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from simon.solver import Model

from marloes.agents.base import Agent
from marloes.agents.battery import BatteryAgent
from marloes.agents.curtailment import CurtailmentAgent
from marloes.agents.demand import DemandAgent
from marloes.agents.electrolyser import ElectrolyserAgent
from marloes.agents.grid import GridAgent
from marloes.agents.solar import SolarAgent
from marloes.agents.wind import WindAgent
from marloes.results.extractor import ExtensiveExtractor, Extractor


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
        self.start_time = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
        self.time_stamp = self.start_time
        self.time_step = 60  # 1 minute in seconds

        self.agents: list[Agent] = []
        self.grid: GridAgent = None
        self.model: Model = None
        self.extractor: Extractor = self.EXTRACTOR_MAP[
            config.pop("extractor_type", "default")
        ]()

        self._initialize_agents(config)
        self._initialize_model(
            algorithm_type
        )  # Model has a graph (nx.DiGraph) with assets as nodes and edges as connections

    def _initialize_agents(self, config: dict) -> None:
        """
        Function to initialize all agents with the given configuration.
        Requires config with "agents" key (list of dicts), and "grid" key (dict).
        """
        logging.info("Adding agents to the environment...")
        for agent_config in config.get("agents", []):
            self._add_agent(agent_config)
        # Add the grid agent
        self.grid = GridAgent(config=config.get("grid", {}), start_time=self.start_time)

    def _add_agent(self, agent_config: dict) -> None:
        """
        Adds an agent based on its type from the configuration.
        """
        agent_type = agent_config.pop("type", None)
        if agent_type not in self.AGENT_TYPE_MAP:
            raise ValueError(f"Unknown agent type: '{agent_type}'")

        agent_class = self.AGENT_TYPE_MAP[agent_type]
        self.agents.append(agent_class(agent_config, self.start_time))

    def _initialize_model(self, algorithm_type: str) -> None:
        """
        Function to initialize the Model imported from Simon.
        It adds all agents to the model, and dynamically adds priorities to agent connections.
        """
        logging.info("Constructing the networkx model...")
        self.model = Model()
        # Add agents to the model, temporarily add the grid agent and curtailment if algorithm is priorities
        self.agents.append(self.grid)
        if algorithm_type == "Priorities":
            self.agents.append(CurtailmentAgent({}, self.start_time))
        for agent in self.agents:
            self.model.add_asset(agent.asset, self._get_targets(agent, algorithm_type))
        # Remove the grid agent and curtailment if algorithm is priorities
        self.agents.pop()
        if algorithm_type == "Priorities":
            self.agents.pop()

    def _get_targets(
        self, agent: Agent, algorithm_type: str
    ) -> list[tuple[Agent, int]]:
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
                self._get_priority(type(agent), type(other_agent), algorithm_type),
            )
            for other_agent in self.agents
            if other_agent != agent
            and is_target(agent, other_agent)
            and can_supply(agent)
        ]

    @staticmethod
    def _get_priority(
        agent_type: type, target_agent_type: type, algorithm_type: str
    ) -> float:
        """
        Get the right priority map for the right algorithm.
        """
        if algorithm_type == "Priorities":
            priority_map = {
                DemandAgent: 3,
                BatteryAgent: 2,
                ElectrolyserAgent: 2,
                CurtailmentAgent: 0,
                GridAgent: -1,
            }
            return priority_map[target_agent_type]
        elif agent_type == GridAgent:
            return 10
        else:
            priority_map = {
                DemandAgent: 0,
                BatteryAgent: 0,
                ElectrolyserAgent: 0,
                GridAgent: 10,
            }
            return priority_map[target_agent_type]

    def _combine_states(self):
        """Function to combine all agents states into one observation"""
        return {agent.id: agent.asset.state.model_dump() for agent in self.agents}

    def _calculate_reward(self):
        """Function to calculate the reward"""
        pass

    def reset(self) -> tuple[dict, dict]:
        """
        Function should return the initial state.
        """
        for agent in self.agents:
            agent.asset.load_default_state(self.start_time)
        self.time_stamp = self.start_time
        return self._combine_states(), {agent.id: {} for agent in self.agents}

    def step(self, actions: dict):
        """Function should return the observation, reward, done, info"""

        # Set setpoints for agents based on actions
        for agent in self.agents:
            if agent.id in actions:
                action = actions[agent.id]
                agent.act(action, self.time_stamp)

        # Update the time_stamp
        self.time_stamp += timedelta(seconds=self.time_step)

        # Solve and step the model
        self.model.solve(self.time_step)
        self.model.step(self.time_step)

        # Extract results and calculate next states
        self.extractor.from_model(self.model)
        observations = self._combine_states()
        rewards = self._calculate_reward()

        # Done and info
        # is the hub ever done? Is life ever done? Is life a simulation?
        dones = {agent.id: False for agent in self.agents}
        dones["__all__"] = False

        # Empty info dicts for agents
        infos = {agent.id: {} for agent in self.agents}

        return observations, rewards, dones, infos
