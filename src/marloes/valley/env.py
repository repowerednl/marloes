"""
Environment that holds all necessary information for the Simulation, called EnergyValley
"""
from datetime import datetime
from zoneinfo import ZoneInfo
from simon.solver import Model
from marloes.agents.base import Agent
from marloes.agents.battery import BatteryAgent
from marloes.agents.electrolyser import ElectrolyserAgent
from marloes.agents.demand import DemandAgent
from marloes.agents.solar import SolarAgent
from marloes.agents.wind import WindAgent
from marloes.agents.grid import GridAgent
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class EnergyValley(MultiAgentEnv):
    def __init__(self, config: dict):
        super().__init__()
        self.start_time = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
        self.time_step = 60  # 1 minute in seconds
        self._initialize_agents(config)
        self._initialize_model()  # Model has a graph (nx.DiGraph) with assets as nodes and edges as connections

        # TODO: handle other config parameters, include in testing

    def add_agent(self, agent_config: dict):
        match agent_config.pop("type"):
            case "battery":
                agent = BatteryAgent(agent_config, self.start_time)
            case "electrolyser":
                agent = ElectrolyserAgent(agent_config, self.start_time)
            case "demand":
                agent = DemandAgent(agent_config, self.start_time)
            case "solar":
                agent = SolarAgent(agent_config, self.start_time)
            case "wind":
                agent = WindAgent(agent_config, self.start_time)
        self.agents.append(agent)

    def _get_targets(self, agent):
        """
        Get the targets for a Supply/Flexible agent, Demand/Flexible/Grid agents are targets
        A list of Tuple(Asset, Priority) with:
            - Demand Agents of priority 3
            - Flexible Agents of priority 2
            - Grid Agent of priority 1
        """

        def _can_supply(a):
            return isinstance(
                a, (SolarAgent, WindAgent, BatteryAgent, ElectrolyserAgent, GridAgent)
            )

        def _is_target(a):
            return isinstance(
                a, (DemandAgent, BatteryAgent, ElectrolyserAgent, GridAgent)
            )

        priority_map = {
            DemandAgent: 3,
            BatteryAgent: 2,
            ElectrolyserAgent: 2,
            GridAgent: 1,
        }
        return [
            (other_agent.asset, priority_map[type(other_agent)])
            for other_agent in self.agents
            if other_agent != agent and _is_target(other_agent) and _can_supply(agent)
        ]

    def _initialize_agents(self, config: dict):
        """
        Function to initialize all agents with the given configuration.
        Requires config with "agents" key (list of dicts), and "grid" key (dict).
        """
        self.agents: list[Agent] = []
        for agent_config in config["agents"]:
            self.add_agent(agent_config)
        # add the grid agent
        self.agents.append(
            GridAgent(config=config.get("grid", {}), start_time=self.start_time)
        )

    def _initialize_model(self):
        """
        Function to initialize the Model imported from Simon.
        It adds all agents to the model, and dynamically adds priorities to agent connections.
        """
        self.model = Model()

        # Add agents to the model
        for agent in self.agents:
            self.model.add_asset(agent.asset, self._get_targets(agent))

    def _combine_states(self):
        """Function to combine all agents states into one observation"""
        pass

    def _calculate_reward(self):
        """Function to calculate the reward"""
        pass

    def _extract_results(self):
        """Function to extract the relevant results from the model"""
        pass

    def reset(self):
        """
        Function should return the initial state.
        """
        for agent in self.agents:
            agent.asset.load_default_state(self.start_time)
        return self._combine_states()

    def step(self, actions: dict):
        """Function should return the observation, reward, done, info"""

        # Set setpoints for agents based on actions
        for agent in self.agents:
            if agent.id in actions:
                action = actions[agent.id]
                agent.act(action)

        # Solve and step the model
        self.model.solve(self.time_step)
        self.model.step(self.time_step)

        # Extract results and calculate next states
        self._extract_results()
        observations = self._combine_states()
        rewards = self._calculate_reward()

        # Done and info
        # is the hub ever done? Is life ever done? Is life a simulation?
        dones = {agent.id: False for agent in self.agents}
        dones["__all__"] = False

        # Empty info dicts for agents
        infos = {agent.id: {} for agent in self.agents}

        return observations, rewards, dones, infos
