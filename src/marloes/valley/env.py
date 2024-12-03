"""
Environment that holds all necessary information for the Simulation, called EnergyValley
"""
from datetime import datetime
from zoneinfo import ZoneInfo
from simon.solver import Model
from marloes.agents.battery import BatteryAgent
from marloes.agents.electrolyser import ElectrolyserAgent
from marloes.agents.demand import DemandAgent
from marloes.agents.solar import SolarAgent
from marloes.agents.wind import WindAgent
from marloes.agents.grid import GridAgent


class EnergyValley:
    def __init__(self, config: dict):
        self.start_time = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
        self._initialize_agents(config)
        # TODO: handle other config parameters, include in testing
        self._initialize_model()  # Model has a graph (nx.DiGraph) with assets as nodes and edges as connections

    def add_agent(self, agent_config: dict):
        # Start time is fixed at 2025-01-01
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
        Returns the targets of the agent (all agents except itself) with a priority
        A list of Tuple(Asset, Priority) with:
            - Demand Agents of priority 30
            - Flexible Agents of priority 20
            - Supply Agents of priority 10
            - Grid Agent of priority 1
        """
        priority_map = {
            DemandAgent: 30,
            BatteryAgent: 20,
            ElectrolyserAgent: 20,
            SolarAgent: 10,
            WindAgent: 10,
            GridAgent: 1,
        }
        return [
            (other_agent.asset, priority_map[type(other_agent)])
            for other_agent in self.agents
            if other_agent != agent
        ]

    def _initialize_agents(self, config: dict):
        """
        Function to initialize all agents with the given configuration.
        Requires config with "agents" key (list of dicts), and "grid" key (dict).
        """
        self.agents = []
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
        # add agents to the model
        for agent in self.agents:
            self.model.add_asset(agent.asset, self._get_targets(agent))

    def _combine_states(self):
        """Function to combine all agents states into one observation"""
        pass

    def _calculate_reward(self):
        """Function to calculate the reward"""
        pass

    def reset(self):
        """
        Function should return the initial observation.
        This environment is continuous, no start/end or terminal state.
        """
        return self._combine_states()

    def step(self, actions: list):
        """Function should return the observation, reward, done, info"""
        # solve the model
        self.model.solve()
        # extract the relevant results from the model
        self._extract_results()
        # step the model
        self.model.step()

        # model.step() steps every asset in self.graph.nodes. We step every agent manually instead:
        [
            agent.act(action) for agent, action in zip(self.agents, actions)
        ]  # function is called act now, can be renamed

        # gather observations
        # either combine every agents state into one observation or a list of observations for each agent
        observation = self._combine_states()
        # calculate reward (CO2 emission factor, sum of taking/feeding in, current net balance)
        reward = self._calculate_reward()
        # is the hub ever done? Is life ever done? Is life a simulation?
        done = False
        # info for debugging purposes
        info = {}
        return observation, reward, done, info

    def _extract_results(self):
        """Function to extract the relevant results from the model"""
        pass
