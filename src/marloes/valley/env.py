"""
Environment that holds all necessary information for the Simulation, called EnergyValley
"""
from datetime import datetime
from zoneinfo import ZoneInfo
from marloes.agents.battery import BatteryAgent
from marloes.agents.electrolyser import ElectrolyserAgent
from marloes.agents.demand import DemandAgent
from marloes.agents.solar import SolarAgent
from marloes.agents.wind import WindAgent


class EnergyValley:
    def __init__(self, config: dict):
        self.algorithm = config["algorithm"]
        self.agents = []
        [self.add_agent(agent_config) for agent_config in config["agents"]]
        # handle other config parameters

    def add_agent(self, config: dict):
        # Start time is fixed at 2025-01-01
        start_time = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
        match config["type"]:
            case "battery":
                agent = BatteryAgent(config, start_time)
            case "electrolyser":
                agent = ElectrolyserAgent(config, start_time)
            case "demand":
                agent = DemandAgent(config, start_time)
            case "solar":
                agent = SolarAgent(config, start_time)
            case "wind":
                agent = WindAgent(config, start_time)
        self.agents.append(agent)
        # add to self.agents with necessary priorities, edges or constraints

    def _combine_states(self):
        """Function to combine all agents states into one observation"""
        pass

    def _calculate_reward(self):
        """Function to calculate the reward"""
        pass

    def step(self, actions: list):
        """Function should return the observation, reward, done, info"""
        # step the agents
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
