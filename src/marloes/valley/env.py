"""
Environment that holds all necessary information for the Simulation, called EnergyValley
"""
from marloes.agents.battery import BatteryAgent
from marloes.agents.electrolyser import ElectrolyserAgent
from marloes.agents.demand import DemandAgent
from marloes.agents.solar import SolarAgent
from marloes.agents.wind import WindAgent


class EnergyValley:
    def __init__(self, config: dict):
        self.agents = []
        [self.add_agent(agent_config) for agent_config in config["agents"]]
        # TODO: handle other config parameters, include in testing

    def add_agent(self, config: dict):
        if config["type"] == "battery":
            agent = BatteryAgent(config)
        if config["type"] == "electrolyser":
            agent = ElectrolyserAgent(config)
        if config["type"] == "demand":
            agent = DemandAgent(config)
        if config["type"] == "solar":
            agent = SolarAgent(config)
        if config["type"] == "wind":
            agent = WindAgent(config)
        self.agents.append(agent)
        # add to self.agents with necessary priorities, edges or constraints

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
