"""
Environment that holds all necessary information for the Simulation, called EnergyHub (popular name, can change it later)
"""
from agents.battery import BatteryAgent
from agents.electrolyser import ElectrolyserAgent
from agents.demand import DemandAgent
from agents.solar import SolarAgent
from agents.wind import WindAgent

class EnergyHub:
    def __init__(self, config: dict):
        self.algorithm = config['algorithm']
        [self.add_agent(agent_config) for agent_config in config['agents']] 
        # handle other config parameters

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
        # add to self.agents with necessary priorities, edges or constraints
    
    def _combine_states(self):
        """ Function to combine all agents states into one observation """
        pass

    def _calculate_reward(self):
        """ Function to calculate the reward """
        pass

    def step(self):
        """ Function should return the observation, reward, done, info """
        # step the agents
        [ agent.act() for agent in self.agents ] # function is called act now, can be renamed
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

