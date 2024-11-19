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
        for agent_config in config['agents']:
            self.add_agent(agent_config)
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
    
    def train(self):
        # train the agents
        pass

