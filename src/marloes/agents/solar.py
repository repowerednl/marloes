"""Solar agent with functionality from Repowered's Simon"""
from simon.assets.supply import Supply
from .base import Agent


class SolarAgent(Agent):
    def __init__(self, config: dict):
        model = Supply(config)

        super().__init__(model)

    def _get_profile(self):
        # TODO: should be something along these lines
        # profile = production_series * solar_park.DC / 1000  # from kWp to MWp
        # profile[profile > solar_park.AC] = solar_park.AC
        pass

    def act(self):
        pass

    def observe(self):
        pass

    def learn(self):
        pass
