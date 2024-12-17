# tests/factories.py

from time import time
import factory
import numpy as np
from simon.assets.grid import Connection
from simon.assets.supply import Supply
from simon.assets.battery import Battery
from simon.assets.asset import Asset
from simon.assets.demand import Demand

from marloes.results.extractor import Extractor
from marloes.valley.reward import Reward, SubReward


class AssetFactory(factory.Factory):
    """
    Base factory for Asset classes.
    """

    class Meta:
        model = Asset

    name = factory.Sequence(lambda n: f"Asset {n}")

    @factory.post_generation
    def load_state(obj, create, extracted, **kwargs):
        # Always load the default state
        if create:
            obj.load_default_state()


class ConnectionFactory(AssetFactory):
    """
    Factory for Connection assets.
    """

    class Meta:
        model = Connection

    name = factory.Sequence(lambda n: f"GridAgent {n}")
    max_power_in = 1000
    max_power_out = 1000


class SolarFactory(AssetFactory):
    """
    Factory for Supply assets.
    """

    class Meta:
        model = Supply

    name = factory.Sequence(lambda n: f"SolarAgent {n}")
    constant_supply = 900
    max_power_out = 1000


class BatteryFactory(AssetFactory):
    """
    Factory for Battery assets.
    """

    class Meta:
        model = Battery

    name = factory.Sequence(lambda n: f"BatteryAgent {n}")
    max_power_in = 1000
    max_power_out = 1000
    max_state_of_charge = 0.95
    min_state_of_charge = 0.05
    energy_capacity = 1000
    ramp_up_rate = 1000
    ramp_down_rate = 1000
    efficiency = 0.85


class DemandFactory(AssetFactory):
    """
    Factory for Demand assets.
    """

    class Meta:
        model = Demand

    name = factory.Sequence(lambda n: f"DemandAgent {n}")
    max_power_in = 1000
    constant_demand = 900


class ExtractorFactory(factory.Factory):
    """
    Factory for Extractor classes.
    """

    class Meta:
        model = Extractor

    chunk_size = 1
    from_model = True

    @factory.post_generation
    def set_dynamic_attributes(self, create, extracted, **kwargs):
        """
        Overwrite dynamically created attributes after Extractor initialization.
        """
        # Default values for dynamically created attributes
        self.total_solar_production = np.array([10, 20, 30])
        self.total_wind_production = np.array([5, 15, 25])
        self.total_battery_production = np.array([0, 10, 20])
        self.total_grid_production = np.array([-5, 5, 15])
        self.grid_state = np.array([-10, 5, 10])
        self.i = 2

        # Overwrite with user-provided values if present in kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, np.array(value))


class RewardFactory(factory.Factory):
    """
    Factory for Reward classes.
    """

    class Meta:
        model = Reward

    EMMISSION_COEFFICIENTS = {
        "solar": 0.2,
        "wind": 0.1,
        "battery": 0.3,
        "grid": 0.5,
        "electrolyser": 0.4,
    }

    VALID_SUB_REWARDS = {"CO2", "SS", "NC", "NB"}

    actual = True

    CO2 = {"active": True, "scaling_factor": 1.0}
    SS = {"active": True, "scaling_factor": 1.0}
    NC = {"active": True, "scaling_factor": 1.0}
    NB = {"active": True, "scaling_factor": 1.0}
