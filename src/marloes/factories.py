# tests/factories.py

import factory
from simon.assets.grid import Connection
from simon.assets.supply import Supply
from simon.assets.battery import Battery
from simon.assets.asset import Asset
from simon.assets.demand import Demand


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
