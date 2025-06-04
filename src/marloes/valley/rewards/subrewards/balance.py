import numpy as np
from marloes.valley.rewards.subrewards.base import SubReward

MAX_DEMAND = 30


class BalanceSubReward(SubReward):
    """
    Simple sub‐reward: negative normalized absolute mismatch
    between total supply and demand.
    """

    name = "Balance"

    def __init__(self, config: dict, active: bool = False, scaling_factor: float = 1.0):
        super().__init__(config, active, scaling_factor)
        self.total_battery_power = sum(
            agent.get("power")
            for agent in self.config["agents"]
            if agent.get("type") == "battery"
        )
        self.max_demand = sum(
            agent.get("scale", 1) * MAX_DEMAND
            for agent in self.config["agents"]
            if agent.get("type") == "demand"
        )
        self.max_supply = sum(
            agent.get("AC")
            for agent in self.config["agents"]
            if agent.get("type") in ["solar", "wind"]
        )

    def calculate(self, extractor, actual: bool, **kwargs) -> float | np.ndarray:
        solar = extractor.total_solar_production[extractor.i]
        wind = extractor.total_wind_production[extractor.i]
        supply = solar + wind
        available_supply = extractor.supply_available_power[extractor.i]
        demand = extractor.total_demand[extractor.i]

        battery_production = extractor.total_battery_production[extractor.i]
        battery_intake = extractor.total_battery_intake[extractor.i]
        battery_actual = battery_production - battery_intake

        battery_prev_actual = (
            extractor.total_battery_production[extractor.i - 1]
            - extractor.total_battery_intake[extractor.i - 1]
        )

        grid_production = extractor.total_grid_production[extractor.i]
        grid_state = extractor.grid_state[extractor.i]

        solar_action = sum(
            extractor.__dict__[attr][extractor.i]
            for attr in extractor.__dict__
            if isinstance(getattr(extractor, attr), np.ndarray) and "SolarAgent" in attr
        )
        wind_action = sum(
            extractor.__dict__[attr][extractor.i]
            for attr in extractor.__dict__
            if isinstance(getattr(extractor, attr), np.ndarray) and "WindAgent" in attr
        )
        supply_action = solar_action + wind_action

        # Get all attributes from extractor that contain "BatteryAgent"
        battery_charge_action = 0
        battery_discharge_action = 0

        for attr in extractor.__dict__:
            if (
                isinstance(getattr(extractor, attr), np.ndarray)
                and "BatteryAgent" in attr
            ):
                action = getattr(extractor, attr)[extractor.i]
                # Discharge is positive, charge is negative
                if action > 0:
                    battery_discharge_action += action
                else:
                    battery_charge_action += action

        surplus = available_supply - demand
        # Encourage charging when surplus, discourage discharging during surplus
        # base_reward = -(grid_production / self.max_demand
        # TODO: fix penalty for multiple batteries
        # charge_setpoint_penalty = (
        #     -abs(battery_charge_action + surplus) / self.total_battery_power
        # )
        # discharge_setpoint_penalty = (
        #     -abs(battery_discharge_action + surplus) / self.total_battery_power
        # )
        # battery_setpoint_penalty = charge_setpoint_penalty + discharge_setpoint_penalty

        # positive surplus → want charge; negative surplus → want discharge
        battery_action = battery_discharge_action + battery_charge_action
        battery_setpoint_penalty = (
            -abs(battery_action + surplus) / self.total_battery_power
        )

        same_sign = np.sign(grid_state) == np.sign(battery_action)
        battery_direction_penalty = (
            0.0 if same_sign else -abs(battery_action) / self.total_battery_power
        )

        # feasability_penalty = -(abs(battery_action - battery_actual) / self.total_battery_power) * 2

        # # Smooth actions by penalizing large changes
        # cycle_penalty = 0.05 * (
        #     -abs(battery_actual - battery_prev_actual) / self.total_battery_power
        # )
        # setpoint_penalty = -battery_action / self.total_battery_power
        # solar_penalty = -extractor.total_solar_production[extractor.i] / 3000

        grid_production_penalty = -grid_production / self.max_demand

        maximize_renewables_penalty = (
            supply_action - self.max_supply
        ) / self.max_supply
        return (
            grid_production_penalty + battery_setpoint_penalty
        )  # + maximize_renewables_penalty # setpoint_penalty # + feasability_penalty # + cycle_penalty + feasability_penalty§
