"""
Environment that holds all necessary information for the Simulation, called EnergyValley
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo
import numpy as np
import torch

from simon.solver import Model

from marloes.handlers import (
    Handler,
    BatteryHandler,
    CurtailmentHandler,
    DemandHandler,
    ElectrolyserHandler,
    GridHandler,
    SolarHandler,
    WindHandler,
)
from marloes.algorithms.util import get_net_forecasted_power
from marloes.data.util import encode_datetime
from marloes.results.extractor import ExtensiveExtractor, Extractor
from marloes.data.replaybuffer import ReplayBuffer
from marloes.valley.rewards.reward import Reward


class EnergyValley:
    """
    Environment that holds all necessary information for the simulation.
    """

    AGENT_TYPE_MAP = {
        "battery": BatteryHandler,
        "electrolyser": ElectrolyserHandler,
        "demand": DemandHandler,
        "solar": SolarHandler,
        "wind": WindHandler,
    }
    EXTRACTOR_MAP = {
        "default": Extractor,
        "extensive": ExtensiveExtractor,
    }

    def __init__(self, config: dict, algorithm_type: str):
        """
        Initializes the environment, handlers, and solver model.
        """
        super().__init__()
        default_start_time = datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC"))
        self.start_time = config.get("simulation_start_time", default_start_time)
        self.time_stamp = self.start_time
        self.time_step = 60  # 1 minute in seconds

        self.handlers: list[Handler] = []
        self.trainable_handlers: list[Handler] = []
        self.grid: Optional[GridHandler] = None
        self.model: Optional[Model] = None
        self.extractor: Extractor = self.EXTRACTOR_MAP[
            config.pop("extractor_type", "default")
        ]()
        self.reward = Reward(config, actual=True, **config.get("subrewards", {}))

        self._initialize_handlers(config, algorithm_type, config.get("data_config", {}))
        self._initialize_model(
            algorithm_type
        )  # Model has a graph (nx.DiGraph) with assets as nodes and edges as connections

        # For efficiency
        self.handler_dict = {handler.id: handler for handler in self.handlers}
        self.trainable_handler_dict = {
            handler.id: handler for handler in self.trainable_handlers
        }
        self._state_cache = {handler.id: None for handler in self.handlers}
        # is the hub ever done? Is life ever done? Is life a simulation?
        self._dones_cache = {handler.id: False for handler in self.handlers}
        self._infos_cache = {handler.id: {} for handler in self.handlers}

        # Add dims to the environment
        self.state_dim = ReplayBuffer.dict_to_tens(self._get_full_observation()).shape
        self.action_dim = torch.Size([len(self.trainable_handlers)])
        self.global_dim = ReplayBuffer.dict_to_tens(
            self._get_global_context(self._combine_states())
        ).shape
        self.handlers_scalar_dim = [
            len(state) for state in self._combine_states(False).values()
        ]
        self.forecasts = [handler.forecast is not None for handler in self.handlers]

    def _initialize_handlers(
        self, config: dict, algorithm_type: str, data_config: dict
    ) -> None:
        """
        Function to initialize all handlers with the given configuration.
        Requires config with "handlers" key (list of dicts), and "grid" key (dict).
        """
        logging.info("Adding handlers to the environment...")

        # Add the grid handler and curtailment handler to handlers
        self.grid = GridHandler(
            config=config.get("grid", {}), start_time=self.start_time
        )
        self.handlers.append(self.grid)
        if algorithm_type == "Priorities":
            self.handlers.append(CurtailmentHandler({}, self.start_time))

        for handler_config in config.get("handlers", []):
            self._add_handler(handler_config, data_config)

    def _add_handler(self, handler_config: dict, data_config: dict) -> None:
        """
        Adds an handler based on its type from the configuration.
        """
        handler_type = handler_config.pop("type", None)
        if handler_type not in self.AGENT_TYPE_MAP:
            raise ValueError(f"Unknown handler type: '{handler_type}'")

        handler_class = self.AGENT_TYPE_MAP[handler_type]
        trainable = handler_config.pop("trainable", True)
        self.handlers.append(
            handler_class(handler_config, self.start_time, data_config=data_config)
        )
        if trainable:
            self.trainable_handlers.append(self.handlers[-1])

    def _initialize_model(self, algorithm_type: str) -> None:
        """
        Function to initialize the Model imported from Simon.
        It adds all handlers to the model, and dynamically adds priorities to handler connections.
        """
        logging.info("Constructing the networkx model...")
        self.model = Model()
        # Add handlers to the model, temporarily add the grid handler and curtailment if algorithm is priorities
        for handler in self.handlers:
            self.model.add_asset(handler.asset, self._get_targets(handler))
        # Remove the grid handler and curtailment if algorithm is priorities
        self.handlers.pop(0)
        if algorithm_type == "Priorities":
            self.handlers.pop(0)

    def _get_targets(self, handler: Handler) -> list[tuple[Handler, int]]:
        """
        Get the targets for a Supply/Flexible handler, Demand/Flexible/Grid handlers are targets.
        A list of Tuple(Asset, Priority) with:
            - Demand Handlers of priority 3
            - Flexible Handlers of priority 2
            - Grid Handler of priority -1
            - Curtailment Handler (only for Solar and Wind) of priority 0
        """

        def can_supply(a):
            return isinstance(
                a,
                (
                    SolarHandler,
                    WindHandler,
                    BatteryHandler,
                    ElectrolyserHandler,
                    GridHandler,
                ),
            )

        def is_target(supplier, target):
            """
            - CurtailmentHandler is a valid target only for SolarHandler and WindHandler.
            """
            if isinstance(target, CurtailmentHandler):
                return isinstance(supplier, (SolarHandler, WindHandler))
            return isinstance(
                target,
                (
                    DemandHandler,
                    BatteryHandler,
                    ElectrolyserHandler,
                    GridHandler,
                ),
            )

        return [
            (
                other_handler.asset,
                self._get_priority(type(handler), type(other_handler)),
            )
            for other_handler in self.handlers
            if other_handler != handler
            and is_target(handler, other_handler)
            and can_supply(handler)
        ]

    @staticmethod
    def _get_priority(handler_type: type, target_handler_type: type) -> float:
        """
        Get the right priority map for the right algorithm.
        """
        if handler_type == GridHandler:
            return -1
        elif (handler_type in [ElectrolyserHandler, BatteryHandler]) and (
            target_handler_type in [ElectrolyserHandler, BatteryHandler]
        ):
            return -2
        else:
            priority_map = {
                DemandHandler: 3,
                BatteryHandler: 2,
                ElectrolyserHandler: 2,
                GridHandler: -1,
                CurtailmentHandler: -2,
            }
            return priority_map[target_handler_type]

    def _combine_states(self, include_forecast: bool = True) -> dict:
        """Function to combine all handlers states into one observation"""
        for handler in self.handlers:
            full_state = handler.get_state(self.time_stamp)
            # time is also in state, and is_fcr for battery is not relevant for now.
            relevant_state = {
                key: value
                for key, value in full_state.items()
                if key not in ["time", "is_fcr"]
                and (include_forecast or key != "forecast")
            }
            self._state_cache[handler.id] = relevant_state
        return self._state_cache

    def _get_global_context(self, observations: dict, normalize: bool = True) -> dict:
        """Function to get additional global information (market prices, etc.)"""
        net_forecasted_power = get_net_forecasted_power(observations)
        if not normalize:
            current_month = self.time_stamp.month
            current_day = self.time_stamp.day
            current_hour = self.time_stamp.hour
            current_minute = self.time_stamp.minute
            return {
                "global_context": {
                    "net_forecasted_power": net_forecasted_power,
                    "month": current_month,
                    "day": current_day,
                    "hour": current_hour,
                    "minute": current_minute,
                }
            }
        else:
            # Use cyclical normalization for time
            return {
                "global_context": {"net_forecasted_power": net_forecasted_power}
                | encode_datetime(self.time_stamp)
            }

    def _get_full_observation(self, normalize: bool = True) -> dict:
        """Function to get the full observation (handler state + additional information)"""
        # TODO: Is the grid information added to the state?
        combined_states = self._combine_states()
        return combined_states | self._get_global_context(combined_states, normalize)

    def _calculate_reward(self):
        """Function to calculate the reward"""
        reward = self.reward.get(self.extractor, self.time_stamp)
        return reward

    def reset(self) -> tuple[dict, dict]:
        """
        Function should return the initial state.
        """
        for handler in self.handlers:
            handler.asset.load_default_state(self.start_time)
        self.time_stamp = self.start_time
        return self._get_full_observation(), {
            handler.id: {} for handler in self.handlers
        }

    def step(
        self, actions: dict, loss_dict: dict | None = None, normalize: bool = True
    ):
        """Function should return the observation, reward, done, info"""
        # Set setpoints for handlers based on actions
        for handler_id, action in actions.items():
            self.handler_dict[handler_id].act(action, self.time_stamp)

        # Update the time_stamp and i
        self.time_stamp += timedelta(seconds=self.time_step)

        # Solve and step the model
        self.model.solve(self.time_step)
        self.model.step(self.time_step)

        # Update the electrolysers that have a slight loss of energy
        electrolysers = (
            handler
            for handler in self.handlers
            if isinstance(handler, ElectrolyserHandler)
        )
        for electrolyser in electrolysers:
            electrolyser._loss_discharge()

        # Get full observation
        observations = self._get_full_observation(normalize)

        # Extract results and calculate next states
        self.extractor.from_model(self.model)
        self.extractor.from_observations(observations, self.trainable_handlers)
        if loss_dict is not None:
            self.extractor.store_loss(loss_dict)

        # All relevant information must be added to the extractor before this is called
        reward = self._calculate_reward()
        self.extractor.store_reward(reward)

        # Update the extractor
        self.extractor.update()

        # After the update, the ExtensiveExtractor needs the model again to save additional information
        self.extractor.add_additional_info_from_model(self.model)

        if normalize:
            observations = self._normalize_observations(observations)

        return observations, reward, self._dones_cache, self._infos_cache

    def _normalize_observations(self, observations: dict):
        """
        Normalize the handler observations to a range of 0 to max_power.
        """
        for handler_id, handler_dict in observations.items():
            if handler_id.startswith("global"):
                continue  # Skip global observations
            max_power = self.handler_dict[handler_id].asset.max_power_out
            if max_power == 0:
                # Demand handler; take max power in instead
                max_power = self.handler_dict[handler_id].asset.max_power_in

            for key, value in handler_dict.items():
                if key in ["degradation", "state_of_charge"]:
                    continue  # No need to normalize these values
                handler_dict[key] = value / max_power

        return observations
