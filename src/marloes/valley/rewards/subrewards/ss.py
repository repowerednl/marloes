import numpy as np

from marloes.results.extractor import Extractor
from marloes.valley.rewards.subrewards.base import SubReward

BUFFER = 2000
ALPHA = 0.999  # running-mean decay
MAX_DEMAND = 30
BATTERY_SCALE = 0.2


class SSSubReward(SubReward):
    """
    Sub-reward for penalizing lack of self-sufficiency.
    """

    name = "SS"

    def __init__(
        self,
        config: dict,
        active: bool = False,
        scaling_factor: float = 1.0,
    ):
        """
        Initializes the SSSubReward instance with activation and scaling properties.
        """
        super().__init__(config, active, scaling_factor)
        self.r_mean = 0
        max_demand = sum(
            agent.get("scale", 1) * MAX_DEMAND
            for agent in self.config["agents"]
            if agent.get("type") == "demand"
        )
        max_battery = sum(
            agent.get("power") * BATTERY_SCALE
            for agent in self.config["agents"]
            if agent.get("type") == "battery"
        )
        self.max_intake = max_demand + max_battery

    def calculate(
        self,
        extractor: Extractor,
        actual: bool,
        input_dict: dict[str, float] | None = None,
        **kwargs
    ) -> float | np.ndarray:
        if actual:
            grid_kw = input_dict["grid_state"]

            import_frac = max(grid_kw, 0) / self.max_intake  # 0..1
            export_frac = max(-grid_kw, 0) / self.max_intake  # 0..1

            # Only penalize taking too much from the grid if net_grid_state is positive,
            # otherwise we are self-sufficient, so no penalty, but to not zero reward gradient (with flat reward)
            # every extra kw returns less of a reward
            if input_dict["net_grid_state"] > 0:
                # Calculate the running mean
                raw_reward = -import_frac
                self.r_mean = (ALPHA * self.r_mean) + (1 - ALPHA) * raw_reward
                reward = raw_reward - self.r_mean
                return reward
            else:
                # Diminishing returns
                surplus_kwh = abs(input_dict["net_grid_state"])
                bonus_scale = 1 / (1 + surplus_kwh / BUFFER)
                return export_frac * bonus_scale

        return -np.maximum(0, np.cumsum(extractor.grid_state))
