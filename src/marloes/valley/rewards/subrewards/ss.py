import numpy as np

from marloes.results.extractor import Extractor
from marloes.valley.rewards.subrewards.base import SubReward

BUFFER = 2000


class SSSubReward(SubReward):
    """
    Sub-reward for penalizing lack of self-sufficiency.
    """

    name = "SS"

    def calculate(
        self,
        extractor: Extractor,
        actual: bool,
        input_dict: dict[str, float] | None = None,
        **kwargs
    ) -> float | np.ndarray:
        if actual:
            total_demand = max(
                input_dict["total_demand"] + input_dict["total_battery_intake"], 1e-6
            )
            grid_kw = input_dict["grid_state"]

            import_frac = max(grid_kw, 0) / total_demand  # 0..1
            export_frac = max(-grid_kw, 0) / total_demand  # 0..1

            # Only penalize taking too much from the grid if net_grid_state is positive,
            # otherwise we are self-sufficient, so no penalty, but to not zero reward gradient (with flat reward)
            # every extra kw returns less of a reward
            if input_dict["net_grid_state"] > 0:
                return -import_frac
            else:
                # Diminishing returns
                surplus_kwh = abs(input_dict["net_grid_state"])
                bonus_scale = 1 / (1 + surplus_kwh / BUFFER)
                return export_frac * bonus_scale

        return -np.maximum(0, np.cumsum(extractor.grid_state))
