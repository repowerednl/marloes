import copy
from datetime import datetime

import numpy as np
from simon.assets.grid import Connection


def patched_get_state(self):
    return copy.deepcopy(self.state)


class GridAgent:
    """
    GridAgent is an agent that represents the Grid.
    config requires:
    - name: name of the agent
    - max_power_in: maximum power that can be drawn from the grid
    - max_power_out: maximum power that can be fed back to the grid
    """

    def __init__(self, config: dict, start_time: datetime, *args, **kwargs):
        self.id = config.get("name", "National Grid")
        self.asset = Connection(
            **self._merge_configs(self._get_default_grid_config(), config)
        )
        self.asset.get_state = patched_get_state.__get__(self.asset)
        self.asset.load_default_state(start_time)

    @classmethod
    def _get_default_grid_config(cls) -> dict:
        return {
            "name": "Grid",
            "max_power_in": np.inf,
            "max_power_out": np.inf,
        }

    @staticmethod
    def _merge_configs(default_config: dict, config: dict) -> dict:
        """Merge the default configuration with user-provided values."""
        return {**default_config, **config}
