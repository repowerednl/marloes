from datetime import datetime

import numpy as np
from simon.assets.demand import Demand


class CurtailmentAgent:
    """
    CurtailmentAgent is an agent that allows the Solar Parks and Wind Farms to be curtailed in case no setpoints are used.
    """

    def __init__(self, config: dict, start_time: datetime):
        self.id = config.get("name", "Curtailment")
        self.asset = Demand(
            name="Curtailment",
            max_power_in=np.inf,
            constant_demand=np.inf,
            curtailable_by_solver=True,
        )
        self.asset.load_default_state(start_time)

    def get_state(self):
        """
        Returns the current state of the 'Connection' asset from Simon.
        This is an AssetState with 'time' and 'power'.
        """
        return self.asset.state
