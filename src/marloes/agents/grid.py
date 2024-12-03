from datetime import datetime

from simon.assets.grid import Connection


class GridAgent:
    """
    GridAgent is an agent that represents the Grid.
    config would require:
    - name: name of the agent
    - max_power_in: maximum power that can be drawn from the grid
    - max_power_out: maximum power that can be fed back to the grid
    """

    def __init__(self, config: dict, start_time: datetime):
        self.asset = Connection(config, start_time)

    def get_state(self):
        """
        Returns the current state of the 'Connection' asset from Simon
        This is an AssetState with 'time' and 'power'
        """
        return self.asset.state
