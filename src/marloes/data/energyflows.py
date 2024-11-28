from collections import defaultdict
from typing import Optional, Tuple
import gc
from simon.simulation import SimulationResults
from simon.data.asset_data import AssetSetpoint, AssetState
from simon.util.encoders import jsonable_encoder
import pandas as pd
from marloes.agents.base import Agent
from pydantic import BaseModel


class EnergyFlows(SimulationResults):
    """
    Saving the energy flows of the simulation. Can be adapted from Simon's SimulationResults class (simon.simulation)
    TODO: if we adapt AssetState to our own AgentState, use this instead of Simon's.
    """

    nothing_to_see_here = True
