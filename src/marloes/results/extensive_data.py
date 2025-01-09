from collections import defaultdict
import gc
import pandas as pd
from simon.solver import Model
from simon.util.encoders import jsonable_encoder


class ExtensiveDataStore:
    """
    A lightweight, chunk-friendly container for states, setpoints, and flows.
    """

    def __init__(self, store_setpoints: bool = False):
        self.store_setpoints = store_setpoints

        # Internal storage (per chunk)
        self._states = defaultdict(list)  # Format: {Asset -> [AssetState, ...]}
        self._setpoints = defaultdict(list)  # Format: {Asset -> [AssetSetpoint, ...]}
        self._flows = defaultdict(list)  # Format: {(Asset, Asset) -> [flow, ...]}

        # Completed chunks as DataFrames
        self._df_chunks = []

    def add_step_data(self, model: Model):
        """
        Add data for one simulation step from a networkx model.
        """
        for asset in model.graph.nodes:
            self._states[asset].append(asset.get_state())
            if self.store_setpoints:
                self._setpoints[asset].append(asset.setpoint)

        for (asset1, asset2), flow_value in model.edge_flow_tracker.items():
            self._flows[(asset1, asset2)].append(flow_value)

    def stash_chunk(self):
        """
        Build a DataFrame from the in-memory states, setpoints, and flows,
        and store it in _df_chunks. Then clear the in-memory data, for efficiency:
        as the pydantic BaseModel dataclasses are not memory-efficient and pandas DataFrames are.
        """
        df = self._build_df()
        self._df_chunks.append(df)

        # Clear data
        self._states.clear()
        self._setpoints.clear()
        self._flows.clear()

        # Clear memory explicitly (TODO: check if this is necessary)
        gc.collect()

    def to_pandas(self, stash_remainder: bool = True) -> pd.DataFrame:
        """
        Return a single DataFrame with all stashed chunks concatenated.
        If stash_remainder is True, also stash any leftover data in memory.
        """
        if stash_remainder and (self._states or self._setpoints or self._flows):
            self.stash_chunk()

        return pd.concat(self._df_chunks, axis=0)

    def _build_df(self) -> pd.DataFrame:
        """
        Build a single DataFrame from the stored states, setpoints, and flows.
        The index is inferred from the 'time' attribute in the states, which is
        assumed to be uniform across all assets for each timestep.
        """
        frames = []

        # Use the time index from the first asset
        first_state = next(iter(self._states.values()))
        last_state = next(iter(reversed(self._states.values())))
        time_index = pd.date_range(
            start=first_state[0].time, end=last_state[-1].time, freq="min", tz="UTC"
        )

        # States
        for asset, states_list in self._states.items():
            df_s = pd.DataFrame(jsonable_encoder(states_list), index=time_index)
            df_s.drop(
                columns=["time"], inplace=True
            )  # 'time' is redundant as it's now the index
            df_s.columns = [f"{asset.name}_state_{col}" for col in df_s.columns]
            frames.append(df_s)

        # Setpoints
        if self.store_setpoints:
            for asset, sps_list in self._setpoints.items():
                df_sp = pd.DataFrame(jsonable_encoder(sps_list), index=time_index)
                df_sp.drop(columns=["time"], inplace=True, errors="ignore")
                df_sp.columns = [
                    f"{asset.name}_setpoint_{col}" for col in df_sp.columns
                ]
                frames.append(df_sp)

        # Flows
        for (asset1, asset2), flow_list in self._flows.items():
            df_f = pd.DataFrame(
                flow_list,
                index=time_index,
                columns=[f"{asset1.name}_to_{asset2.name}"],
            )
            frames.append(df_f)

        # Combine all frames into a single DataFrame
        return pd.concat(frames, axis=1)
