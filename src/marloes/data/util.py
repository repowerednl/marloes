from datetime import datetime
import os
from zoneinfo import ZoneInfo
import logging

import pandas as pd
import numpy as np


def read_series(
    filepath: str,
    in_kw: bool = True,
    filetype: str = "parquet",
    forecast: bool = False,
    data_config: dict = None,
) -> pd.Series:
    """
    Reads a Parquet file and returns it as a minutely kW series.
    """
    # Adjust filepath for forecast
    if forecast:
        subdir = f"forecasts/{filepath}"
    else:
        subdir = f"profiles/{filepath}"
    # Check cache first
    cached_filepath = f"src/marloes/data/cached/{subdir.replace('.parquet', '.pkl')}"
    try:
        series = pd.read_pickle(cached_filepath)
        logging.info(f"Loaded series from cache: {cached_filepath}")
    except FileNotFoundError:
        # Adjust filepath for root
        filepath = f"src/marloes/data/{subdir}"

        # Read in the file given the filetype
        read_function = getattr(pd, f"read_{filetype}")
        df = read_function(filepath)

        # Ensure there are not multiple columns in the DataFrame
        if df.shape[1] != 1:
            raise ValueError("Only one column is allowed to convert to series.")

        # Make sure the data is in kWh to convert it to minutely kW
        if in_kw:
            df = convert_kw_to_kwh(df)
        df = convert_kwh_to_minutely_kw(df)

        # Convert the DataFrame to a Series
        series = df.squeeze("columns")

        # Shift the series to the year 2025 so all data aligns
        series = shift_series(
            series,
            datetime(2025, 1, 1, tzinfo=ZoneInfo("UTC")),
            datetime(2025, 12, 31, 23, 59, tzinfo=ZoneInfo("UTC")),
        )

        # Save to cache
        dumped_filepath = (
            f"src/marloes/data/cached/{subdir.replace('.parquet', '.pkl')}"
        )
        os.makedirs(os.path.dirname(dumped_filepath), exist_ok=True)
        series.to_pickle(dumped_filepath)
        logging.info(f"Saved series to cache: {dumped_filepath}")

    series = add_noise_to_series(
        series, noise=data_config.get("noise")
    )  # Add noise to the series
    if not forecast:
        # Simulate dropouts in the series
        series = drop_out_series(
            series,
            drop_prob=data_config.get("drop_prob"),
            long_drop_prob=data_config.get("long_drop_prob"),
            max_long_drop_days=data_config.get("max_long_drop_days"),
        )

    return series


def drop_out_series(
    series: pd.Series,
    drop_prob: float = 0,  # TODO: Adjust these parameters to simulate different dropout scenarios
    long_drop_prob: float = 0,
    max_long_drop_days: int = 5,
):
    """
    Simulates random dropouts in a time series.
    drop_prob = Probability of a single row being set to 0.
    long_drop_prob = Probability of initiating a longer dropout period.
    max_long_drop_days = Maximum number of consecutive days to drop out.
    These parameters can be tuned to simulate different dropout scenarios.
    """

    ## Single dropouts
    modified_series = series.copy()
    random_values = np.random.rand(len(series))
    single_dropout_mask = random_values < drop_prob
    modified_series[single_dropout_mask] = 0

    ## Long dropout periods
    long_dropout_initiation = np.random.rand(len(series)) < long_drop_prob
    long_dropout_indices = np.where(long_dropout_initiation)[0]

    # Convert days to minutes
    max_long_drop_minutes = max_long_drop_days * 1440  # 1440 minutes in a day
    for start_idx in long_dropout_indices:
        # Determine the length of the long dropout period
        long_drop_length = np.random.randint(1, max_long_drop_minutes + 1)

        # Find end index
        end_idx = min(start_idx + long_drop_length, len(series))

        # Set values to 0 for entire period
        modified_series.iloc[start_idx:end_idx] = 0

    return modified_series


def convert_kwh_to_minutely_kw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts kWh data to minutely kW data by distributing the energy over each minute.
    """
    time_step = df.index[1] - df.index[0]

    # Calculate the average kW during each interval
    df_kw = df / (time_step.total_seconds() / 3600.0)

    # Make sure the index is complete until :59
    start = df.index[0]
    end = df.index[-1] + time_step - pd.Timedelta(minutes=1)
    full_index = pd.date_range(start=start, end=end, freq="1min")
    df_minutely_kw = df_kw.reindex(full_index, method="ffill")

    return df_minutely_kw


def convert_kw_to_kwh(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts kW data to kWh based on the time interval between data points.
    """
    # Calculate the time interval in hours
    time_step = df.index[1] - df.index[0]
    interval_hours = time_step.total_seconds() / 3600.0

    # Convert kW to kWh
    df_kwh = df * interval_hours

    return df_kwh


def shift_series(
    series: pd.Series, start_date: datetime, end_date: datetime
) -> pd.Series:
    """
    Shifts the dates of a DatetimeIndex series to fit within the specified start and end date.
    The dates are cyclically shifted such that the month, day, and time stay the same, but the year changes
    to fit within the target range. Leap years are handled automatically:
    - If the original data contains a leap day but the new target year does not, the leap day is removed.
    - If the original data does not contain a leap day, but the new target year does, February 29th inherits data from February 28th.
    """
    series = convert_to_utc(series)

    def map_to_target_year(date):
        if date >= start_date.replace(year=date.year):
            target_year = start_date.year
        else:
            target_year = start_date.year + 1

        try:
            return date.replace(year=target_year)
        except ValueError:  # Handle leap year issues for February 29th
            return None

    # Map the dates to the new target year
    new_index = series.index.map(map_to_target_year)

    # Filter out None values (which occur when shifting from a leap year to a non-leap year)
    valid_index = pd.DatetimeIndex([date for date in new_index if not pd.isnull(date)])

    # Adjust index if shifting from leap to non-leap
    if _contains_leap_day(
        series.index.min(), series.index.max()
    ) and not _contains_leap_day(start_date, end_date):
        series = series[~((series.index.month == 2) & (series.index.day == 29))]

    shifted_series = pd.Series(series.values, index=valid_index)

    # Handle the case where the target year has a leap day but the original does not
    if not _contains_leap_day(
        series.index.min(), series.index.max()
    ) and _contains_leap_day(start_date, end_date):
        # Find February 28th items
        feb_28_items = shifted_series[
            (shifted_series.index.day == 28) & (shifted_series.index.month == 2)
        ]

        # Shift the index by 1 day to make it February 29th
        feb_29_items = feb_28_items.copy()
        feb_29_items.index = feb_29_items.index + pd.DateOffset(days=1)

        # Append the February 29th data to the shifted series
        shifted_series = pd.concat([shifted_series, feb_29_items])

    return shifted_series.sort_index()


def _contains_leap_day(start_date: datetime, end_date: datetime) -> bool:
    """
    Checks if the time range between start_date and end_date includes a leap day (February 29th).
    """
    # Loop through each year in the range between start_date and end_date
    for year in range(start_date.year, end_date.year + 1):
        # Check if the current year is a leap year
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            leap_day = datetime(year, 2, 29, tzinfo=start_date.tzinfo)
            # Check if the leap day is within the provided date range
            if start_date <= leap_day <= end_date:
                return True
    return False


def convert_to_utc(series: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """
    Localize the index of a DataFrame to UTC.
    If no timezone-info is defined, it is assumed to be in Europe/Amsterdam.
    """
    if series.index.tz is None:
        series.index = (
            pd.to_datetime(series.index)
            .tz_localize("Europe/Amsterdam")
            .tz_convert("UTC")
        )
    else:
        series.index = pd.to_datetime(series.index).tz_convert("UTC")
    return series


def add_noise_to_series(series: pd.Series, noise: float) -> pd.Series:
    """
    Adds normally distributed noise (5% of the standard deviation) to a series.
    """
    dev = series.std() * noise
    return series + np.random.normal(0, dev, series.shape[0])


def convert_to_hourly_nomination(series: pd.Series) -> pd.Series:
    """
    Convert a minutely production series of a solar park into an hourly nomination for the day-ahead market.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        return pd.Series()

    hourly_nomination = series.resample("h").agg("mean")

    return hourly_nomination


def encode_datetime(dt: datetime) -> dict:
    return {
        "month_sin": np.sin(2 * np.pi * dt.month / 12),
        "month_cos": np.cos(2 * np.pi * dt.month / 12),
        "day_sin": np.sin(2 * np.pi * dt.day / 31),
        "day_cos": np.cos(2 * np.pi * dt.day / 31),
        "hour_sin": np.sin(2 * np.pi * dt.hour / 24),
        "hour_cos": np.cos(2 * np.pi * dt.hour / 24),
        "minute_sin": np.sin(2 * np.pi * dt.minute / 60),
        "minute_cos": np.cos(2 * np.pi * dt.minute / 60),
    }
