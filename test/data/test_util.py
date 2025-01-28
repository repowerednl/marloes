import unittest
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from datetime import datetime

import pytest

from marloes.data.util import (
    _contains_leap_day,
    convert_kw_to_kwh,
    convert_kwh_to_minutely_kw,
    convert_to_utc,
    drop_out_series,
    read_series,
    shift_series,
)


class TestShiftSeries(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2020-02-28", "2020-03-01", freq="D", tz="UTC")
        values = [10, 20, 30]
        self.series = pd.Series(values, index=dates)
        self.start_date = datetime(2021, 1, 1, tzinfo=dates.tzinfo)
        self.end_date = datetime(2021, 12, 31, tzinfo=dates.tzinfo)

    def test_shift_series_simple_case(self):
        # Shift dates from 2020 to 2021
        shifted = shift_series(self.series, self.start_date, self.end_date)
        expected_dates = pd.DatetimeIndex(["2021-02-28", "2021-03-01"], tz="UTC")
        self.assertEqual(list(shifted.index), list(expected_dates))

    def test_shift_series_leap_to_non_leap(self):
        # Shift from a leap year to a non-leap year
        leap_dates = pd.date_range("2020-02-28", "2020-02-29", freq="D", tz="UTC")
        values = [10, 20]
        leap_series = pd.Series(values, index=leap_dates)

        shifted = shift_series(leap_series, self.start_date, self.end_date)
        expected_dates = pd.DatetimeIndex(["2021-02-28"], tz="UTC")
        self.assertEqual(list(shifted.index), list(expected_dates))

    def test_shift_series_non_leap_to_leap(self):
        # Shift from a non-leap year to a leap year
        non_leap_dates = pd.date_range("2021-02-28", "2021-03-01", freq="D", tz="UTC")
        values = [10, 20]
        non_leap_series = pd.Series(values, index=non_leap_dates)

        shifted = shift_series(
            non_leap_series,
            datetime(2020, 2, 28, tzinfo=ZoneInfo("UTC")),
            datetime(2020, 3, 1, tzinfo=ZoneInfo("UTC")),
        )
        expected_dates = pd.DatetimeIndex(
            ["2020-02-28", "2020-02-29", "2020-03-01"], tz="UTC"
        )
        pd.testing.assert_index_equal(shifted.index, expected_dates)

    def test_contains_leap_day(self):
        self.assertTrue(
            _contains_leap_day(
                datetime(2020, 1, 1, tzinfo=ZoneInfo("UTC")),
                datetime(2020, 12, 31, tzinfo=ZoneInfo("UTC")),
            )
        )
        self.assertFalse(
            _contains_leap_day(
                datetime(2021, 1, 1, tzinfo=ZoneInfo("UTC")),
                datetime(2021, 12, 31, tzinfo=ZoneInfo("UTC")),
            )
        )


class TestReadSeries(unittest.TestCase):
    def setUp(self):
        self.filepath = "Solar_EW.parquet"

    @pytest.mark.slow
    def test_read_series(self):
        series = read_series(self.filepath)

        # Number of minutes in a yaer
        self.assertEqual(len(series.index), 525600)

        # Check if the first index is 2025-01-01 00:00
        self.assertEqual(
            series.index[0], pd.Timestamp("2025-01-01 00:00", tz=ZoneInfo("UTC"))
        )

        # Check if the last index is 2025-12-31 23:59
        self.assertEqual(
            series.index[-1], pd.Timestamp("2025-12-31 23:59", tz=ZoneInfo("UTC"))
        )

    def test_read_unsupported_filetype(self):
        with self.assertRaises(AttributeError):
            read_series(self.filepath, filetype="txt")

    def test_convert_kwh_to_minutely_kw(self):
        # Make a simple DataFrame with minutely kWh data
        df = pd.DataFrame(
            {
                "value": [1.0, 2.0, 3.0, 4.0, 5.0],
            },
            index=pd.date_range(
                "2025-01-01", periods=5, freq="min", tz=ZoneInfo("UTC")
            ),
        )

        # Manually convert the minutely kW data to kWh
        time_step_seconds = 60
        df_kwh = df * time_step_seconds / 3600

        # Use the function to convert back to minutely kW
        df_converted = convert_kwh_to_minutely_kw(df_kwh)

        pd.testing.assert_frame_equal(
            df_converted, df, check_freq=False, check_index_type=False
        )

    def test_convert_kw_to_kwh(self):
        # Make a simple DataFrame with minutely kW data
        df = pd.DataFrame(
            {
                "value": [1.0, 2.0, 3.0, 4.0, 5.0],
            },
            index=pd.date_range(
                "2025-01-01", periods=5, freq="min", tz=ZoneInfo("UTC")
            ),
        )

        # Convert the minutely kW to kWh using the function
        df_converted = convert_kw_to_kwh(df)

        # Manually calculate the expected kWh for each interval
        expected_kwh = df * (1 / 60)

        pd.testing.assert_frame_equal(df_converted, expected_kwh)


class TestDropOutSeries(unittest.TestCase):
    def setUp(self):
        self.date_range = pd.date_range(start="2023-01-01", periods=1440, freq="T")
        self.values = np.random.rand(len(self.date_range))
        self.series = pd.Series(self.values, index=self.date_range)

    def test_no_dropouts(self):
        """Test that no dropouts occur when drop_prob and long_drop_prob are 0."""
        modified_series = drop_out_series(
            self.series, drop_prob=0, long_drop_prob=0, max_long_drop_days=5
        )
        pd.testing.assert_series_equal(modified_series, self.series)

    def test_single_dropouts(self):
        """Test that single dropouts occur with the specified probability."""
        drop_prob = 0.1
        modified_series = drop_out_series(
            self.series,
            drop_prob=drop_prob,
            long_drop_prob=0,
            max_long_drop_days=5,
        )

        # Count the number of dropped rows (values set to 0)
        num_dropped = (modified_series == 0).sum()
        expected_dropped = int(len(self.series) * drop_prob)

        # Allow for some variance due to randomness
        self.assertLess(
            abs(num_dropped - expected_dropped),
            len(self.series) * 0.02,  # 2% tolerance
            msg=f"Expected approximately {expected_dropped} dropouts, got {num_dropped}",
        )

    def test_long_dropouts(self):
        """Test that long dropouts occur with the specified probability and last for the correct duration."""
        long_drop_prob = 0.02
        max_long_drop_days = 2
        modified_series = drop_out_series(
            self.series,
            drop_prob=0,
            long_drop_prob=long_drop_prob,
            max_long_drop_days=max_long_drop_days,
        )

        # Convert days to minutes
        max_long_drop_minutes = max_long_drop_days * 1440

        # Find segments of consecutive 0s
        is_dropped = modified_series == 0
        dropped_streaks = np.diff(np.where(~is_dropped)[0]) - 1

        # Check if any dropout streak exceeds the maximum length
        if len(dropped_streaks) > 0:
            self.assertTrue(
                (dropped_streaks <= max_long_drop_minutes).all(),
                "Found a dropout longer than max_long_drop_days",
            )
