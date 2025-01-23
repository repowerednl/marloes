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
