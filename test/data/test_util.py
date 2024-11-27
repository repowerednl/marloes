import unittest
from zoneinfo import ZoneInfo
import pandas as pd
from datetime import datetime

from marloes.data.util import _contains_leap_day, convert_to_utc, shift_series


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
