"""
Unit tests for common/utils.py

Tests critical functions for correctness, especially:
- coerce_quarter_cols: Quarter date parsing (lookahead bias prevention)
- sum_ttm: Trailing twelve month calculation
- apply_lag: Reporting lag application
"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime

from utils import (
    coerce_quarter_cols,
    sum_ttm,
    apply_lag,
    pick_row_from_sheet,
    normalize_ticker,
)


class TestCoerceQuarterCols:
    """Tests for coerce_quarter_cols - CRITICAL for preventing lookahead bias"""

    def test_basic_quarter_parsing(self):
        """Test that quarters are parsed correctly"""
        row = pd.Series({
            "2023/3": 100,
            "2023/6": 200,
            "2023/9": 300,
            "2023/12": 400,
        })
        result = coerce_quarter_cols(row)

        assert len(result) == 4
        assert result.iloc[0] == 100
        assert result.iloc[3] == 400

    def test_quarter_end_dates_not_day_one(self):
        """
        CRITICAL TEST: Quarter dates must be month-end, not day=1.
        This prevents 30-day lookahead bias in lag calculations.
        """
        row = pd.Series({
            "2023/3": 100,   # Should be 2023-03-31, NOT 2023-03-01
            "2023/6": 200,   # Should be 2023-06-30
            "2023/9": 300,   # Should be 2023-09-30
            "2023/12": 400,  # Should be 2023-12-31
        })
        result = coerce_quarter_cols(row)

        # Check that dates are month-end
        assert result.index[0].day == 31  # March has 31 days
        assert result.index[1].day == 30  # June has 30 days
        assert result.index[2].day == 30  # September has 30 days
        assert result.index[3].day == 31  # December has 31 days

    def test_handles_turkish_decimal_format(self):
        """Test that Turkish decimal format (comma) is handled"""
        row = pd.Series({
            "2023/3": "1.234,56",  # Turkish format
            "2023/6": "2 345,67",  # With space separator
        })
        result = coerce_quarter_cols(row)

        # Should parse correctly (note: current impl replaces , with .)
        assert len(result) == 2
        assert pd.notna(result.iloc[0])

    def test_invalid_months_ignored(self):
        """Test that non-quarter months are ignored"""
        row = pd.Series({
            "2023/1": 100,   # January - not a quarter end
            "2023/3": 200,   # March - valid
            "2023/5": 300,   # May - not a quarter end
        })
        result = coerce_quarter_cols(row)

        assert len(result) == 1
        assert result.iloc[0] == 200

    def test_invalid_years_ignored(self):
        """Test that years outside valid range are ignored"""
        row = pd.Series({
            "1990/3": 100,   # Too old
            "2023/3": 200,   # Valid
            "2040/3": 300,   # Too far future
        })
        result = coerce_quarter_cols(row)

        assert len(result) == 1
        assert result.iloc[0] == 200

    def test_empty_result_for_invalid_data(self):
        """Test that invalid data returns empty Series"""
        row = pd.Series({
            "not_a_date": 100,
            "invalid": 200,
        })
        result = coerce_quarter_cols(row)

        assert len(result) == 0


class TestSumTTM:
    """Tests for sum_ttm - Trailing Twelve Month calculation"""

    def test_basic_ttm_calculation(self):
        """Test basic 4-quarter TTM sum"""
        # Q1=100, Q2=100, Q3=100, Q4=100 → TTM=400
        dates = pd.date_range("2023-03-31", periods=4, freq="QE")
        series = pd.Series([100, 100, 100, 100], index=dates)

        result = sum_ttm(series)

        # Should have TTM starting from 4th quarter
        assert len(result) >= 1
        assert result.iloc[-1] == 400

    def test_ttm_with_missing_quarter(self):
        """Test TTM scaling when one quarter is missing"""
        dates = pd.date_range("2023-03-31", periods=4, freq="QE")
        # Q1=100, Q2=NaN, Q3=100, Q4=100 → should scale
        values = [100, np.nan, 100, 100]
        series = pd.Series(values, index=dates)

        result = sum_ttm(series)

        # With 3 quarters summing to 300, scaled by 4/3 = 400
        # The last valid TTM should be approximately 400
        last_ttm = result.iloc[-1]
        assert 380 <= last_ttm <= 420  # Allow some tolerance

    def test_annual_data_detection(self):
        """Test that annual data (>120 day gaps) is returned as-is"""
        # Annual data with 365-day gaps
        dates = pd.to_datetime(["2021-12-31", "2022-12-31", "2023-12-31"])
        series = pd.Series([1000, 1100, 1200], index=dates)

        result = sum_ttm(series)

        # Should return original series (already annual)
        assert len(result) == 3
        assert result.iloc[-1] == 1200

    def test_empty_series(self):
        """Test that empty series returns empty"""
        series = pd.Series(dtype=float)
        result = sum_ttm(series)
        assert len(result) == 0


class TestApplyLag:
    """Tests for apply_lag - CRITICAL for preventing lookahead bias"""

    def test_q4_longer_lag(self):
        """Test that Q4 (December) has longer lag than other quarters"""
        # Q4 2023 data
        dates = pd.date_range("2023-12-31", periods=1, freq="D")
        series = pd.Series([1000], index=dates)

        target_dates = pd.date_range("2024-01-01", "2024-03-31", freq="D")
        result = apply_lag(series, target_dates, q4_lag_days=70, other_lag_days=40)

        # Q4 with 70-day lag: Dec 31 + 70 = March 11
        # Data should be NaN before March 11
        march_1 = pd.Timestamp("2024-03-01")
        march_15 = pd.Timestamp("2024-03-15")

        assert pd.isna(result.loc[march_1])  # Before lag expires
        assert pd.notna(result.loc[march_15])  # After lag expires

    def test_q1q2q3_shorter_lag(self):
        """Test that Q1-Q3 have shorter lag"""
        # Q1 2024 data (March)
        dates = pd.to_datetime(["2024-03-31"])
        series = pd.Series([1000], index=dates)

        target_dates = pd.date_range("2024-04-01", "2024-06-30", freq="D")
        result = apply_lag(series, target_dates, q4_lag_days=70, other_lag_days=40)

        # Q1 with 40-day lag: March 31 + 40 = May 10
        may_1 = pd.Timestamp("2024-05-01")
        may_15 = pd.Timestamp("2024-05-15")

        assert pd.isna(result.loc[may_1])  # Before lag expires
        assert pd.notna(result.loc[may_15])  # After lag expires

    def test_forward_fill_behavior(self):
        """Test that values are forward-filled after lag expires"""
        dates = pd.to_datetime(["2024-03-31"])
        series = pd.Series([1000], index=dates)

        target_dates = pd.date_range("2024-05-15", "2024-06-30", freq="D")
        result = apply_lag(series, target_dates, q4_lag_days=70, other_lag_days=40)

        # All dates after lag should have the same value
        assert (result == 1000).all()

    def test_no_lookahead_bias(self):
        """
        CRITICAL TEST: Ensure data from quarter Q is NOT available before
        Q_end + lag_days
        """
        # Simulate Q1 2024 ending March 31
        q1_end = pd.Timestamp("2024-03-31")
        series = pd.Series([1000], index=[q1_end])

        # Create target dates that span before and after when data should be available
        target_dates = pd.date_range("2024-03-01", "2024-05-31", freq="D")
        result = apply_lag(series, target_dates, other_lag_days=40)

        # Data available date = March 31 + 40 = May 10
        data_available_date = pd.Timestamp("2024-05-10")

        # ALL dates before May 10 should be NaN (no lookahead)
        before_available = result.loc[result.index < data_available_date]
        assert before_available.isna().all(), "LOOKAHEAD BIAS: Data available before lag!"

        # Dates on/after May 10 should have data
        on_or_after = result.loc[result.index >= data_available_date]
        assert on_or_after.notna().all(), "Data should be available after lag"


class TestNormalizeTicker:
    """Tests for ticker normalization"""

    def test_removes_suffix(self):
        assert normalize_ticker("AKBNK.IS") == "AKBNK"
        assert normalize_ticker("THYAO.E") == "THYAO"

    def test_uppercase(self):
        assert normalize_ticker("akbnk") == "AKBNK"
        assert normalize_ticker("Thyao") == "THYAO"

    def test_already_normalized(self):
        assert normalize_ticker("AKBNK") == "AKBNK"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
