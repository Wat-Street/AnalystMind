"""Tests for the Fundamentals ingestion pipeline.

All tests use synthetic data — no network, no DB, no yfinance.
Tests exercise the pure helpers directly: ``_compute_fcf_yield``,
``_compute_net_debt``, ``_compute_revenue_cagr_3y``, ``_extract_metrics``,
and ``_to_record``.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.ingestion.fundamentals import (
    PERIOD_LABEL,
    _compute_fcf_yield,
    _compute_net_debt,
    _compute_revenue_cagr_3y,
    _extract_metrics,
    _normalize_margin,
    _positive_or_none,
    _to_record,
    fetch_fundamentals,
)


# ── _positive_or_none ────────────────────────────────────────────────────────

class TestPositiveOrNone:

    def test_positive_value_passthrough(self):
        assert _positive_or_none(25.0) == 25.0

    def test_none_returns_none(self):
        assert _positive_or_none(None) is None

    def test_zero_returns_none(self):
        assert _positive_or_none(0) is None

    def test_negative_returns_none(self):
        assert _positive_or_none(-5.0) is None

    def test_nan_returns_none(self):
        assert _positive_or_none(float("nan")) is None

    def test_inf_returns_none(self):
        assert _positive_or_none(float("inf")) is None

    def test_string_returns_none(self):
        assert _positive_or_none("abc") is None


# ── _normalize_margin ────────────────────────────────────────────────────────

class TestNormalizeMargin:

    def test_valid_margin(self):
        assert _normalize_margin(0.43) == 0.43

    def test_zero_is_valid(self):
        assert _normalize_margin(0.0) == 0.0

    def test_one_is_valid(self):
        assert _normalize_margin(1.0) == 1.0

    def test_none_returns_none(self):
        assert _normalize_margin(None) is None

    def test_negative_returns_none(self):
        assert _normalize_margin(-0.1) is None

    def test_above_one_returns_none(self):
        assert _normalize_margin(1.5) is None

    def test_nan_returns_none(self):
        assert _normalize_margin(float("nan")) is None


# ── _compute_fcf_yield ───────────────────────────────────────────────────────

class TestComputeFcfYield:

    def test_normal_case(self):
        # $1B FCF on $10B market cap = 10% yield
        assert _compute_fcf_yield(1_000_000_000, 10_000_000_000) == pytest.approx(0.1)

    def test_missing_fcf(self):
        assert _compute_fcf_yield(None, 10e9) is None

    def test_missing_market_cap(self):
        assert _compute_fcf_yield(1e9, None) is None

    def test_both_missing(self):
        assert _compute_fcf_yield(None, None) is None

    def test_zero_market_cap(self):
        assert _compute_fcf_yield(1e9, 0) is None

    def test_negative_market_cap(self):
        assert _compute_fcf_yield(1e9, -1e9) is None

    def test_nan_fcf(self):
        assert _compute_fcf_yield(float("nan"), 1e12) is None

    def test_large_values(self):
        # AAPL-scale: $101B FCF on $4.5T market cap
        result = _compute_fcf_yield(101_000_000_000, 4_500_000_000_000)
        assert result == pytest.approx(0.02244, rel=1e-3)


# ── _compute_net_debt ────────────────────────────────────────────────────────

class TestComputeNetDebt:

    def test_normal_case(self):
        assert _compute_net_debt(50.0, 20.0) == 30.0

    def test_net_cash(self):
        """More cash than debt = negative net debt (net cash position)."""
        assert _compute_net_debt(20.0, 50.0) == -30.0

    def test_both_missing(self):
        assert _compute_net_debt(None, None) is None

    def test_only_debt_known(self):
        assert _compute_net_debt(50.0, None) == 50.0

    def test_only_cash_known(self):
        assert _compute_net_debt(None, 20.0) == -20.0

    def test_zero_both(self):
        assert _compute_net_debt(0.0, 0.0) == 0.0


# ── _compute_revenue_cagr_3y ─────────────────────────────────────────────────

class TestComputeRevenueCagr3y:

    def test_known_case_100_to_200(self):
        # $100B → $200B over 3 years => CAGR ≈ 25.99%
        series = [100.0, 130.0, 170.0, 200.0]
        result = _compute_revenue_cagr_3y(series)
        assert result == pytest.approx(0.2599, abs=1e-3)

    def test_no_growth(self):
        series = [100.0, 100.0, 100.0, 100.0]
        assert _compute_revenue_cagr_3y(series) == pytest.approx(0.0, abs=1e-9)

    def test_too_few_points(self):
        assert _compute_revenue_cagr_3y([100.0, 110.0, 120.0]) is None

    def test_empty_series(self):
        assert _compute_revenue_cagr_3y([]) is None

    def test_zero_revenue_returns_none(self):
        assert _compute_revenue_cagr_3y([0.0, 100.0, 200.0, 400.0]) is None

    def test_negative_revenue_returns_none(self):
        assert _compute_revenue_cagr_3y([100.0, 50.0, -10.0, -20.0]) is None

    def test_with_none_values_skipped(self):
        """Nones are filtered out; valid values still anchor the CAGR.
        Positions 1 and 4 span 3 years, so the CAGR is computed."""
        series = [None, 100.0, 130.0, 170.0, 200.0]
        result = _compute_revenue_cagr_3y(series)
        assert result == pytest.approx(0.2599, abs=1e-3)

    def test_too_few_years_with_none_gaps(self):
        """Two valid values at positions 1 and 3 span only 2 years → None."""
        series = [None, 100.0, None, 200.0]
        assert _compute_revenue_cagr_3y(series) is None

    def test_more_than_4_points(self):
        """With 5 points spanning 4 years, uses first→last."""
        series = [100.0, 120.0, 140.0, 170.0, 200.0]
        result = _compute_revenue_cagr_3y(series)
        # 100 → 200 over 4 years = ~18.9%
        assert result == pytest.approx(0.1892, abs=1e-3)

    def test_declining_revenue(self):
        """Negative CAGR is valid — revenue can shrink."""
        series = [200.0, 180.0, 150.0, 100.0]
        result = _compute_revenue_cagr_3y(series)
        assert result is not None
        assert result < 0  # revenue declined


# ── _extract_metrics ─────────────────────────────────────────────────────────

class TestExtractMetrics:

    def _full_info(self) -> dict:
        """AAPL-scale info dict with all fields present."""
        return {
            "trailingPE":         37.04,
            "forwardPE":          31.88,
            "enterpriseToEbitda": 28.22,
            "marketCap":          4_498_883_870_720,
            "freeCashflow":       101_090_746_368,
            "grossMargins":       0.47862,
            "totalDebt":          84_710_998_016,
            "totalCash":          68_507_000_832,
        }

    def test_all_present(self):
        info = self._full_info()
        revenue = [200e9, 210e9, 214e9, 221e9]
        out = _extract_metrics(info, revenue)

        assert out["trailing_pe"] == pytest.approx(37.04)
        assert out["forward_pe"] == pytest.approx(31.88)
        assert out["ev_ebitda"] == pytest.approx(28.22)
        assert out["fcf_yield"] == pytest.approx(101e9 / 4499e9, rel=1e-2)
        assert out["gross_margin"] == pytest.approx(0.47862)
        assert out["net_debt"] == pytest.approx(84.7e9 - 68.5e9, rel=1e-2)
        assert out["revenue_cagr_3y"] is not None and out["revenue_cagr_3y"] > 0

    def test_sparse_info(self):
        """Only marketCap + freeCashflow → fcf_yield computed; rest None."""
        info = {"marketCap": 1e12, "freeCashflow": 1e10}
        out = _extract_metrics(info, [])

        assert out["fcf_yield"] == pytest.approx(0.01)
        assert out["trailing_pe"] is None
        assert out["forward_pe"] is None
        assert out["ev_ebitda"] is None
        assert out["gross_margin"] is None
        assert out["net_debt"] is None
        assert out["revenue_cagr_3y"] is None

    def test_filters_non_positive_ratios(self):
        """PE <= 0 means no earnings → must be None, not a negative number."""
        info = {"trailingPE": -5.0, "forwardPE": 0, "enterpriseToEbitda": 12.0}
        out = _extract_metrics(info, [])
        assert out["trailing_pe"] is None
        assert out["forward_pe"] is None
        assert out["ev_ebitda"] == 12.0

    def test_output_keys_match_schema(self):
        out = _extract_metrics({}, [])
        assert set(out.keys()) == {
            "fcf_yield", "trailing_pe", "forward_pe", "ev_ebitda",
            "revenue_cagr_3y", "gross_margin", "net_debt",
        }


# ── _to_record ───────────────────────────────────────────────────────────────

class TestToRecord:

    def _make_df(self, **overrides) -> pd.DataFrame:
        defaults = {
            "fcf_yield": 0.05, "trailing_pe": 25.0, "forward_pe": 22.0,
            "ev_ebitda": 18.0, "revenue_cagr_3y": 0.12,
            "gross_margin": 0.43, "net_debt": 40e9,
        }
        defaults.update(overrides)
        df = pd.DataFrame([defaults], columns=list(defaults))
        df.index = pd.MultiIndex.from_tuples(
            [("AAPL", PERIOD_LABEL)], names=["ticker", "period"]
        )
        return df

    def test_builds_expected_dict(self):
        df = self._make_df()
        record = _to_record(df, "AAPL")
        assert record["ticker"] == "AAPL"
        assert record["period"] == PERIOD_LABEL
        assert record["fcf_yield"] == 0.05
        assert record["trailing_pe"] == 25.0
        assert record["net_debt"] == 40e9

    def test_none_values_preserved(self):
        df = self._make_df(trailing_pe=float("nan"), fcf_yield=float("nan"))
        record = _to_record(df, "MSFT")
        assert record["trailing_pe"] is None
        assert record["fcf_yield"] is None

    def test_all_numeric_values_are_float_or_none(self):
        df = self._make_df()
        record = _to_record(df, "TSLA")
        for key in _extract_metrics({}, []):
            val = record[key]
            assert val is None or isinstance(val, float), f"{key} should be float or None"


# ── fetch_fundamentals error-path tests ──────────────────────────────────────

class TestFetchFundamentalsErrors:

    def test_raises_on_empty_ticker(self):
        with pytest.raises(ValueError, match="ticker"):
            fetch_fundamentals("")

    def test_raises_on_none_ticker(self):
        with pytest.raises(ValueError, match="ticker"):
            fetch_fundamentals(None)  # type: ignore[arg-type]

    def test_raises_on_whitespace_ticker(self):
        with pytest.raises(ValueError, match="ticker"):
            fetch_fundamentals("   ")
