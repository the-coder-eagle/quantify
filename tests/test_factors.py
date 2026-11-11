"""Tests for north-bound capital data interfaces, financial indicators, fund flow, industry info, universe filtering, and factor computation engine."""
import pandas as pd
import numpy as np
import pytest
from core.data import fetch_north_bound_individual, fetch_north_bound_history
from core.universe import filter_universe


@pytest.fixture(autouse=True)
def offline_factor_sources(request, monkeypatch):
    """Factor unit tests must not depend on live APIs or local data caches."""
    if request.node.get_closest_marker("live"):
        return
    from core import data
    import news
    samples = {
        "fetch_financial_indicators": {"roe": 20, "eps": 2, "data_date": "2026-01-01"},
        "fetch_north_bound_individual": {"hold_pct": 3},
        "fetch_margin_data": {"margin_balance": 1000, "margin_buy": 100},
        "fetch_fund_flow": {"main_net_inflow": 100, "super_large_net_inflow": 50},
    }
    for name, sample in samples.items():
        monkeypatch.setattr(data, name, lambda code, sample=sample:
                            sample.copy() if code == "600519" else None)
    monkeypatch.setattr(news, "fetch", lambda *args, **kwargs: [])


@pytest.mark.live
def test_fetch_limit_up_pool_returns_list():
    from core.data import fetch_limit_up_pool
    result = fetch_limit_up_pool()
    if result is not None:
        assert isinstance(result, list)


@pytest.mark.live
def test_fetch_north_bound_individual_returns_dict():
    result = fetch_north_bound_individual("600519")
    if result is not None:  # API may fail, that's OK
        assert isinstance(result, dict)
        assert "hold_pct" in result


@pytest.mark.live
def test_fetch_north_bound_history_returns_dataframe():
    df = fetch_north_bound_history()
    if df is not None and not df.empty:
        assert isinstance(df, pd.DataFrame)
        assert "date" in df.columns or df.index.name == "date"


@pytest.mark.live
def test_fetch_margin_data_returns_dict():
    from core.data import fetch_margin_data
    result = fetch_margin_data("600519")
    if result is not None:  # API may fail, that's OK
        assert isinstance(result, dict)
        assert "margin_balance" in result
        assert "margin_buy" in result
        assert "created_at" in result


@pytest.mark.live
def test_fetch_financial_indicators_returns_dict():
    from core.data import fetch_financial_indicators
    result = fetch_financial_indicators("600519")
    if result is not None:  # API may fail, that's OK
        assert isinstance(result, dict)
        assert "roe" in result


@pytest.mark.live
def test_fetch_fund_flow_returns_dict():
    from core.data import fetch_fund_flow
    result = fetch_fund_flow("600519")
    if result is not None:  # API may fail, that's OK
        assert isinstance(result, dict)


@pytest.mark.live
def test_fetch_industry_info_returns_dict():
    from core.data import fetch_industry_info
    result = fetch_industry_info("600519")
    if result is not None:  # API may fail, that's OK
        assert isinstance(result, dict)
        assert "industry_name" in result


# ---------------------------------------------------------------------------
# Universe / stock pool filter tests
# ---------------------------------------------------------------------------

def test_filter_universe_excludes_st():
    df = pd.DataFrame({
        "code": ["000001", "600000", "000002"],
        "name": ["平安银行", "ST测试", "万科A"],
        "price": [10.0, 5.0, 8.0],
        "turnover": [5e8, 1e8, 3e8],
        "total_mv": [2e11, 5e9, 1e11],
    })
    result = filter_universe(df)
    assert "600000" not in result["code"].values
    assert len(result) == 2


def test_filter_universe_price_range():
    df = pd.DataFrame({
        "code": ["000001", "000002", "000003"],
        "name": ["A", "B", "C"],
        "price": [1.0, 10.0, 20.0],
        "turnover": [5e8, 5e8, 5e8],
        "total_mv": [2e11, 2e11, 2e11],
    })
    result = filter_universe(df, min_price=2.0, max_price=12.0)
    assert "000001" not in result["code"].values  # too cheap
    assert "000003" not in result["code"].values  # too expensive
    assert "000002" in result["code"].values


# ============================================================================
# Factor computation tests (Task 2.1)
# ============================================================================

# --- Test helpers ---

def _make_ohlcv(n=200, seed=42):
    """Generate random OHLCV data for factor testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    price = 10.0 + np.cumsum(rng.normal(0, 0.2, n))
    price = np.maximum(price, 1.0)
    return pd.DataFrame({
        "Open":  price * (1 + rng.uniform(-0.01, 0.01, n)),
        "High":  price * (1 + abs(rng.uniform(0, 0.02, n))),
        "Low":   price * (1 - abs(rng.uniform(0, 0.02, n))),
        "Close": price,
        "Volume": rng.integers(1_000_000, 100_000_000, n),
    }, index=dates)


def _make_snapshot_row():
    """Return a dict mimicking a single row from market_snapshot()."""
    return {
        "code": "000001",
        "name": "平安银行",
        "price": 12.5,
        "pe_ttm": 8.5,
        "pb": 1.2,
        "total_mv": 2.5e11,
        "circ_mv": 2.0e11,
        "turnover": 5.0e8,
        "pct_chg": 1.5,
    }


# --- _safe_float helper ---

class TestSafeFloat:
    def test_valid_float(self):
        from core.factors import _safe_float
        assert _safe_float(3.14) == 3.14
        assert _safe_float("2.5") == 2.5
        assert _safe_float(0) == 0.0

    def test_none_returns_none(self):
        from core.factors import _safe_float
        assert _safe_float(None) is None

    def test_nan_returns_none(self):
        from core.factors import _safe_float
        assert _safe_float(float("nan")) is None

    def test_invalid_string_returns_none(self):
        from core.factors import _safe_float
        assert _safe_float("abc") is None


# --- Momentum factors ---

class TestMomentumFactors:
    def test_produces_all_expected_keys(self):
        from core.factors import compute_momentum_factors
        df = _make_ohlcv(300)
        result = compute_momentum_factors(df["Close"])
        assert "mom_1m" in result
        assert "mom_3m" in result
        assert "mom_6m" in result
        assert "mom_12m" in result
        assert "mom_1m_ex5d" in result

    def test_short_series_returns_empty(self):
        from core.factors import compute_momentum_factors
        df = _make_ohlcv(10)
        result = compute_momentum_factors(df["Close"])
        assert len(result) == 0

    def test_momentum_values_are_reasonable(self):
        from core.factors import compute_momentum_factors
        df = _make_ohlcv(300)
        result = compute_momentum_factors(df["Close"])
        # Momentum should be within [-0.9, 10.0] for random walk
        for key in result:
            assert -0.9 < result[key] < 10.0, f"{key}={result[key]} out of range"

    def test_flat_price_gives_zero_momentum(self):
        from core.factors import compute_momentum_factors
        flat = pd.Series([100.0] * 300, index=pd.date_range("2025-01-01", periods=300, freq="B"))
        result = compute_momentum_factors(flat)
        for key in result:
            assert abs(result[key]) < 0.001, f"{key}={result[key]} should be near zero"


# --- Volatility factors ---

class TestVolatilityFactors:
    def test_produces_vol_values(self):
        from core.factors import compute_volatility_factors
        df = _make_ohlcv(200)
        result = compute_volatility_factors(df["Close"])
        assert "vol_20d" in result
        assert "vol_60d" in result
        assert "vol_ratio" in result

    def test_volatility_is_positive(self):
        from core.factors import compute_volatility_factors
        df = _make_ohlcv(200)
        result = compute_volatility_factors(df["Close"])
        assert result["vol_20d"] > 0
        assert result["vol_60d"] > 0

    def test_short_series_60d_only(self):
        from core.factors import compute_volatility_factors
        df = _make_ohlcv(30)
        result = compute_volatility_factors(df["Close"])
        assert "vol_20d" in result
        assert "vol_60d" not in result  # not enough data

    def test_very_short_series_returns_empty(self):
        from core.factors import compute_volatility_factors
        df = _make_ohlcv(15)
        result = compute_volatility_factors(df["Close"])
        assert len(result) == 0


# --- Technical factors ---

class TestTechnicalFactors:
    def test_produces_expected_keys(self):
        from core.factors import compute_technical_factors
        df = _make_ohlcv(200)
        result = compute_technical_factors(
            df["Close"], df["High"], df["Low"], df["Volume"]
        )
        # Should have at least some of these
        key_set = set(result.keys())
        expected = {"rsi_deviation", "macd_hist", "bb_position", "ma20_slope", "volume_ratio"}
        assert len(key_set & expected) >= 2, f"Got keys: {key_set}"

    def test_short_series_returns_empty(self):
        from core.factors import compute_technical_factors
        df = _make_ohlcv(15)
        result = compute_technical_factors(
            df["Close"], df["High"], df["Low"], df["Volume"]
        )
        assert len(result) == 0

    def test_rsi_deviation_in_range(self):
        from core.factors import compute_technical_factors
        df = _make_ohlcv(200)
        result = compute_technical_factors(
            df["Close"], df["High"], df["Low"], df["Volume"]
        )
        if "rsi_deviation" in result:
            assert -1.0 <= result["rsi_deviation"] <= 1.0

    def test_macd_above_signal_is_binary(self):
        from core.factors import compute_technical_factors
        df = _make_ohlcv(200)
        result = compute_technical_factors(
            df["Close"], df["High"], df["Low"], df["Volume"]
        )
        if "macd_above_signal" in result:
            assert result["macd_above_signal"] in (1.0, -1.0)

    def test_volume_ratio_reasonable(self):
        from core.factors import compute_technical_factors
        df = _make_ohlcv(200)
        result = compute_technical_factors(
            df["Close"], df["High"], df["Low"], df["Volume"]
        )
        if "volume_ratio" in result:
            assert result["volume_ratio"] > 0


# --- Value factors ---

class TestValueFactors:
    def test_produces_ep_bp_from_valid_snapshot(self):
        from core.factors import compute_value_factors
        snap = _make_snapshot_row()
        result = compute_value_factors("000001", snap)
        assert "ep_ratio" in result
        assert "bp_ratio" in result
        assert result["ep_ratio"] > 0
        assert result["bp_ratio"] > 0

    def test_negative_pe_gives_zero_ep(self):
        from core.factors import compute_value_factors
        snap = {"pe_ttm": -5.0, "pb": 1.5, "circ_mv": 1e11, "turnover": 5e8}
        result = compute_value_factors("000001", snap)
        assert result["ep_ratio"] == 0.0

    def test_missing_pe_returns_no_ep(self):
        from core.factors import compute_value_factors
        snap = {"pb": 1.5, "circ_mv": 1e11, "turnover": 5e8}
        result = compute_value_factors("000001", snap)
        assert "ep_ratio" not in result

    def test_missing_pb_returns_no_bp(self):
        from core.factors import compute_value_factors
        snap = {"pe_ttm": 10.0, "circ_mv": 1e11, "turnover": 5e8}
        result = compute_value_factors("000001", snap)
        assert "bp_ratio" not in result


# --- Size factors ---

class TestSizeFactors:
    def test_produces_log_market_cap(self):
        from core.factors import compute_size_factors
        snap = _make_snapshot_row()
        result = compute_size_factors(snap)
        assert "ln_circ_mv" in result
        assert "ln_total_mv" in result
        assert result["ln_circ_mv"] > 0

    def test_missing_circ_mv_skips(self):
        from core.factors import compute_size_factors
        snap = {"total_mv": 1e11}
        result = compute_size_factors(snap)
        assert "ln_circ_mv" not in result
        assert "ln_total_mv" in result

    def test_zero_market_cap_skips(self):
        from core.factors import compute_size_factors
        snap = {"circ_mv": 0, "total_mv": 0}
        result = compute_size_factors(snap)
        assert "ln_circ_mv" not in result
        assert "ln_total_mv" not in result


# --- API-dependent factor groups (graceful failure) ---

class TestQualityFactors:
    def test_returns_dict(self):
        from core.factors import compute_quality_factors
        result = compute_quality_factors("600519")
        assert result["roe"] == 0.2
        assert result["eps"] == 2

    def test_handles_invalid_code(self):
        from core.factors import compute_quality_factors
        result = compute_quality_factors("000000")
        assert isinstance(result, dict)  # shouldn't crash


class TestNorthBoundFactors:
    def test_returns_dict(self):
        from core.factors import compute_north_bound_factors
        result = compute_north_bound_factors("600519")
        assert result == {"north_hold_pct": 3.0}

    def test_invalid_code_no_crash(self):
        from core.factors import compute_north_bound_factors
        result = compute_north_bound_factors("999999")
        assert isinstance(result, dict)


class TestMarginFactors:
    def test_returns_dict(self):
        from core.factors import compute_margin_factors
        result = compute_margin_factors("600519")
        assert result == {"margin_balance": 1000.0, "margin_buy": 100.0}


class TestFundFlowFactors:
    def test_returns_dict(self):
        from core.factors import compute_fund_flow_factors
        result = compute_fund_flow_factors("600519")
        assert result == {"main_net_inflow": 100.0, "super_large_net_inflow": 50.0}


class TestSectorMomentumFactors:
    def test_returns_dict(self):
        from core.factors import compute_sector_momentum_factors
        result = compute_sector_momentum_factors("600519")
        assert isinstance(result, dict)


class TestSentimentFactors:
    def test_returns_dict(self):
        from core.factors import compute_sentiment_factors
        result = compute_sentiment_factors("600519")
        assert isinstance(result, dict)


# --- Integration: compute_all_factors ---

class TestComputeAllFactors:
    def test_returns_dict_with_factors(self):
        from core.factors import compute_all_factors
        df = _make_ohlcv(300)
        snap = _make_snapshot_row()
        result = compute_all_factors(df, snap)
        assert isinstance(result, dict)
        assert len(result) >= 5  # should have at least momentum + vol + tech + value + size factors

    def test_no_none_values_in_result(self):
        from core.factors import compute_all_factors
        df = _make_ohlcv(300)
        snap = _make_snapshot_row()
        result = compute_all_factors(df, snap)
        for key, val in result.items():
            assert val is not None, f"Factor '{key}' is None"
            if isinstance(val, float):
                assert not np.isnan(val), f"Factor '{key}' is NaN"

    def test_short_ohlcv_is_handled(self):
        from core.factors import compute_all_factors
        df = _make_ohlcv(10)
        snap = _make_snapshot_row()
        result = compute_all_factors(df, snap)
        assert isinstance(result, dict)  # should not crash, may have fewer factors


# --- FRESHNESS constant ---

class TestFreshness:
    def test_all_groups_have_freshness(self):
        from core.factors import FRESHNESS
        expected_groups = [
            "value", "momentum", "quality", "volatility", "size",
            "north_bound", "sector_momentum", "technical", "sentiment",
            "margin", "fund_flow",
        ]
        for group in expected_groups:
            assert group in FRESHNESS, f"Missing freshness for {group}"
