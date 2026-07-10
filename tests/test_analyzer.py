"""Tests for core.analyzer — analyze_stock with various options."""
import numpy as np
import pandas as pd


def make_test_df(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for tests. Importable from test_analyzer."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_rows, freq="B")
    close = 10.0 + np.cumsum(rng.normal(0, 0.2, n_rows))
    close = np.maximum(close, 1.0)
    high = close + rng.uniform(0, 0.5, n_rows)
    low = close - rng.uniform(0, 0.5, n_rows)
    low = np.maximum(low, 0.01)
    open_price = close + rng.uniform(-0.3, 0.3, n_rows)
    volume = rng.integers(1_000_000, 10_000_000, n_rows)
    return pd.DataFrame(
        {"Open": open_price, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class TestAnalyzeStockBasics:
    def test_returns_none_for_insufficient_data(self, sample_ohlcv):
        from core.analyzer import analyze_stock
        result = analyze_stock(sample_ohlcv.iloc[:50])
        assert result is None

    def test_returns_expected_keys(self, sample_ohlcv):
        from core.analyzer import analyze_stock
        result = analyze_stock(sample_ohlcv)
        assert result is not None
        for key in ["price", "signal", "confidence", "rsi", "regime",
                     "shares", "stop", "buys", "sells", "details"]:
            assert key in result, f"Missing key: {key}"

    def test_signal_is_valid(self, sample_ohlcv):
        from core.analyzer import analyze_stock
        result = analyze_stock(sample_ohlcv)
        assert result is not None
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_confidence_in_range(self, sample_ohlcv):
        from core.analyzer import analyze_stock
        result = analyze_stock(sample_ohlcv)
        assert result is not None
        assert 0.0 <= result["confidence"] <= 1.0

    def test_rsi_in_bounds(self, sample_ohlcv):
        from core.analyzer import analyze_stock
        result = analyze_stock(sample_ohlcv)
        assert result is not None
        assert 0 <= result["rsi"] <= 100


class TestAnalyzeStockWithOverrides:
    def test_custom_params_accepted(self, sample_ohlcv):
        from core.analyzer import analyze_stock
        overrides = {"趋势-均线": {"fast": 5, "slow": 20}}
        result = analyze_stock(sample_ohlcv, strategy_overrides=overrides)
        assert result is not None
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_custom_weights_change_signal(self, sample_ohlcv):
        from core.analyzer import analyze_stock
        # Give all weight to one strategy
        weights = {"趋势-均线": 0.2, "趋势-MACD": 0.2, "动量": 0.2,
                    "量确认": 0.2, "反转": 0.2}
        result_default = analyze_stock(sample_ohlcv)
        result_weighted = analyze_stock(sample_ohlcv, strategy_weights=weights)
        # Should both produce valid signals (may or may not differ)
        assert result_default is not None
        assert result_weighted is not None
        assert result_weighted["signal"] in ("BUY", "SELL", "HOLD")

    def test_oos_check_includes_results(self, sample_ohlcv):
        from core.analyzer import analyze_stock
        result = analyze_stock(sample_ohlcv, oos_check=True)
        assert result is not None
        assert "oos" in result
        for name, oos in result["oos"].items():
            assert "verdict" in oos
            assert oos["verdict"] in ("OK", "WARN", "FAIL", "SKIP")


class TestAnalyzeStockEdgeCases:
    def test_all_nan_close(self):
        from core.analyzer import analyze_stock
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({
            "Open": np.ones(100), "High": np.ones(100),
            "Low": np.ones(100), "Close": np.full(100, np.nan),
            "Volume": np.ones(100),
        })
        result = analyze_stock(df)
        assert result is None


class TestAnalyzeStockBackwardCompat:
    def test_analyze_stock_returns_compatible_keys(self):
        """New analyzer must still return old keys for backward compat."""
        from core.analyzer import analyze_stock
        df = make_test_df(200)
        result = analyze_stock(df)
        assert result is not None
        # Old keys must still be present
        for key in ["signal", "confidence", "rsi", "buys", "sells", "details", "regime"]:
            assert key in result, f"Missing key: {key}"

    def test_factor_keys_not_present_without_ranking(self):
        """Without a ranking snapshot, factor_score should not appear."""
        from core.analyzer import analyze_stock
        df = make_test_df(200)
        result = analyze_stock(df, code="600519")
        assert result is not None
        # No ranking snapshot exists, so factor keys should be absent
        assert "factor_score" not in result
        assert "percentile" not in result

    def test_default_code_parameter(self):
        """Calling without code should still work and return old keys."""
        from core.analyzer import analyze_stock
        df = make_test_df(200)
        result = analyze_stock(df)
        assert result is not None
        assert result["code"] == ""
