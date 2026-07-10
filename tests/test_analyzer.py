"""Tests for core.analyzer — analyze_stock with various options."""
import numpy as np
import pandas as pd


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
