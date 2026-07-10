"""Tests for core.regime — detect_regime, regime_risk_multiplier."""
from core.regime import detect_regime, regime_risk_multiplier


class TestDetectRegime:
    def test_insufficient_data(self, sample_ohlcv_short):
        result = detect_regime(sample_ohlcv_short)
        assert result["regime"] == "insufficient_data"
        assert result["trend_strength"] == 0

    def test_returns_dict_with_keys(self, sample_ohlcv):
        result = detect_regime(sample_ohlcv)
        expected_keys = [
            "regime", "trend_strength", "vol_20", "vol_change",
            "rsi", "price_vs_ma20", "price_vs_ma50",
        ]
        for k in expected_keys:
            assert k in result, f"Missing key: {k}"

    def test_valid_regime_value(self, sample_ohlcv):
        result = detect_regime(sample_ohlcv)
        assert result["regime"] in [
            "trending_up", "trending_down", "high_volatility",
            "low_volatility", "ranging",
        ]


class TestRiskMultiplier:
    def test_high_vol_reduces_risk(self, sample_ohlcv):
        info = {"regime": "high_volatility", "vol_change": 2.0}
        assert regime_risk_multiplier(info) == 0.5

    def test_low_vol_increases_risk(self, sample_ohlcv):
        info = {"regime": "low_volatility", "vol_change": 0.5}
        assert regime_risk_multiplier(info) == 1.2

    def test_default_is_one(self, sample_ohlcv):
        info = {"regime": "trending_up", "vol_change": 1.0}
        assert regime_risk_multiplier(info) == 1.0

    def test_missing_keys(self):
        assert regime_risk_multiplier({}) == 1.0
