"""Tests for core.indicators — sma, ema, rsi, macd, bollinger_bands, atr."""
import numpy as np
import pandas as pd
from core.indicators import sma, ema, rsi, macd, bollinger_bands, atr


class TestSMA:
    def test_basic(self, sample_ohlcv):
        result = sma(sample_ohlcv["Close"], 20)
        assert len(result) == len(sample_ohlcv)
        # First 19 values should be NaN (min_periods=20)
        assert result.iloc[:19].isna().all()
        assert not pd.isna(result.iloc[19])

    def test_constant_series(self):
        s = pd.Series([5.0] * 50)
        result = sma(s, 10)
        assert (result.dropna() == 5.0).all()


class TestEMA:
    def test_basic(self, sample_ohlcv):
        result = ema(sample_ohlcv["Close"], 12)
        assert len(result) == len(sample_ohlcv)
        assert not pd.isna(result.iloc[-1])


class TestRSI:
    def test_bounds(self, sample_ohlcv):
        result = rsi(sample_ohlcv["Close"], 14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_all_up_gives_high_rsi(self):
        """A steadily rising series should produce RSI approaching 100."""
        s = pd.Series(np.linspace(10, 20, 200))
        result = rsi(s, 14)
        assert result.iloc[-1] > 70


class TestMACD:
    def test_histogram_identity(self, sample_ohlcv):
        m = macd(sample_ohlcv["Close"])
        hist = m["macd"] - m["signal"]
        pd.testing.assert_series_equal(m["histogram"], hist, check_names=False)

    def test_output_columns(self, sample_ohlcv):
        m = macd(sample_ohlcv["Close"])
        assert list(m.columns) == ["macd", "signal", "histogram"]


class TestBollingerBands:
    def test_upper_above_lower(self, sample_ohlcv):
        bb = bollinger_bands(sample_ohlcv["Close"], 20, 2)
        valid = bb.dropna()
        assert (valid["upper"] > valid["lower"]).all()

    def test_middle_between(self, sample_ohlcv):
        bb = bollinger_bands(sample_ohlcv["Close"], 20, 2)
        valid = bb.dropna()
        assert (valid["upper"] >= valid["middle"]).all()
        assert (valid["middle"] >= valid["lower"]).all()


class TestATR:
    def test_positive(self, sample_ohlcv):
        result = atr(sample_ohlcv, 14)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_correct_length(self, sample_ohlcv):
        result = atr(sample_ohlcv, 14)
        assert len(result) == len(sample_ohlcv)
