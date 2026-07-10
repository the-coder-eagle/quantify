"""Tests for core.strategies — all 5 strategies."""
import numpy as np
import pandas as pd
from core.strategies import (
    Strategy,
    MATrend,
    MACDTrend,
    Momentum,
    VolumeConfirm,
    MeanReversion,
    ALL_STRATEGIES,
)


class TestMATrend:
    def test_buy_signal_on_crossover(self, sample_ohlcv):
        strat = MATrend(fast=5, slow=20)
        signals = strat.generate_signals(sample_ohlcv)
        assert signals.isin([-1, 0, 1]).all()
        # After warmup, there should be some non-zero signals
        assert (signals.iloc[20:] != 0).any()

    def test_warmup_period_is_zero(self, sample_ohlcv):
        strat = MATrend(fast=10, slow=50)
        signals = strat.generate_signals(sample_ohlcv)
        # First 'slow' values should be 0
        assert (signals.iloc[:50] == 0).all()


class TestMACDTrend:
    def test_signal_values_valid(self, sample_ohlcv):
        strat = MACDTrend()
        signals = strat.generate_signals(sample_ohlcv)
        assert signals.isin([-1, 0, 1]).all()


class TestMomentum:
    def test_signal_values_valid(self, sample_ohlcv):
        strat = Momentum(lookback=20)
        signals = strat.generate_signals(sample_ohlcv)
        assert signals.isin([-1, 0, 1]).all()
        assert (signals.iloc[:20] == 0).all()


class TestVolumeConfirm:
    def test_signal_values_valid(self, sample_ohlcv):
        strat = VolumeConfirm()
        signals = strat.generate_signals(sample_ohlcv)
        assert signals.isin([-1, 0, 1]).all()


class TestMeanReversion:
    def test_signal_values_valid(self, sample_ohlcv):
        strat = MeanReversion(bb_window=20, rsi_period=14)
        signals = strat.generate_signals(sample_ohlcv)
        assert signals.isin([-1, 0, 1]).all()


class TestAllStrategies:
    def test_has_five_strategies(self):
        assert len(ALL_STRATEGIES) == 5

    def test_all_implement_strategy(self):
        for name, strat in ALL_STRATEGIES:
            assert isinstance(strat, Strategy), f"{name} is not a Strategy"

    def test_all_generate_signals(self, sample_ohlcv):
        for name, strat in ALL_STRATEGIES:
            signals = strat.generate_signals(sample_ohlcv)
            assert len(signals) == len(sample_ohlcv), f"{name} wrong length"
            assert signals.isin([-1, 0, 1]).all(), f"{name} bad values"
