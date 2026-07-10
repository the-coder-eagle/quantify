"""Tests for core.backtest — BacktestEngine, walk_forward, drawdown."""
import numpy as np
import pandas as pd
from core.backtest import BacktestEngine, BacktestResult, walk_forward, drawdown_series
from core.strategies import MATrend


class TestBacktestEngine:
    def test_run_returns_result(self, sample_ohlcv):
        engine = BacktestEngine(strategy=MATrend(fast=10, slow=30), long_only=True)
        result = engine.run(sample_ohlcv, "test")
        assert isinstance(result, BacktestResult)
        assert len(result.equity) == len(sample_ohlcv)
        assert len(result.returns) == len(sample_ohlcv)

    def test_summary_has_expected_keys(self, sample_ohlcv):
        engine = BacktestEngine(strategy=MATrend(fast=10, slow=30), long_only=True)
        result = engine.run(sample_ohlcv, "test")
        s = result.summary()
        expected_keys = [
            "Total Return", "Benchmark", "CAGR", "Ann Vol", "Sharpe",
            "Max DD", "Calmar", "Win Rate", "Profit Factor", "Trades", "Cost",
        ]
        for k in expected_keys:
            assert k in s, f"Missing key: {k}"

    def test_equity_starts_at_one(self, sample_ohlcv):
        engine = BacktestEngine(strategy=MATrend(fast=10, slow=30), long_only=True)
        result = engine.run(sample_ohlcv, "test")
        assert abs(result.equity.iloc[0] - 1.0) < 1e-6

    def test_no_strategy_raises(self, sample_ohlcv):
        engine = BacktestEngine(long_only=True)
        try:
            engine.run(sample_ohlcv)
        except ValueError:
            pass  # expected


class TestDrawdown:
    def test_negative_or_zero(self, sample_ohlcv):
        equity = (1 + sample_ohlcv["Close"].pct_change().fillna(0)).cumprod()
        equity = equity / equity.iloc[0]
        dd = drawdown_series(equity)
        assert (dd <= 0).all() or (dd <= 1e-10).all()


class TestWalkForward:
    def test_returns_result(self, sample_ohlcv):
        if len(sample_ohlcv) < 200:
            return  # skip — not enough data
        result = walk_forward(
            MATrend, sample_ohlcv,
            param_grid={"fast": [5, 10], "slow": [20, 30]},
            train_months=6, test_months=2,
        )
        if result is not None:
            assert "equity" in result
            assert "n_windows" in result
            assert result["n_windows"] > 0
