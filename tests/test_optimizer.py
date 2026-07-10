"""Tests for core.optimizer — TradingCalendar, ParamScanner, EnsembleWeight."""
import json
import numpy as np
import pandas as pd
from datetime import date, datetime, time as dt_time
from pathlib import Path
from core.optimizer import (
    TradingCalendar, ParamScanner, EnsembleWeight,
    load_latest_weights, load_latest_params,
)
from core.strategies import MATrend, ALL_STRATEGY_CLASSES


class TestTradingCalendar:
    def test_weekday_detection(self):
        cal = TradingCalendar()
        # Monday through Friday
        for d in [date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8),
                  date(2026, 7, 9), date(2026, 7, 10)]:
            assert cal.is_weekday(d), f"{d} should be weekday"
        # Saturday, Sunday
        assert not cal.is_weekday(date(2026, 7, 11))
        assert not cal.is_weekday(date(2026, 7, 12))

    def test_already_optimized_nonexistent(self):
        cal = TradingCalendar(results_dir="results/test_optimizer")
        assert not cal.already_optimized(date(2020, 1, 1))

    def test_should_run_weekend_false(self, monkeypatch):
        cal = TradingCalendar(results_dir="results/test_optimizer")
        # Saturday
        sat = date(2026, 7, 11)
        assert not cal.should_run(sat), "Should not run on weekend"


class TestParamScanner:
    def test_scan_single_returns_results(self, sample_ohlcv):
        scanner = ParamScanner()
        results = scanner.scan_single(
            sample_ohlcv, "趋势-均线", MATrend,
            param_grid={"fast": [5, 10], "slow": [20, 30]},
        )
        assert isinstance(results, list)
        if results:  # may be empty if no valid combos
            assert "params" in results[0]
            assert "score" in results[0]
            assert "sharpe" in results[0]

    def test_scan_single_sorted_by_score(self, sample_ohlcv):
        scanner = ParamScanner()
        results = scanner.scan_single(
            sample_ohlcv, "趋势-均线", MATrend,
            param_grid={"fast": [5, 20], "slow": [30, 50]},
        )
        if len(results) >= 2:
            assert results[0]["score"] >= results[1]["score"]

    def test_scan_single_no_param_grid(self, sample_ohlcv):
        """Strategy without param_grid returns empty."""
        class NoGrid(MATrend):
            param_grid = {}
        scanner = ParamScanner()
        results = scanner.scan_single(sample_ohlcv, "test", NoGrid)
        assert results == []

    def test_score_weights_default(self):
        scanner = ParamScanner()
        # High Sharpe, low drawdown → good score
        score_good = scanner._compute_score(1.5, -0.1, 0.6, 2.0)
        # Low Sharpe, high drawdown → worse score
        score_bad = scanner._compute_score(-0.5, -0.4, 0.3, 0.5)
        assert score_good > score_bad


class TestEnsembleWeight:
    def test_weights_sum_to_one(self):
        ew = EnsembleWeight()
        perf = {
            "趋势-均线": [0.001, -0.002, 0.003] * 7,
            "趋势-MACD": [0.002, 0.001, -0.001] * 7,
            "动量": [0.0, 0.0, 0.001] * 7,
            "量确认": [-0.001, 0.002, 0.0] * 7,
            "反转": [0.003, -0.001, 0.002] * 7,
        }
        weights = ew.update(perf)
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.02, f"Weights sum to {total}"

    def test_min_weight_enforced(self):
        ew = EnsembleWeight(config={"window_days": 20, "temperature": 1.0, "min_weight": 0.05})
        perf = {
            "趋势-均线": [0.01] * 20,   # Much better
            "趋势-MACD": [-0.01] * 20,  # Worse
            "动量": [0.0] * 20,
            "量确认": [0.0] * 20,
            "反转": [0.0] * 20,
        }
        weights = ew.update(perf)
        for name, w in weights.items():
            assert w >= 0.04, f"{name} weight {w} below min"

    def test_default_weights_when_empty(self):
        ew = EnsembleWeight()
        w = ew.get_weights()
        assert len(w) == 5
        for v in w.values():
            assert abs(v - 0.2) < 0.01

    def test_save_and_load(self, tmp_path):
        ew = EnsembleWeight()
        perf = {"趋势-均线": [0.001] * 20, "趋势-MACD": [0.002] * 20}
        ew.update(perf)
        ew.save(path=str(tmp_path))

        # Check file exists
        today = date.today().isoformat()
        wf = tmp_path / today / "weights.json"
        assert wf.exists()
        with open(wf) as f:
            data = json.load(f)
        assert "weights" in data
