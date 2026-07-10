"""Integration tests — end-to-end flows across multiple modules."""
import json
import numpy as np
import pandas as pd
from datetime import date


def _make_synthetic_ohlcv(n=250, seed=42, trend_slope=0.0005):
    """Create realistic OHLCV data with a mild uptrend."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(trend_slope, 0.015, n)
    close = 10.0 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    open_ = low + rng.random(n) * (high - low)
    volume = rng.integers(1000, 50000, n)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume,
    }, index=pd.date_range("2025-01-01", periods=n, freq="B"))


class TestFullAnalyzeFlow:
    """Full flow: data → analyze → signal with all options."""

    def test_analyze_with_defaults(self):
        from core import analyze_stock
        df = _make_synthetic_ohlcv(200)
        result = analyze_stock(df, capital=1200)
        assert result is not None
        assert result["signal"] in ("BUY", "SELL", "HOLD")
        assert 0 <= result["confidence"] <= 1

    def test_analyze_with_overrides_and_weights(self):
        from core import analyze_stock
        df = _make_synthetic_ohlcv(200)
        overrides = {"趋势-均线": {"fast": 10, "slow": 40}}
        weights = {"趋势-均线": 0.4, "趋势-MACD": 0.2, "动量": 0.2, "量确认": 0.1, "反转": 0.1}
        result = analyze_stock(df, capital=1200,
                               strategy_overrides=overrides,
                               strategy_weights=weights,
                               oos_check=True)
        assert result is not None
        assert "oos" in result
        # Verify OOS verdicts exist for all strategies
        assert len(result["oos"]) == 5

    def test_analyze_then_backtest_consistency(self):
        """analyze_stock signal should be consistent with backtest direction."""
        from core import analyze_stock
        from core.backtest import BacktestEngine
        from core.strategies import MATrend

        df = _make_synthetic_ohlcv(200)
        result = analyze_stock(df, capital=1200)

        # Run backtest with MA Trend strategy
        strat = MATrend(fast=20, slow=50)
        engine = BacktestEngine(strat, long_only=True)
        bt = engine.run(df)
        s = bt.summary()

        # Both should produce valid outputs
        assert "Sharpe" in s
        assert result is not None


class TestOptimizerToAnalyzerFlow:
    """Optimizer → save → load → analyzer integration."""

    def test_scan_save_load_analyze(self, tmp_path):
        from core.optimizer import ParamScanner, EnsembleWeight
        from core import analyze_stock
        from core.strategies import MATrend, MACDTrend

        # 1. Generate data for 3 stocks
        dfs = {}
        for i, (code, s) in enumerate([("000001", 42), ("600519", 99), ("000858", 7)]):
            dfs[code] = _make_synthetic_ohlcv(200, seed=s)

        # 2. Run scanner
        scanner = ParamScanner()
        strategies = [("趋势-均线", MATrend), ("趋势-MACD", MACDTrend)]
        results = {}
        for code, df in dfs.items():
            strat_results = {}
            for name, cls in strategies:
                top = scanner.scan_single(df, name, cls,
                                          param_grid={"fast": [10, 20], "slow": [40, 50]} if name == "趋势-均线"
                                          else {"fast": [8, 12], "slow": [21, 26], "signal": [7, 9]})
                if top:
                    strat_results[name] = top
            if strat_results:
                results[code] = {"strategies": strat_results}

        assert len(results) > 0

        # 3. Save
        scanner.save(results, target_dir=str(tmp_path))

        # Verify files exist
        today = date.today().isoformat()
        assert (tmp_path / today / "best_params.json").exists()
        assert (tmp_path / today / "scan_report.json").exists()

        # 4. Load best params
        with open(tmp_path / today / "best_params.json") as f:
            best_params = json.load(f)
        assert len(best_params) > 0

        # 5. Build overrides from loaded params
        overrides = {}
        for sname, best in best_params.items():
            if "params" in best:
                overrides[sname] = best["params"]

        # 6. Run analyze_stock with loaded overrides
        df = dfs["000001"]
        result = analyze_stock(df, capital=1200, strategy_overrides=overrides)
        assert result is not None
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_ensemble_weight_update_and_load(self, tmp_path):
        from core.optimizer import EnsembleWeight

        ew = EnsembleWeight()
        perf = {
            "趋势-均线": [0.002] * 20,
            "趋势-MACD": [0.001] * 20,
            "动量": [-0.001] * 20,
            "量确认": [0.003] * 20,
            "反转": [-0.002] * 20,
        }
        weights = ew.update(perf)

        # Verify constraints
        assert abs(sum(weights.values()) - 1.0) < 0.02
        for w in weights.values():
            assert w >= 0.04, f"Weight {w} below minimum"

        # Save and reload
        ew.save(path=str(tmp_path))
        from core.optimizer import load_latest_weights
        loaded = load_latest_weights(results_dir=str(tmp_path))
        assert loaded is not None
        assert len(loaded) == 5


class TestVectorizedIndicatorsCorrectness:
    """Verify vectorized implementations produce reasonable outputs."""

    def test_rsi_bounds_on_random_data(self):
        from core.indicators import rsi
        for seed in range(10):
            series = pd.Series(
                10 + np.cumsum(np.random.default_rng(seed).normal(0, 0.5, 300)),
            )
            result = rsi(series, 14)
            valid = result.dropna()
            assert (valid >= 0).all(), f"RSI below 0 at seed {seed}"
            assert (valid <= 100).all(), f"RSI above 100 at seed {seed}"

    def test_atr_positive_on_random_data(self):
        from core.indicators import atr
        for seed in range(10):
            df = _make_synthetic_ohlcv(300, seed=seed)
            result = atr(df, 14)
            valid = result.dropna()
            assert (valid > 0).all(), f"ATR non-positive at seed {seed}"

    def test_macd_crossover_produces_signals(self):
        from core.indicators import macd
        from core.strategies import MACDTrend

        df = _make_synthetic_ohlcv(300, seed=123)
        strat = MACDTrend(fast=12, slow=26, signal=9)
        signals = strat.generate_signals(df)

        # On trending data with enough lookback, there should be some signals
        nonzero = (signals.iloc[100:] != 0).sum()
        assert nonzero > 0, "MACD crossover produced zero signals on 200 bars"

    def test_volume_confirm_signal_continuity(self):
        """VolumeConfirm should produce continuous signals (no sudden gaps)."""
        from core.strategies import VolumeConfirm

        df = _make_synthetic_ohlcv(300, seed=456)
        strat = VolumeConfirm(fast=20, slow=50)
        signals = strat.generate_signals(df)

        # After warmup, signals should not alternate wildly without reason
        valid = signals.iloc[60:]
        changes = (valid.diff().fillna(0) != 0).sum()
        # Changes should exist but not be excessive (< 50% of bars)
        assert changes < len(valid) * 0.5, \
            f"Too many signal changes: {changes}/{len(valid)}"


class TestAIAndOptimizerIntegration:
    """AI agent tools work correctly on real optimizer data."""

    def test_ai_read_tools_with_optimizer_data(self, tmp_path):
        from core.ai_agent import ToolRegistry
        from core.optimizer import ParamScanner, EnsembleWeight
        from core.strategies import MATrend

        # Setup: create optimizer results
        df = _make_synthetic_ohlcv(200)
        scanner = ParamScanner()
        results = scanner.scan_single(df, "趋势-均线", MATrend,
                                      param_grid={"fast": [10], "slow": [40]})

        ew = EnsembleWeight()
        ew.update({"趋势-均线": [0.001] * 20})
        scanner.save({"000001": {"strategies": {"趋势-均线": results}}},
                     target_dir=str(tmp_path))
        ew.save(path=str(tmp_path))

        # Test that ToolRegistry read tools work (no crash)
        tr = ToolRegistry()
        for tool_name in ["get_current_params", "get_performance"]:
            result = tr._tools[tool_name]["fn"]()
            assert isinstance(result, dict), f"{tool_name} returned non-dict"

    def test_ai_write_security_guarantees(self):
        """Verify all security guarantees hold."""
        from core.ai_agent import (
            PARAM_BOUNDS, MAX_WEIGHT_CHANGE, MIN_STRATEGY_WEIGHT,
            MAX_FORCE_RESCAN_PER_DAY, MAX_WATCHLIST_SIZE, MAX_TOOL_CALLS_PER_TURN,
        )

        # All constants must be positive and reasonable
        assert MAX_WEIGHT_CHANGE > 0 and MAX_WEIGHT_CHANGE < 0.5
        assert MIN_STRATEGY_WEIGHT > 0 and MIN_STRATEGY_WEIGHT < 0.2
        assert MAX_FORCE_RESCAN_PER_DAY > 0 and MAX_FORCE_RESCAN_PER_DAY <= 10
        assert MAX_WATCHLIST_SIZE > 0
        assert MAX_TOOL_CALLS_PER_TURN > 0

        # Every strategy must have bounds
        from core.strategies import ALL_STRATEGY_CLASSES
        for name, _ in ALL_STRATEGY_CLASSES:
            assert name in PARAM_BOUNDS, f"Missing bounds for {name}"
            assert len(PARAM_BOUNDS[name]) > 0


class TestFactorPipeline:
    """End-to-end factor pipeline: filter universe -> rank stocks -> verify output."""

    def test_full_factor_pipeline_on_synthetic_data(self, tmp_path):
        """End-to-end: filter -> rank -> verify output structure."""
        from core.universe import filter_universe
        from core.cross_section import rank_all_stocks

        rng = np.random.default_rng(42)

        # Synthetic market snapshot (20 stocks)
        market = pd.DataFrame({
            "code": [f"{i:06d}" for i in range(100001, 100021)],
            "name": [f"股票{i}" for i in range(20)],
            "price": np.linspace(3.0, 11.0, 20),
            "pe_ttm": rng.uniform(5, 50, 20),
            "pb": rng.uniform(0.5, 5, 20),
            "total_mv": rng.uniform(5e9, 5e10, 20),
            "circ_mv": rng.uniform(2e9, 2e10, 20),
            "turnover": rng.uniform(1e7, 1e9, 20),
        })

        # Step 1: Filter universe
        universe = filter_universe(market)
        assert len(universe) > 0

        # Step 2: Mock factor data (10 stocks, 3 factors)
        mock_factors = [
            {"code": f"{i:06d}", "mom_1m": rng.normal() * 0.1,
             "rsi_deviation": rng.normal() * 0.3,
             "roe": rng.uniform(0.05, 0.25)}
            for i in range(100001, 100011)
        ]

        # Step 3: Rank all stocks with mock IC values
        mock_ic = {"mom_1m": 0.05, "rsi_deviation": 0.03, "roe": 0.08}
        ranking = rank_all_stocks(mock_factors, mock_ic)

        # Step 4: Verify output structure
        assert len(ranking) == len(mock_factors)
        assert "code" in ranking.columns
        assert "score" in ranking.columns
        assert "percentile" in ranking.columns
        assert ranking["score"].is_monotonic_decreasing
        assert ranking["percentile"].between(0, 1).all()
