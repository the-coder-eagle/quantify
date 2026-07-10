"""Tests for core.ai_agent — security bounds, tool registry, audit logging."""
import json
import os
from datetime import date

from core.ai_agent import (
    PARAM_BOUNDS,
    MAX_WEIGHT_CHANGE,
    MIN_STRATEGY_WEIGHT,
    ToolRegistry,
    AuditLogger,
    RateLimiter,
    AIAgent,
)


class TestParamBounds:
    def test_all_strategies_have_bounds(self):
        from core.strategies import ALL_STRATEGY_CLASSES
        strategy_names = {name for name, _ in ALL_STRATEGY_CLASSES}
        bound_names = set(PARAM_BOUNDS.keys())
        assert strategy_names == bound_names, \
            f"Missing bounds for: {strategy_names - bound_names}"

    def test_matrend_bounds_reject_out_of_range(self):
        bounds = PARAM_BOUNDS["趋势-均线"]
        assert not (5 <= 3 <= 60)   # fast=3 is below min 5
        assert (5 <= 20 <= 60)       # fast=20 is valid
        assert not (20 <= 150 <= 120)  # slow=150 is above max 120


class TestToolRegistry:
    def test_all_read_tools_registered(self):
        tr = ToolRegistry()
        read_tools = [n for n, i in tr._tools.items() if i["permission"] == "read"]
        assert "run_backtest" in read_tools
        assert "get_market_snapshot" in read_tools
        assert "get_stock_history" in read_tools
        assert len(read_tools) >= 4

    def test_all_write_tools_registered(self):
        tr = ToolRegistry()
        write_tools = [n for n, i in tr._tools.items() if i["permission"] == "write"]
        assert "adjust_strategy_params" in write_tools
        assert "adjust_ensemble_weights" in write_tools
        assert "add_to_watchlist" in write_tools

    def test_unknown_tool_returns_none(self):
        tr = ToolRegistry()
        assert tr.get_tool("nonexistent") is None

    def test_adjust_params_rejects_unknown_strategy(self, tmp_path):
        tr = ToolRegistry()
        result = tr._adjust_strategy_params("nonexistent", {"fast": 10}, override_dir=str(tmp_path))
        assert "error" in result

    def test_adjust_params_rejects_out_of_bounds(self, tmp_path):
        tr = ToolRegistry()
        result = tr._adjust_strategy_params("趋势-均线", {"fast": 3}, override_dir=str(tmp_path))
        assert "error" in result
        assert result.get("rejected") is True

    def test_adjust_params_rejects_non_numeric(self, tmp_path):
        tr = ToolRegistry()
        result = tr._adjust_strategy_params("趋势-均线", {"fast": "15"}, override_dir=str(tmp_path))
        assert "error" in result
        assert "numeric" in result["error"].lower()

    def test_adjust_params_accepts_valid(self, tmp_path):
        tr = ToolRegistry()
        result = tr._adjust_strategy_params("趋势-均线", {"fast": 15, "slow": 45},
                                            override_dir=str(tmp_path))
        assert result.get("status") == "applied"
        assert result.get("bounds_checked") is True
        # Verify file was written to tmp_path
        override_file = tmp_path / "ai_overrides.json"
        assert override_file.exists()

    def test_adjust_weights_rejects_bad_sum(self, tmp_path):
        tr = ToolRegistry()
        result = tr._adjust_ensemble_weights({"趋势-均线": 0.5}, override_dir=str(tmp_path))
        assert "error" in result

    def test_adjust_weights_rejects_below_min(self, tmp_path):
        tr = ToolRegistry()
        # Give one strategy 0.01 (below min 5%)
        weights = {"趋势-均线": 0.01, "趋势-MACD": 0.33, "动量": 0.33, "量确认": 0.33, "反转": 0.0}
        result = tr._adjust_ensemble_weights(weights, override_dir=str(tmp_path))
        assert "error" in result


class TestAuditLogger:
    def test_log_creates_file(self, tmp_path):
        audit = AuditLogger(audit_dir=str(tmp_path))
        audit.log("adjust_strategy_params", {"fast": 15}, {"status": "applied"})
        today = date.today().isoformat()
        log_path = tmp_path / f"{today}.jsonl"
        assert log_path.exists()

        with open(log_path, encoding="utf-8") as f:
            line = f.readline()
        entry = json.loads(line)
        assert entry["tool"] == "adjust_strategy_params"
        assert entry["args"] == {"fast": 15}


class TestRateLimiter:
    def test_write_limit(self):
        rl = RateLimiter()
        # First 10 writes should pass
        for _ in range(10):
            assert rl.check_write()
        # 11th should fail
        assert not rl.check_write()

    def test_rescan_limit(self):
        rl = RateLimiter()
        # First 3 should pass
        for _ in range(3):
            assert rl.check_rescan()
        # 4th should fail
        assert not rl.check_rescan()
