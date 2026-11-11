"""
AI Agent with application-level tool validation for strategy research.

The AI can call named read tools (backtest, market data, performance),
and save parameter/weight proposals bounded by hardcoded validation limits.
All write operations are audit-logged to results/audit/{date}.jsonl.

No arbitrary filesystem or network tool, real-trading tool, or audit deletion
tool is exposed. Named tools can fetch market data and write to fixed paths.
This is not an operating-system security sandbox.
"""

import json, time, os
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import numpy as np
from loguru import logger

# ── Hardcoded safety bounds (AI cannot bypass these) ──

PARAM_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "趋势-均线":   {"fast": (5, 60), "slow": (20, 120)},
    "趋势-MACD":  {"fast": (5, 30), "slow": (15, 50), "signal": (5, 20)},
    "动量":        {"lookback": (10, 120), "threshold": (-0.05, 0.05)},
    "量确认":      {"fast": (5, 40), "slow": (20, 80), "vol_mult": (0.8, 2.0)},
    "反转":        {"bb_window": (10, 40), "oversold": (15, 40), "overbought": (60, 85)},
}

MAX_WEIGHT_CHANGE = 0.30      # max single adjustment
MIN_STRATEGY_WEIGHT = 0.05    # floor per strategy
MAX_FORCE_RESCAN_PER_DAY = 3
MAX_WRITE_OPS_PER_MINUTE = 10
MAX_WATCHLIST_SIZE = 20
MAX_TOOL_CALLS_PER_TURN = 5


# ─────────────────────────────────────────────────────────────
# Audit Logger
# ─────────────────────────────────────────────────────────────

class AuditLogger:
    """Writes an append-only JSONL log of all AI write operations."""

    def __init__(self, audit_dir="results/audit"):
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    def log(self, tool_name: str, args: dict, result: dict, reason: str = ""):
        entry = {
            "time": datetime.now().isoformat(),
            "tool": tool_name,
            "args": args,
            "result": result,
            "reason": reason,
        }
        today = date.today().isoformat()
        path = self.audit_dir / f"{today}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# ─────────────────────────────────────────────────────────────
# Rate Limiter
# ─────────────────────────────────────────────────────────────

class RateLimiter:
    """Track write operations per minute and force_rescan per day."""

    def __init__(self):
        self._write_times: list[float] = []
        self._force_rescan_count = 0
        self._rescan_date = date.today()

    def check_write(self) -> bool:
        """True if under the per-minute write limit."""
        now = time.time()
        cutoff = now - 60
        self._write_times = [t for t in self._write_times if t > cutoff]
        if len(self._write_times) >= MAX_WRITE_OPS_PER_MINUTE:
            return False
        self._write_times.append(now)
        return True

    def check_rescan(self) -> bool:
        """True if under the per-day force_rescan limit."""
        today = date.today()
        if today != self._rescan_date:
            self._force_rescan_count = 0
            self._rescan_date = today
        if self._force_rescan_count >= MAX_FORCE_RESCAN_PER_DAY:
            return False
        self._force_rescan_count += 1
        return True


# ─────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────

class ToolRegistry:
    """Holds all available tools with their implementations and bounds."""

    def __init__(self):
        self._tools: dict[str, dict] = {}
        self._register_all()

    def _register_all(self):
        # Read tools
        self._register("run_backtest", self._run_backtest, "read")
        self._register("get_market_snapshot", self._get_market_snapshot, "read")
        self._register("get_stock_history", self._get_stock_history, "read")
        self._register("get_optimization_results", self._get_optimization_results, "read")
        self._register("get_performance", self._get_performance, "read")
        self._register("get_current_params", self._get_current_params, "read")
        # Write tools
        self._register("adjust_strategy_params", self._adjust_strategy_params, "write")
        self._register("adjust_ensemble_weights", self._adjust_ensemble_weights, "write")
        self._register("add_to_watchlist", self._add_to_watchlist, "write")
        self._register("force_rescan", self._force_rescan, "write")

    def _register(self, name: str, fn: Callable, permission: str):
        self._tools[name] = {"fn": fn, "permission": permission}

    def get_tool(self, name: str) -> dict | None:
        return self._tools.get(name)

    def tool_descriptions(self) -> str:
        """Generate tool descriptions for the AI system prompt."""
        lines = []
        for name, info in self._tools.items():
            perm = "🔓读取" if info["permission"] == "read" else "🔒写入(受限)"
            doc = (info["fn"].__doc__ or "").strip().split("\n")[0]
            lines.append(f"- {name} [{perm}]: {doc}")
        return "\n".join(lines)

    # ── Read tools ──

    def _run_backtest(self, code: str, strategy_name: str, params: dict = None) -> dict:
        """Run backtest on a stock with given strategy and parameters."""
        from .data import DataManager
        from .strategies import ALL_STRATEGY_CLASSES
        from .backtest import BacktestEngine

        dm = DataManager()
        end = date.today().strftime("%Y-%m-%d")
        start = (date.today() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        df = dm.get(code, start, end)
        if isinstance(df, dict):
            df = df.get(code.zfill(6), pd.DataFrame())

        if len(df) < 60:
            return {"error": f"Insufficient data for {code}: {len(df)} rows"}

        cls = next((c for n, c in ALL_STRATEGY_CLASSES if n == strategy_name), None)
        if cls is None:
            return {"error": f"Unknown strategy: {strategy_name}"}

        strat = cls(**(params or {}))
        engine = BacktestEngine(strat, long_only=True)
        result = engine.run(df)
        s = result.summary()
        return {k: str(v) if not isinstance(v, (int, float)) else v for k, v in s.items()}

    def _get_market_snapshot(self, limit: int = 50) -> dict:
        """Get current A-share market snapshot (top N by turnover)."""
        from .data import market_snapshot
        from config import app_max_price, app_min_price
        df = market_snapshot(
            limit=limit, min_price=app_min_price(), max_price=app_max_price()
        )
        if df.empty:
            return {"error": "No market data available"}
        return {"count": len(df), "data": df.head(limit).to_dict(orient="records")}

    def _get_stock_history(self, code: str, days: int = 250) -> dict:
        """Get historical OHLCV data for a stock."""
        from .data import DataManager
        dm = DataManager()
        end = date.today().strftime("%Y-%m-%d")
        start = (date.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        df = dm.get(code, start, end)
        if isinstance(df, dict):
            df = df.get(code.zfill(6), pd.DataFrame())
        if df.empty:
            return {"error": f"No data for {code}"}
        latest = df.iloc[-1]
        return {
            "code": code,
            "rows": len(df),
            "latest_close": float(latest["Close"]),
            "latest_volume": int(latest["Volume"]) if "Volume" in df.columns else 0,
        }

    def _get_optimization_results(self) -> dict:
        """Read latest optimization results from disk."""
        from .optimizer import load_latest_params, load_latest_weights
        params = load_latest_params() or {}
        weights = load_latest_weights() or {}
        return {"best_params": params, "weights": weights}

    def _get_performance(self, strategy_name: str = None) -> dict:
        """Get recent strategy performance metrics (from optimizer results)."""
        results = self._get_optimization_results()
        if strategy_name:
            bp = results.get("best_params", {}).get(strategy_name)
            w = results.get("weights", {}).get(strategy_name, 0)
            return {"strategy": strategy_name, "best_params": bp, "weight": w}
        return results

    def _get_current_params(self) -> dict:
        """Get all currently active strategy parameters and weights."""
        from .optimizer import load_latest_params, load_latest_weights
        from .strategies import ALL_STRATEGIES
        params = load_latest_params() or {}
        weights = load_latest_weights() or {}
        # Return actual default parameter values, not class names
        defaults = {}
        for name, strat in ALL_STRATEGIES:
            cfg = {k: v for k, v in strat.__dict__.items()
                   if not k.startswith("_") and k != "param_grid"}
            defaults[name] = cfg
        return {
            "params": params,
            "weights": weights,
            "defaults": defaults,
        }

    # ── Write tools ──

    def _adjust_strategy_params(self, strategy_name: str, params: dict,
                                reason: str = "", override_dir: str = None) -> dict:
        """Save a strategy parameter proposal; not automatically activated."""
        bounds = PARAM_BOUNDS.get(strategy_name)
        if bounds is None:
            return {"error": f"Unknown strategy: {strategy_name}. Valid: {list(PARAM_BOUNDS.keys())}"}

        if not isinstance(params, dict) or not params:
            return {"error": "Params must be a nonempty object"}

        # Validate each parameter against bounds
        for key, val in params.items():
            if key not in bounds:
                return {"error": f"Unknown param '{key}' for {strategy_name}. Valid: {list(bounds.keys())}"}
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                return {"error": f"Param '{key}={val}' must be numeric, got {type(val).__name__}"}
            if key in {"fast", "slow", "signal", "lookback", "bb_window"} and not isinstance(val, int):
                return {"error": f"Param '{key}' must be an integer"}
            lo, hi = bounds[key]
            if not (lo <= val <= hi):
                return {
                    "error": f"Param '{key}={val}' out of bounds [{lo}, {hi}] for {strategy_name}",
                    "rejected": True,
                }

        # Apply: write to disk as a manual override
        base = Path(override_dir) if override_dir else Path("results/optimizer")
        override_path = base / "ai_overrides.json"
        override_path.parent.mkdir(parents=True, exist_ok=True)
        overrides = {}
        if override_path.exists():
            with open(override_path, encoding="utf-8") as f:
                overrides = json.load(f)
        overrides.setdefault("params", {})
        overrides["params"][strategy_name] = params
        overrides["_last_modified"] = datetime.now().isoformat()
        overrides["_reason"] = reason
        with open(override_path, "w", encoding="utf-8") as f:
            json.dump(overrides, f, ensure_ascii=False, indent=2)

        return {
            "status": "saved",
            "active": False,
            "message": "Saved as an override proposal; runtime loaders do not apply AI overrides.",
            "strategy": strategy_name,
            "params": params,
            "bounds_checked": True,
        }

    def _adjust_ensemble_weights(self, weights: dict[str, float],
                                 reason: str = "", override_dir: str = None) -> dict:
        """Save inactive weight proposals: sum = 1.0, each >= 5%, max change <= 30%."""
        from .optimizer import load_latest_weights

        if not isinstance(weights, dict) or set(weights) != set(PARAM_BOUNDS):
            return {"error": "Weights must include exactly all five known strategies"}
        if any(isinstance(w, bool) or not isinstance(w, (int, float))
               or not math.isfinite(w) for w in weights.values()):
            return {"error": "Weights must be finite numbers"}

        # Validate sum
        total = sum(weights.values())
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-8):
            return {"error": f"Weights must sum to 1.0, got {total:.3f}"}

        # Validate individual weights
        for name, w in weights.items():
            if w < MIN_STRATEGY_WEIGHT:
                return {"error": f"Weight for '{name}' ({w:.2f}) below minimum {MIN_STRATEGY_WEIGHT}"}

        # Check max change vs current (respect override_dir for testability)
        results_dir = override_dir if override_dir else "results/optimizer"
        current = load_latest_weights(results_dir=results_dir) or dict.fromkeys(PARAM_BOUNDS, 0.2)
        if current:
            for name, w in weights.items():
                prev = current.get(name, 0.2)
                if abs(w - prev) > MAX_WEIGHT_CHANGE:
                    return {
                        "error": f"Weight change for '{name}' ({prev:.2f}→{w:.2f}) exceeds max {MAX_WEIGHT_CHANGE:.0%}",
                        "rejected": True,
                    }

        # Apply
        base = Path(override_dir) if override_dir else Path("results/optimizer")
        override_path = base / "ai_overrides.json"
        override_path.parent.mkdir(parents=True, exist_ok=True)
        overrides = {}
        if override_path.exists():
            with open(override_path, encoding="utf-8") as f:
                overrides = json.load(f)
        overrides["weights"] = weights
        overrides["_last_modified"] = datetime.now().isoformat()
        overrides["_reason"] = reason
        with open(override_path, "w", encoding="utf-8") as f:
            json.dump(overrides, f, ensure_ascii=False, indent=2)

        return {
            "status": "saved",
            "active": False,
            "message": "Saved as an override proposal; runtime loaders do not apply AI overrides.",
            "weights": weights,
            "max_change_checked": True,
            "min_weight_checked": True,
        }

    def _add_to_watchlist(self, code: str, reason: str = "", override_dir: str = None) -> dict:
        """Add a stock to the AI watchlist (max 20 stocks)."""
        base = Path(override_dir) if override_dir else Path("results/optimizer")
        wl_path = base / "ai_watchlist.json"
        wl_path.parent.mkdir(parents=True, exist_ok=True)

        watchlist = []
        if wl_path.exists():
            with open(wl_path, encoding="utf-8") as f:
                watchlist = json.load(f)

        if len(watchlist) >= MAX_WATCHLIST_SIZE:
            return {"error": f"Watchlist full (max {MAX_WATCHLIST_SIZE})"}

        if code not in watchlist:
            watchlist.append(code)
            with open(wl_path, "w", encoding="utf-8") as f:
                json.dump(watchlist, f, ensure_ascii=False)
            return {"status": "added", "code": code, "watchlist_size": len(watchlist)}

        return {"status": "already_exists", "code": code}

    def _force_rescan(self) -> dict:
        """Trigger an immediate parameter rescan (max 3 per day). Not implemented here — scheduled via scheduler.py."""
        return {"status": "not_implemented", "message": "Use 'python scheduler.py --force' to trigger rescan"}


# ─────────────────────────────────────────────────────────────
# AIAgent
# ─────────────────────────────────────────────────────────────

class AIAgent:
    """AI agent with sandboxed tool access for Quantify.

    Usage:
        from ai import AIClient
        from core.ai_agent import AIAgent

        client = AIClient(api_key="sk-...")
        agent = AIAgent(client)

        # Single-turn with tools
        result = agent.chat("分析600519并优化参数",
                            market_context="全市场500只 | 上涨300下跌200")
    """

    def __init__(self, ai_client):
        self.ai = ai_client
        self.tools = ToolRegistry()
        self.audit = AuditLogger()
        self.rate_limiter = RateLimiter()

        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return f"""你是Quantify量化交易系统的AI代理。用户是大学生，资金约1200元人民币。

你有以下工具可用（通过JSON格式调用）：

{self.tools.tool_descriptions()}

规则：
1. 分析股票时优先考虑用户买得起的（股价×100股 ≤ 1200元）
2. 调整参数时必须在安全范围内，你会收到边界违规的错误提示
3. 先读取数据了解情况，再做出修改建议
4. 每次回复最多调用{MAX_TOOL_CALLS_PER_TURN}个工具
5. 用中文回复，给出具体的数字和理由

工具调用格式（放在回复末尾）：
<function_call>
{{"name": "run_backtest", "arguments": {{"code": "600519", "strategy_name": "趋势-均线", "params": {{"fast": 15, "slow": 45}}}}}}
</function_call>"""

    def execute_tool(self, tool_name: str, arguments: dict, reason: str = "") -> dict:
        """Execute a tool with permission and bounds checking. Returns result dict."""
        if not isinstance(tool_name, str):
            return {"error": "Tool name must be a string"}
        if not isinstance(arguments, dict):
            return {"error": "Tool arguments must be an object"}
        # override_dir is only a trusted Python/test hook, never model input.
        if "override_dir" in arguments:
            return {"error": "Tool callers cannot choose filesystem destinations"}
        tool = self.tools.get_tool(tool_name)
        if tool is None:
            return {"error": f"Unknown tool: {tool_name}"}

        # Rate-limit write operations
        if tool["permission"] == "write":
            if tool_name == "force_rescan" and not self.rate_limiter.check_rescan():
                return {"error": f"force_rescan limit reached ({MAX_FORCE_RESCAN_PER_DAY}/day)"}
            if not self.rate_limiter.check_write():
                return {"error": f"Write rate limit reached ({MAX_WRITE_OPS_PER_MINUTE}/min)"}

        try:
            result = tool["fn"](**arguments)
        except Exception as e:
            result = {"error": str(e)}
            logger.warning(f"Tool '{tool_name}' failed: {e}")

        # Audit-log write operations
        if tool["permission"] == "write":
            self.audit.log(tool_name, arguments, result, reason)

        return result

    def chat(self, user_message: str, market_context: str = "",
             stock_info: str = "") -> str:
        """Single-turn chat with tool calling. Returns AI response text.

        The AI may include <function_call> blocks in its response.
        These are parsed, executed, and the results are appended.
        """
        prompt_parts = [self._system_prompt]
        if market_context:
            prompt_parts.append(f"\n当前市场数据:\n{market_context}")
        if stock_info:
            prompt_parts.append(f"\n个股数据:\n{stock_info}")

        full_prompt = "\n".join(prompt_parts)
        response = self.ai.chat(full_prompt, user_message)

        # Parse and execute tool calls
        if "<function_call>" in response:
            parts = response.split("<function_call>")
            text_response = parts[0].strip()
            tool_results = []

            for part in parts[1:MAX_TOOL_CALLS_PER_TURN + 1]:
                call_text = part.split("</function_call>")[0].strip()
                try:
                    call = json.loads(call_text)
                    if not isinstance(call, dict):
                        tool_results.append("[解析失败] Tool call must be an object")
                        continue
                    name = call.get("name", "")
                    args = call.get("arguments", {})
                    reason = call.get("reason", "")
                    result = self.execute_tool(name, args, reason)
                    tool_results.append(f"[工具: {name}] {json.dumps(result, ensure_ascii=False)}")
                except json.JSONDecodeError:
                    tool_results.append(f"[解析失败] {call_text[:200]}")

            if len(parts) - 1 > MAX_TOOL_CALLS_PER_TURN:
                tool_results.append(f"[限制] 最多执行{MAX_TOOL_CALLS_PER_TURN}个工具，已跳过其余调用")

            if tool_results:
                # Send tool results back to AI for final response
                followup = self.ai.chat(
                    full_prompt,
                    f"工具执行结果:\n" + "\n".join(tool_results) + "\n\n请基于这些结果给用户最终答复。",
                )
                return text_response + "\n\n---\n**工具执行结果:**\n" + "\n".join(tool_results) + "\n\n**AI:** " + followup

        return response
