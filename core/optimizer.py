"""Auto-optimizer: ParamScanner + EnsembleWeight + TradingCalendar.

Automatically tests strategy parameter combinations, scores them via backtest,
and dynamically adjusts strategy voting weights based on recent performance.

Usage:
    from core.optimizer import TradingCalendar, ParamScanner, EnsembleWeight

    # Check if today is a trading day
    cal = TradingCalendar()
    if cal.should_run():
        scanner = ParamScanner()
        results = scanner.scan_all(stock_data)
        scanner.save(results)

        ew = EnsembleWeight()
        weights = ew.update({"strategy": [0.01, -0.01, 0.02, 0.0, 0.01]})
"""

import os, json, time
from datetime import date, datetime, time as dt_time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from .strategies import ALL_STRATEGY_CLASSES
from .backtest import BacktestEngine, _sharpe


# ─────────────────────────────────────────────────────────────
# TradingCalendar
# ─────────────────────────────────────────────────────────────

class TradingCalendar:
    """Detect A-share trading days to avoid running on weekends/holidays.

    Uses akshare's trade-date calendar when available, falls back to
    weekday-only check + directory-existence guard.
    """

    def __init__(self, results_dir="results/optimizer"):
        self.results_dir = Path(results_dir)

    def is_weekday(self, d: date = None) -> bool:
        """True if Mon–Fri."""
        d = d or date.today()
        return d.weekday() < 5  # 0=Mon .. 4=Fri

    def is_after_market_close(self) -> bool:
        """True if current time is after 15:00 CST (rough check)."""
        now = datetime.now()
        return now.time() >= dt_time(15, 0)

    def already_optimized(self, d: date = None) -> bool:
        """True if results/{date}/ directory already exists."""
        d = d or date.today()
        path = self.results_dir / d.isoformat()
        return path.exists()

    def _fetch_trade_dates(self) -> set:
        """Try to fetch A-share trading calendar from akshare."""
        try:
            import akshare as ak
            df = ak.tool_trade_date_hist_sina()
            if df is not None and not df.empty:
                col = df.columns[0]
                dates = pd.to_datetime(df[col]).dt.date
                return set(dates)
        except Exception:
            logger.debug("Could not fetch trade calendar from akshare")
        return set()

    def should_run(self, d: date = None, force: bool = False) -> bool:
        """Return True if we should run optimization today.

        Guards: (1) weekday, (2) after market close, (3) not already run,
        (4) is a known trading day (if calendar available).
        ``force=True`` skips all checks except duplicate run.
        """
        if force:
            return not self.already_optimized(d)

        d = d or date.today()

        if self.already_optimized(d):
            logger.info(f"{d}: results already exist, skipping")
            return False

        if not self.is_weekday(d):
            logger.info(f"{d}: weekend, skipping")
            return False

        if not self.is_after_market_close():
            logger.info("Market still open, skipping optimization")
            return False

        trade_dates = self._fetch_trade_dates()
        if trade_dates and d not in trade_dates:
            logger.info(f"{d}: not in trading calendar (holiday?), skipping")
            return False

        return True


# ─────────────────────────────────────────────────────────────
# ParamScanner
# ─────────────────────────────────────────────────────────────

def _load_score_weights():
    """Load score weights from unified config."""
    try:
        from config import get as cfg_get
        return {
            "sharpe": cfg_get("optimizer", "score_weights", "sharpe", default=0.4),
            "max_dd": cfg_get("optimizer", "score_weights", "max_dd", default=0.3),
            "win_rate": cfg_get("optimizer", "score_weights", "win_rate", default=0.2),
            "profit_factor": cfg_get("optimizer", "score_weights", "profit_factor", default=0.1),
        }
    except Exception:
        return {"sharpe": 0.4, "max_dd": 0.3, "win_rate": 0.2, "profit_factor": 0.1}


class ParamScanner:
    """Grid-search strategy parameters, score via backtest, return top-N."""

    def __init__(self, score_weights=None):
        self.score_weights = score_weights or _load_score_weights()

    def _compute_score(self, sharpe, max_dd, win_rate, profit_factor):
        """Composite score: higher is better."""
        w = self.score_weights
        dd_score = max(0.0, 1.0 + max_dd) if max_dd < 0 else 1.0  # max_dd is negative
        pf = min(profit_factor, 5.0) if profit_factor > 0 else 0.0
        return (
            w["sharpe"] * max(sharpe, -2.0)
            + w["max_dd"] * dd_score
            + w["win_rate"] * max(win_rate, 0.0)
            + w["profit_factor"] * (pf / 5.0)
        )

    def scan_single(self, df, strategy_name, strategy_cls, param_grid=None):
        """Scan parameter combinations for one strategy on one stock.

        Returns list of {"params": {...}, "score": float, "sharpe": float, ...}
        sorted by score descending.  Top-3 at most.
        """
        if param_grid is None:
            param_grid = getattr(strategy_cls, "param_grid", None)
        if not param_grid:
            logger.warning(f"{strategy_name}: no param_grid, skipping")
            return []

        keys = list(param_grid.keys())
        combos = list(product(*param_grid.values()))
        results = []

        for combo in combos:
            params = dict(zip(keys, combo))
            try:
                strat = strategy_cls(**params)
                engine = BacktestEngine(strat, long_only=True)
                result = engine.run(df)
                s = result.summary()

                sharpe = float(s.get("Sharpe", 0))
                max_dd_str = s.get("Max DD", "0%")
                max_dd = float(max_dd_str.replace("%", "")) / 100.0
                wr_str = s.get("Win Rate", "0%")
                win_rate = float(wr_str.replace("%", "")) / 100.0
                pf = float(s.get("Profit Factor", 0))

                # Skip obviously bad or zero-signal results
                trades = int(s.get("Trades", 0))
                if trades == 0:
                    continue

                score = self._compute_score(sharpe, max_dd, win_rate, pf)

                results.append({
                    "params": params,
                    "score": round(score, 4),
                    "sharpe": sharpe,
                    "max_dd": max_dd,
                    "win_rate": win_rate,
                    "profit_factor": pf,
                    "trades": trades,
                })
            except Exception as e:
                logger.debug(f"{strategy_name} {params} failed: {e}")

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:3]

    def scan_all(self, stock_data, strategy_classes=None):
        """Scan all strategies for all stocks.

        Args:
            stock_data: dict of code → DataFrame
            strategy_classes: list of (name, class). Defaults to ALL_STRATEGY_CLASSES.

        Returns:
            dict: code → {"strategies": {name: [top3 results]}, "timestamp": iso}
        """
        if strategy_classes is None:
            strategy_classes = ALL_STRATEGY_CLASSES

        all_results = {}
        for code, df in stock_data.items():
            if len(df) < 60:
                continue
            strat_results = {}
            for name, cls in strategy_classes:
                top = self.scan_single(df, name, cls)
                if top:
                    strat_results[name] = top
            if strat_results:
                all_results[code] = {
                    "strategies": strat_results,
                    "timestamp": datetime.now().isoformat(),
                }
        return all_results

    def save(self, results, target_dir="results/optimizer", label=None):
        """Save scan results to disk."""
        d = label or date.today().isoformat()
        path = Path(target_dir) / d
        path.mkdir(parents=True, exist_ok=True)

        # best_params.json: per-strategy best params across all stocks
        best_params = {}
        for code, info in results.items():
            for sname, top_list in info["strategies"].items():
                if top_list:
                    best = top_list[0]
                    if sname not in best_params:
                        best_params[sname] = best
                    elif best["score"] > best_params[sname]["score"]:
                        best_params[sname] = best

        with open(path / "best_params.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)

        # scan_report.json: full results
        with open(path / "scan_report.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Optimization results saved to {path}")


# ─────────────────────────────────────────────────────────────
# EnsembleWeight
# ─────────────────────────────────────────────────────────────

def _load_ensemble_config():
    """Load ensemble config from unified config."""
    try:
        from config import get as cfg_get
        return {
            "window_days": cfg_get("ensemble", "window_days", default=20),
            "temperature": cfg_get("ensemble", "temperature", default=2.0),
            "min_weight": cfg_get("ensemble", "min_weight", default=0.05),
        }
    except Exception:
        return {"window_days": 20, "temperature": 2.0, "min_weight": 0.05}


class EnsembleWeight:
    """Dynamically adjust strategy voting weights based on recent Sharpe."""

    def __init__(self, config=None):
        cfg = config or _load_ensemble_config()
        self.window_days = cfg["window_days"]
        self.temperature = cfg["temperature"]
        self.min_weight = cfg["min_weight"]
        self._weights: dict[str, float] = {}
        self._performance: dict[str, list[float]] = {}

    def _softmax(self, scores: list[float]) -> list[float]:
        """Softmax with temperature scaling."""
        if not scores:
            return []
        arr = np.array(scores) / self.temperature
        arr = arr - arr.max()  # numerical stability
        exp = np.exp(arr)
        return list(exp / exp.sum())

    def _rolling_sharpe(self, returns: list[float]) -> float:
        """Compute Sharpe from a list of daily returns."""
        if len(returns) < 5:
            return 0.0
        arr = np.array(returns[-self.window_days:])
        mean = arr.mean()
        std = arr.std()
        if std == 0:
            return 0.0
        return mean / std * np.sqrt(252)

    def update(self, recent_performance: dict[str, list[float]]) -> dict[str, float]:
        """Update weights from recent strategy performance.

        Args:
            recent_performance: dict of strategy_name → list of daily returns

        Returns:
            dict of strategy_name → weight (sums to 1.0)
        """
        self._performance = recent_performance

        sharpe_scores = {}
        for name, returns in recent_performance.items():
            sharpe_scores[name] = self._rolling_sharpe(returns)

        return self.update_from_sharpes(sharpe_scores)

    def update_from_sharpes(self, sharpe_scores: dict[str, float]) -> dict[str, float]:
        """Weight precomputed Sharpe estimates, not samples of daily returns."""
        if not sharpe_scores:
            # Fall back to equal weights
            n = len(ALL_STRATEGY_CLASSES)
            self._weights = {name: 1.0 / n for name, _ in ALL_STRATEGY_CLASSES}
            return self._weights

        names = list(sharpe_scores.keys())
        scores = [sharpe_scores[n] for n in names]
        raw_weights = self._softmax(scores)

        # Apply minimum weight constraint
        weights = {}
        remaining = 1.0
        unclamped = []
        for i, name in enumerate(names):
            if raw_weights[i] < self.min_weight:
                weights[name] = self.min_weight
                remaining -= self.min_weight
            else:
                unclamped.append((i, name))

        if unclamped and remaining > 0:
            unclamped_total = sum(raw_weights[i] for i, _ in unclamped)
            for i, name in unclamped:
                weights[name] = max(
                    self.min_weight,
                    (raw_weights[i] / unclamped_total) * remaining,
                )
        elif remaining <= 0:
            # Over-minimum already — renormalize
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

        # Ensure valid
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            weights = {k: v / total for k, v in weights.items()}

        self._weights = weights
        return weights

    def get_weights(self) -> dict[str, float]:
        """Return current weights, or equal weights if never updated."""
        if not self._weights:
            n = len(ALL_STRATEGY_CLASSES)
            return {name: 1.0 / n for name, _ in ALL_STRATEGY_CLASSES}
        return dict(self._weights)

    def save(self, path="results/optimizer"):
        """Save weights to disk."""
        d = date.today().isoformat()
        p = Path(path) / d
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "weights.json", "w", encoding="utf-8") as f:
            json.dump({
                "weights": self._weights,
                "timestamp": datetime.now().isoformat(),
            }, f, ensure_ascii=False, indent=2)


def load_latest_weights(results_dir="results/optimizer") -> dict[str, float] | None:
    """Load the most recent weights from disk. Returns None if not found."""
    base = Path(results_dir)
    if not base.exists():
        return None

    # Find latest date directory with weights.json
    dirs = sorted(
        [d for d in base.iterdir() if d.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    for d in dirs:
        wf = d / "weights.json"
        if wf.exists():
            try:
                with open(wf, encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("weights", {})
            except Exception:
                continue
    return None


def load_latest_params(results_dir="results/optimizer") -> dict | None:
    """Load the most recent best_params from disk. Returns None if not found."""
    base = Path(results_dir)
    if not base.exists():
        return None

    dirs = sorted(
        [d for d in base.iterdir() if d.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    for d in dirs:
        pf = d / "best_params.json"
        if pf.exists():
            try:
                with open(pf, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return None
