"""Unified configuration loader for Quantify.

All modules import settings from here — no more hardcoded constants.

Usage:
    from config import get, app_capital, app_risk_pct, app_max_price

    capital = app_capital()       # 1200.0
    val = get("regime", "vol_change_high", default=1.5)
"""
import yaml
from pathlib import Path
from functools import lru_cache

_CONFIG_PATH = Path(__file__).parent / "settings.yaml"


@lru_cache(maxsize=1)
def _load():
    """Load settings.yaml once, cache forever."""
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get(*keys, default=None):
    """Get a nested config value: get("app", "capital") -> 1200.0."""
    node = _load()
    for k in keys:
        if isinstance(node, dict):
            node = node.get(k)
            if node is None:
                return default
        else:
            return default
    return node if node is not None else default


# ── Typed accessors for common config values ──

def app_capital() -> float:
    return float(get("app", "capital", default=1200.0))


def app_risk_pct() -> float:
    return float(get("app", "risk_pct", default=0.02))


def app_max_price() -> float:
    return float(get("app", "max_price", default=12.0))


def app_min_price() -> float:
    return float(get("app", "min_price", default=2.0))


def app_scan_default() -> int:
    return int(get("app", "scan_default", default=500))


def research_capital() -> float:
    """Larger capital for research/daily_report purposes."""
    return float(get("account", "research_capital", default=100000.0))
