"""Market regime detection. Tells you WHAT kind of market we're in.

Thresholds are loaded from ``config/settings.yaml`` → ``regime`` section.
Custom thresholds can be passed via the ``thresholds`` dict parameter.
"""
import pandas as pd, numpy as np
from .indicators import sma, rsi, atr

# Defaults (used as fallback if config is unavailable)
_DEFAULT_THRESHOLDS = {
    "vol_change_high": 1.5,
    "vol_change_low": 0.7,
    "ret_threshold_low_vol": 0.03,
}


def _load_thresholds():
    """Load regime thresholds from unified config, with defaults as fallback."""
    try:
        from config import get as cfg_get
        return {
            "vol_change_high": cfg_get("regime", "vol_change_high", default=1.5),
            "vol_change_low": cfg_get("regime", "vol_change_low", default=0.7),
            "ret_threshold": cfg_get("regime", "ret_threshold_low_vol", default=0.03),
        }
    except Exception:
        return dict(_DEFAULT_THRESHOLDS,
                    ret_threshold=_DEFAULT_THRESHOLDS["ret_threshold_low_vol"])


def detect_regime(df: pd.DataFrame, thresholds: dict = None) -> dict:
    """Detect current market regime from OHLCV data.

    Classifies the market into one of five regimes:
    - trending_up:    price > MA20 > MA50 and 20-day return > 0
    - trending_down:  price < MA20 < MA50 and 20-day return < 0
    - high_volatility:  20-day vol > threshold × 60-day vol
    - low_volatility:   20-day vol < threshold × 60-day vol AND |ret_20| < threshold
    - ranging:        none of the above

    Thresholds (loaded from config by default):
    - vol_change_high (1.5):  20d vol / 60d vol ratio for "high vol"
    - vol_change_low (0.7):   20d vol / 60d vol ratio for "low vol"
    - ret_threshold (0.03):   max |20d return| for confirming low_volatility
    """
    if thresholds is None:
        thresholds = _load_thresholds()

    c = df["Close"]; n = len(c)
    if n < 60: return {"regime": "insufficient_data", "trend_strength": 0}

    # 1. Trend detection: is MA20 > MA50 > MA200? (alignment)
    ma20 = sma(c, 20).iloc[-1]; ma50 = sma(c, 50).iloc[-1]
    ma200 = sma(c, min(200, n)).iloc[-1] if n >= 200 else ma50
    price = c.iloc[-1]
    trend_aligned = (price > ma20 > ma50) if n < 200 else (price > ma20 > ma50 > ma200)
    trend_reversed = (price < ma20 < ma50) if n < 200 else (price < ma20 < ma50 < ma200)

    # 2. ADX-like: is price consistently moving in one direction?
    ret_20 = (c.iloc[-1] / c.iloc[-min(21, n)] - 1)
    ret_60 = (c.iloc[-1] / c.iloc[-min(61, n)] - 1)

    # 3. Volatility regime
    vol_20 = c.pct_change(fill_method=None).tail(20).std() * np.sqrt(252)
    vol_60 = c.pct_change(fill_method=None).tail(60).std() * np.sqrt(252)
    vol_change = vol_20 / vol_60 if vol_60 > 0 else 1.0

    # 4. RSI for momentum regime
    r = rsi(c, 14).iloc[-1]

    # Classify — thresholds from config
    tvh = thresholds.get("vol_change_high", 1.5)
    tvl = thresholds.get("vol_change_low", 0.7)
    tret = thresholds.get("ret_threshold", 0.03)

    trend_strength = abs(ret_60)
    if trend_aligned and ret_20 > 0:          regime = "trending_up"
    elif trend_reversed and ret_20 < 0:       regime = "trending_down"
    elif vol_change > tvh:                    regime = "high_volatility"
    elif vol_change < tvl and abs(ret_20) < tret: regime = "low_volatility"
    else:                                     regime = "ranging"

    return {
        "regime": regime,
        "trend_strength": round(trend_strength, 3),
        "vol_20": round(vol_20, 3),
        "vol_change": round(vol_change, 2),
        "rsi": round(r, 1),
        "price_vs_ma20": "above" if price > ma20 else "below",
        "price_vs_ma50": "above" if price > ma50 else "below",
    }

def regime_strategy_weights(regime_info: dict) -> dict:
    """Convert regime to strategy weights. Trending=trend strategies, Ranging=reversion."""
    r = regime_info.get("regime", "ranging")
    if r == "trending_up" or r == "trending_down":
        return {"trend": 0.40, "momentum": 0.30, "reversion": 0.10, "volume": 0.20}
    elif r == "high_volatility":
        return {"trend": 0.20, "momentum": 0.15, "reversion": 0.35, "volume": 0.30}
    elif r == "low_volatility":
        return {"trend": 0.30, "momentum": 0.25, "reversion": 0.15, "volume": 0.30}
    else:  # ranging
        return {"trend": 0.15, "momentum": 0.15, "reversion": 0.50, "volume": 0.20}

def regime_risk_multiplier(regime_info: dict) -> float:
    """Higher vol = reduce position size. Returns multiplier for risk_pct."""
    r = regime_info.get("regime", "")
    vol_change = regime_info.get("vol_change", 1.0)
    if r == "high_volatility": return 0.5
    if r == "low_volatility": return 1.2
    if vol_change > 1.3: return 0.7
    return 1.0
