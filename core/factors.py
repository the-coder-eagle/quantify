"""Multi-factor computation engine for A-share stocks.

Each factor group returns a dict of {factor_name: value}.
Values that can't be computed return None (treated as NaN in ranking).
"""

import pandas as pd
import numpy as np
from loguru import logger


# -- Factor freshness (used to discount stale factors in IC weighting) --

FRESHNESS = {
    "value": "daily",
    "momentum": "daily",
    "quality": "quarterly",
    "volatility": "daily",
    "size": "daily",
    "north_bound": "daily",
    "sector_momentum": "daily",
    "technical": "daily",
    "sentiment": "on_demand",
    "margin": "daily",
    "fund_flow": "daily",
}


def compute_all_factors(df_ohlcv, snapshot_row) -> dict[str, float | None]:
    """Compute all factors for a single stock.

    Args:
        df_ohlcv: OHLCV DataFrame with Date index
        snapshot_row: Single row from market_snapshot DataFrame

    Returns:
        Flat dict of {factor_name: value}. None = unavailable.
    """
    close = df_ohlcv["Close"]
    high = df_ohlcv["High"]
    low = df_ohlcv["Low"]
    volume = df_ohlcv["Volume"]
    code = str(snapshot_row.get("code", "")).zfill(6)

    factors = {}

    # Value
    factors.update(compute_value_factors(code, snapshot_row))

    # Momentum
    factors.update(compute_momentum_factors(close))

    # Quality
    factors.update(compute_quality_factors(code))

    # Volatility
    factors.update(compute_volatility_factors(close))

    # Size
    factors.update(compute_size_factors(snapshot_row))

    # North-bound
    factors.update(compute_north_bound_factors(code))

    # Sector momentum
    factors.update(compute_sector_momentum_factors(code))

    # Technical
    factors.update(compute_technical_factors(close, high, low, volume))

    # Margin
    factors.update(compute_margin_factors(code))

    # Fund flow
    factors.update(compute_fund_flow_factors(code))

    # Filter out None values
    return {k: v for k, v in factors.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}


# -- Factor group implementations --

def compute_value_factors(code: str, snapshot) -> dict:
    """PE (inverse), PB (inverse), PS (from turnover)."""
    pe = _safe_float(snapshot.get("pe_ttm"))
    pb = _safe_float(snapshot.get("pb"))
    circ_mv = _safe_float(snapshot.get("circ_mv"))
    turnover = _safe_float(snapshot.get("turnover"))

    factors = {}
    # E/P ratio (inverse PE -- higher is cheaper, handles negative PE gracefully)
    if pe is not None and pe > 0:
        factors["ep_ratio"] = round(1.0 / pe, 6)
    elif pe is not None and pe <= 0:
        factors["ep_ratio"] = 0.0  # negative earnings -> expensive

    # B/P ratio (inverse PB)
    if pb is not None and pb > 0:
        factors["bp_ratio"] = round(1.0 / pb, 6)

    # PS (price-to-sales) proxy: market_cap / turnover_ratio
    # Rough: turnover gives us a sense of revenue scale
    if circ_mv is not None and turnover is not None and turnover > 0:
        factors["ps_proxy"] = round(circ_mv / turnover, 2)

    return factors


def compute_momentum_factors(close: pd.Series) -> dict:
    """1-month (skip 5d), 3-month, 6-month, 12-month momentum."""
    factors = {}
    price = close.iloc[-1]

    for period, label in [(21, "mom_1m"), (63, "mom_3m"), (126, "mom_6m"), (252, "mom_12m")]:
        if len(close) > period:
            factors[label] = round(float(price / close.iloc[-period - 1] - 1), 6)

    # Short-term reversal: skip last 5 days
    if len(close) > 26:
        factors["mom_1m_ex5d"] = round(float(close.iloc[-6] / close.iloc[-26] - 1), 6)

    return factors


def compute_quality_factors(code: str) -> dict:
    """ROE, gross margin, debt ratio from financial reports."""
    try:
        from .data import fetch_financial_indicators
        fin = fetch_financial_indicators(code)
        if fin is None:
            return {}
        factors = {}
        for key in ["roe", "gross_margin", "debt_ratio", "eps"]:
            if key in fin and fin[key] is not None:
                factors[key] = round(float(fin[key]) / 100.0, 6) if key in ("roe", "gross_margin", "debt_ratio") else float(fin[key])
        factors["fin_data_date"] = fin.get("data_date", "unknown")
        return factors
    except Exception:
        return {}


def compute_volatility_factors(close: pd.Series) -> dict:
    """20D vol, 60D vol, Amihud illiquidity, Beta proxy."""
    factors = {}
    rets = close.pct_change(fill_method=None).dropna()

    if len(rets) >= 20:
        vol_20 = float(rets.tail(20).std() * np.sqrt(252))
        factors["vol_20d"] = round(vol_20, 6)

    if len(rets) >= 60:
        vol_60 = float(rets.tail(60).std() * np.sqrt(252))
        factors["vol_60d"] = round(vol_60, 6)
        # Volatility ratio (20d/60d -- regime change signal)
        if len(rets) >= 20:
            factors["vol_ratio"] = round(vol_20 / vol_60, 4) if vol_60 > 0 else 1.0

    return factors


def compute_size_factors(snapshot) -> dict:
    """Log market cap (small cap premium in A-shares)."""
    factors = {}
    circ_mv = _safe_float(snapshot.get("circ_mv"))
    total_mv = _safe_float(snapshot.get("total_mv"))

    if circ_mv is not None and circ_mv > 0:
        factors["ln_circ_mv"] = round(np.log(circ_mv), 4)
    if total_mv is not None and total_mv > 0:
        factors["ln_total_mv"] = round(np.log(total_mv), 4)

    return factors


def compute_north_bound_factors(code: str) -> dict:
    """North-bound holding %, 5D/20D net inflow."""
    try:
        from .data import fetch_north_bound_individual
        hk = fetch_north_bound_individual(code)
        if hk is None:
            return {}
        factors = {}
        for key in ["hold_pct", "hold_shares", "hold_mv"]:
            if key in hk and hk[key] is not None:
                factors[f"north_{key}"] = float(hk[key])
        return factors
    except Exception:
        return {}


def compute_sector_momentum_factors(code: str) -> dict:
    """Industry group momentum -- simplified: use board index returns."""
    # Full implementation requires sector index OHLCV data
    # For now, return empty -- populated when scheduler runs cross-section
    return {}


def compute_technical_factors(close: pd.Series, high: pd.Series,
                              low: pd.Series, volume: pd.Series) -> dict:
    """RSI, MACD, Bollinger, MA slope, volume -- reusing old strategy logic."""
    from .indicators import rsi, macd, bollinger_bands, sma

    factors = {}
    n = len(close)
    if n < 20:
        return factors

    # RSI(14) -- deviation from 50 (neutral)
    r = rsi(close, 14)
    if r.dropna().iloc[-1:].notna().any():
        factors["rsi_deviation"] = round(float(r.dropna().iloc[-1] - 50) / 50, 4)

    # MACD histogram signal
    m = macd(close)
    if len(m) > 0:
        factors["macd_hist"] = round(float(m["histogram"].iloc[-1]) / float(close.iloc[-1]), 6)
        # MACD crossover: is MACD above signal?
        factors["macd_above_signal"] = 1.0 if m["macd"].iloc[-1] > m["signal"].iloc[-1] else -1.0

    # Bollinger Band position (%B)
    bb = bollinger_bands(close, 20, 2)
    if bb["pct_b"].dropna().iloc[-1:].notna().any():
        factors["bb_position"] = round(float(bb["pct_b"].dropna().iloc[-1]), 4)
        factors["bb_width"] = round(float(bb["width"].dropna().iloc[-1]) * 100, 4)

    # MA slope (20-day)
    ma20 = sma(close, 20)
    if len(ma20.dropna()) >= 2:
        factors["ma20_slope"] = round(float(ma20.dropna().iloc[-1] / ma20.dropna().iloc[-2] - 1) * 100, 4)

    # Volume ratio (5-day avg vs 20-day avg)
    if n >= 20:
        vol_5 = volume.tail(5).mean()
        vol_20 = volume.tail(20).mean()
        factors["volume_ratio"] = round(float(vol_5 / vol_20), 4) if vol_20 > 0 else 1.0

    return factors


def compute_sentiment_factors(code: str, ai_client=None) -> dict:
    """News sentiment via AI or keyword fallback."""
    try:
        from news import fetch as fetch_news, sentiment as news_sentiment
        news_list = fetch_news(code, max_items=5)
        if not news_list:
            return {}
        result = news_sentiment(news_list, ai_client)
        if "score" in result:
            return {"news_sentiment": result["score"], "sentiment_confidence": result.get("confidence", 0)}
        return {}
    except Exception:
        return {}


def compute_margin_factors(code: str) -> dict:
    """Margin balance change rate, margin buy ratio."""
    try:
        from .data import fetch_margin_data
        m = fetch_margin_data(code)
        if m is None:
            return {}
        factors = {}
        if "margin_balance" in m and m["margin_balance"] is not None:
            factors["margin_balance"] = float(m["margin_balance"])
        if "margin_buy" in m and m["margin_buy"] is not None:
            factors["margin_buy"] = float(m["margin_buy"])
        return factors
    except Exception:
        return {}


def compute_fund_flow_factors(code: str) -> dict:
    """Main net inflow ratio, super-large order net inflow ratio."""
    try:
        from .data import fetch_fund_flow
        ff = fetch_fund_flow(code)
        if ff is None:
            return {}
        factors = {}
        for key in ["main_net_inflow", "super_large_net_inflow"]:
            if key in ff and ff[key] is not None:
                factors[key] = float(ff[key])
        return factors
    except Exception:
        return {}


# -- Helpers --

def _safe_float(val) -> float | None:
    """Convert to float, return None on failure."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if f == f else None  # NaN check
    except (ValueError, TypeError):
        return None
