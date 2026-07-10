"""Core stock analyzer — shared by app.py and daily_report.py."""
from loguru import logger
from .indicators import sma, rsi, atr, bollinger_bands
from .strategies import ALL_STRATEGIES, ALL_STRATEGY_CLASSES
from .regime import detect_regime, regime_risk_multiplier
from .risk import position_size, stop_loss_price
from .backtest import oos_quality_check


def analyze_stock(df, capital=100_000, risk_pct=0.02,
                  strategy_overrides=None, strategy_weights=None,
                  oos_check=False):
    """Run full analysis on a stock DataFrame.

    Args:
        df: OHLCV DataFrame (min 60 rows).
        capital: Total equity for position sizing.
        risk_pct: Risk per trade (0.02 = 2%).
        strategy_overrides: Optional dict of strategy_name → custom param dict.
            e.g. {"趋势-均线": {"fast": 15, "slow": 45}}
        strategy_weights: Optional dict of strategy_name → weight (sum should be 1.0).
            e.g. {"趋势-均线": 0.3, "趋势-MACD": 0.25, ...}
            When provided, votes are weighted instead of equal-weighted.
        oos_check: If True, run out-of-sample quality check on each strategy.

    Returns dict with signal, confidence, strategy votes, regime info,
    position sizing, and indicator values.  Returns None if data insufficient.
    """
    if len(df) < 60:
        return None

    c = df["Close"]
    if c.isna().all() or c.dropna().empty:
        return None
    price = c.iloc[-1]

    # ── Build strategy instances (with optional overrides) ──
    strategies_to_use = []
    for name, strat_cls in ALL_STRATEGY_CLASSES:
        if strategy_overrides and name in strategy_overrides:
            strat = strat_cls(**strategy_overrides[name])
        else:
            # Find default instance
            strat = next((s for n, s in ALL_STRATEGIES if n == name), strat_cls())
        strategies_to_use.append((name, strat))

    # ── Weights (default: equal) ──
    if not strategy_weights:
        n_strat = len(strategies_to_use)
        strategy_weights = {name: 1.0 / n_strat for name, _ in strategies_to_use}

    # ── Strategy votes ──
    buys = sells = 0
    weighted_buys = 0.0
    weighted_sells = 0.0
    details = {}
    oos_results = {}
    for name, strat in strategies_to_use:
        try:
            s = strat.generate_signals(df).iloc[-1]
            details[name] = s
            w = strategy_weights.get(name, 0.0)
            if s == 1:
                buys += 1
                weighted_buys += w
            elif s == -1:
                sells += 1
                weighted_sells += w

            if oos_check:
                oos = oos_quality_check(strat, df)
                oos_results[name] = oos
        except Exception as e:
            logger.error(f"Strategy '{name}' failed: {e}")

    total_weight = weighted_buys + weighted_sells
    conf = weighted_buys / total_weight if total_weight > 0 else 0
    if weighted_buys > weighted_sells and buys > 0:
        sig = "BUY"
    elif weighted_sells > weighted_buys:
        sig = "SELL"
    else:
        sig = "HOLD"

    # ── Regime ──
    reg = detect_regime(df)

    # ── Indicators ──
    rsi_val = rsi(c, 14).iloc[-1]
    a = atr(df, 14).iloc[-1] if len(df) >= 14 else 0.0
    ma20 = sma(c, 20).iloc[-1] if len(c) >= 20 else 0.0
    ma50 = sma(c, 50).iloc[-1] if len(c) >= 50 else 0.0
    bb = bollinger_bands(c, 20, 2)
    bb_pct = bb["pct_b"].iloc[-1]

    # ── Position sizing ──
    shares = 0
    pos_value = 0.0
    stop = 0.0
    if sig == "BUY" and price > 0:
        adj_risk = risk_pct * regime_risk_multiplier(reg)
        shares = position_size(capital, price, adj_risk, 0.05)
        pos_value = shares * price
        stop = stop_loss_price(price, a, 2.0) if a > 0 else round(price * 0.95, 2)

    result = {
        "code": "",
        "price": price,
        "signal": sig,
        "confidence": conf,
        "rsi": round(rsi_val, 1),
        "atr": round(a, 2) if a > 0 else 0,
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2),
        "bb_pct": round(bb_pct, 3),
        "regime": reg,
        "shares": shares,
        "pos_value": round(pos_value, 0),
        "stop": stop,
        "stop_loss": stop,  # alias for daily_report.py compatibility
        "buys": buys,
        "sells": sells,
        "details": details,
    }
    if oos_check:
        result["oos"] = oos_results
    return result
