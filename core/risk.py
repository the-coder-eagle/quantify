"""Risk management — position sizing + stops."""
import numpy as np, pandas as pd

# A-share standard lot size (1手 = 100股)
A_SHARE_LOT = 100


def position_size(equity, price, risk_pct=0.02, stop_pct=0.05, lot=A_SHARE_LOT):
    """Fixed fractional: risk `risk_pct` of equity per trade."""
    if price <= 0 or stop_pct <= 0: return 0
    value = equity * risk_pct / stop_pct
    shares = int(value / price)
    return (shares // lot) * lot

def kelly_fraction(win_prob, avg_win, avg_loss, fraction=0.5):
    """Half-Kelly criterion."""
    if avg_win <= 0 or avg_loss <= 0 or win_prob <= 0: return 0.0
    q = 1 - win_prob
    f = (win_prob / avg_loss) - (q / avg_win)
    return max(0.0, min(f * fraction, 0.25))

def volatility_target_shares(equity, price, target_vol=0.20, recent_vol=0.30, lot=A_SHARE_LOT):
    """Size inversely to recent volatility."""
    if recent_vol <= 0 or price <= 0: return 0
    scalar = min(target_vol / recent_vol, 1.0)
    shares = int(equity * scalar / price)
    return (shares // lot) * lot

def stop_loss_price(entry, atr_val, multiplier=2.0, direction=1):
    """ATR-based stop. direction: 1=long, -1=short."""
    return round(entry - direction * multiplier * atr_val, 2)

# Risk guard — checks multiple stop conditions
def check_stops(entry, current, highest, atr_val, holding_days,
                stop_pct=0.05, trail_pct=0.03, max_days=30, atr_mult=2.0):
    """Check all stop conditions. Returns (exit: bool, reason: str)."""
    # Fixed stop
    if (current - entry) / entry <= -stop_pct:
        return True, f"固定止损触发 ({stop_pct*100:.0f}%)"
    # Trailing stop
    if highest > entry:
        trail = highest * (1 - trail_pct)
        if current <= trail:
            return True, f"追踪止损触发"
    # ATR stop
    stop_atr = entry - atr_mult * atr_val
    if current <= stop_atr:
        return True, f"ATR止损触发"
    # Time stop
    if holding_days >= max_days:
        return True, f"时间止损 ({max_days}天)"
    return False, ""
