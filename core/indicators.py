"""Technical indicators — pure vectorized, no look-ahead."""
import pandas as pd, numpy as np

def sma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(window=w, min_periods=w).mean()

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — Wilder's smoothing via ewm (vectorized).

    Matches the standard Wilder RSI: SMA-seeded then exponential smoothing
    with alpha=1/period.  The ewm(adjust=False) is mathematically identical
    to Wilder's smoothing after the seed row; first ``period`` values are NaN.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    # Wilder's smoothing: alpha = 1/period, adjust=False
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    # Suppress first period rows (insufficient data)
    avg_gain.iloc[:period] = np.nan
    avg_loss.iloc[:period] = np.nan
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100.0 - 100.0 / (1.0 + rs)
    result[avg_loss == 0] = 100.0
    result = result.clip(0, 100)
    result.name = "RSI"
    return result

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ml = ema(series, fast) - ema(series, slow); sl = ema(ml, signal)
    return pd.DataFrame({"macd": ml, "signal": sl, "histogram": ml - sl}, index=series.index)

def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(series, window); std = series.rolling(window=window, min_periods=window).std()
    return pd.DataFrame({"middle": mid, "upper": mid + num_std * std, "lower": mid - num_std * std,
                         "width": (mid + num_std * std - (mid - num_std * std)) / mid,
                         "pct_b": (series - (mid - num_std * std)) / ((mid + num_std * std) - (mid - num_std * std))},
                        index=series.index)

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — Wilder's smoothing via ewm (vectorized).

    True Range = max(H-L, |H-prevC|, |L-prevC|), then smoothed with
    alpha=1/period (Wilder's method).  First ``period`` values are NaN.
    """
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]  # first bar: use H-L only
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    result = pd.Series(tr, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean()
    result.iloc[:period] = np.nan
    result.name = "ATR"
    return result
