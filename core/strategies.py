"""Strategies — trend, momentum, reversion, volume. All in one file."""
from abc import ABC, abstractmethod
import pandas as pd, numpy as np
from .indicators import sma, ema, rsi, macd, bollinger_bands, atr

class Strategy(ABC):
    param_grid: dict = {}  # subclasses override with their search space

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series: ...

    def get_position(self, df: pd.DataFrame) -> pd.Series:
        return self.generate_signals(df).shift(1).fillna(0).astype(int)

# --- Trend ---
class MATrend(Strategy):
    param_grid = {"fast": [10, 20, 30], "slow": [40, 50, 60], "ma_type": ["sma", "ema"]}

    def __init__(self, fast=20, slow=50, ma_type="sma"):
        self.fast, self.slow, self.ma_type = fast, slow, ma_type
    def generate_signals(self, df):
        c = df["Close"]; f = sma(c, self.fast) if self.ma_type=="sma" else ema(c, self.fast)
        s = sma(c, self.slow) if self.ma_type=="sma" else ema(c, self.slow)
        sig = pd.Series(0, index=df.index, dtype=int); sig[f > s] = 1; sig[f < s] = -1
        sig.iloc[:self.slow] = 0; return sig

class MACDTrend(Strategy):
    param_grid = {"fast": [8, 12, 16], "slow": [21, 26, 31], "signal": [7, 9, 11]}

    def __init__(self, fast=12, slow=26, signal=9):
        self.fast, self.slow, self.signal = fast, slow, signal
    def generate_signals(self, df):
        m = macd(df["Close"], self.fast, self.slow, self.signal)
        ml, sl = m["macd"], m["signal"]
        sig = pd.Series(0, index=df.index, dtype=int)
        # Vectorized crossover detection
        cross_up = (ml > sl) & (ml.shift(1) <= sl.shift(1))
        cross_down = (ml < sl) & (ml.shift(1) >= sl.shift(1))
        sig[cross_up] = 1
        sig[cross_down] = -1
        sig.iloc[:self.slow] = 0  # warmup
        return sig

# --- Momentum ---
class Momentum(Strategy):
    param_grid = {"lookback": [30, 60, 90], "threshold": [0.0, 0.02]}

    def __init__(self, lookback=60, threshold=0.0):
        self.lookback, self.threshold = lookback, threshold
    def generate_signals(self, df):
        c = df["Close"]; mom = c.pct_change(periods=self.lookback, fill_method=None)
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[mom > self.threshold] = 1; sig[mom < -self.threshold] = -1
        sig.iloc[:self.lookback] = 0; return sig

class VolumeConfirm(Strategy):
    param_grid = {"fast": [10, 20], "slow": [40, 50], "vol_mult": [1.0, 1.2, 1.5]}

    def __init__(self, fast=20, slow=50, vol_window=20, vol_mult=1.2):
        self.fast, self.slow, self.vol_window, self.vol_mult = fast, slow, vol_window, vol_mult
    def generate_signals(self, df):
        c, v = df["Close"], df["Volume"]
        fm = sma(c, self.fast)
        sm = sma(c, self.slow)
        vm = sma(v, self.vol_window)
        base = pd.Series(0, index=df.index, dtype=int)
        base[fm > sm] = 1
        base[fm < sm] = -1
        base.iloc[:self.slow] = 0
        confirmed = v > (vm * self.vol_mult)
        # Where position changes but volume does NOT confirm, NaN-out and forward-fill
        changed = base.diff().fillna(0) != 0
        sig = base.astype(float)
        sig.loc[changed & ~confirmed] = np.nan
        sig = sig.ffill().fillna(0).astype(int)
        return sig

# --- Reversion ---
class MeanReversion(Strategy):
    param_grid = {"bb_window": [15, 20, 25], "oversold": [25, 30, 35], "overbought": [65, 70, 75]}

    def __init__(self, bb_window=20, bb_std=2.0, rsi_period=14, oversold=30.0, overbought=70.0):
        self.bb_window, self.bb_std, self.rsi_period = bb_window, bb_std, rsi_period
        self.oversold, self.overbought = oversold, overbought
    def generate_signals(self, df):
        c = df["Close"]; bb = bollinger_bands(c, self.bb_window, self.bb_std); r = rsi(c, self.rsi_period)
        sig = pd.Series(0, index=df.index, dtype=int)
        sig[(c<bb["lower"])&(r<self.oversold)]=1; sig[(c>bb["upper"])&(r>self.overbought)]=-1
        sig.iloc[:max(self.bb_window,self.rsi_period)]=0; return sig

# --- All strategies for voting ---
ALL_STRATEGIES = [
    ("趋势-均线", MATrend(fast=20, slow=50)),
    ("趋势-MACD", MACDTrend(fast=12, slow=26, signal=9)),
    ("动量", Momentum(lookback=60)),
    ("量确认", VolumeConfirm(fast=20, slow=50)),
    ("反转", MeanReversion(bb_window=20, rsi_period=14)),
]

# Mapping: strategy name → class (for constructing custom-param instances)
ALL_STRATEGY_CLASSES = [
    ("趋势-均线", MATrend),
    ("趋势-MACD", MACDTrend),
    ("动量", Momentum),
    ("量确认", VolumeConfirm),
    ("反转", MeanReversion),
]
