"""Backtest engine + metrics + walk-forward validation."""
from dataclasses import dataclass
import pandas as pd, numpy as np
from loguru import logger
from .strategies import Strategy

TRADING_DAYS = 252

# Load cost config from unified config
from config import get as cfg_get
_DEFAULT_MIN_COMMISSION = cfg_get("costs", "min_commission", default=5.0)

@dataclass
class BacktestResult:
    equity: pd.Series; returns: pd.Series; positions: pd.Series
    benchmark_equity: pd.Series; benchmark_returns: pd.Series
    trades: pd.DataFrame; total_cost: float

    def summary(self, rf=0.03):
        s = _metrics(self.equity, self.returns, self.benchmark_returns, rf)
        trades = _extract_trades(self.positions, self.benchmark_returns)
        tr = trades["return"] if not trades.empty else pd.Series(dtype=float)
        s["Total Return"] = f"{_total_return(self.equity)*100:.2f}%"
        s["Benchmark"] = f"{_total_return(self.benchmark_equity)*100:.2f}%"
        s["CAGR"] = f"{_annual_return(self.equity)*100:.2f}%"
        s["Ann Vol"] = f"{_annual_vol(self.returns)*100:.2f}%"
        s["Sharpe"] = round(s["sharpe"], 3); del s["sharpe"]
        s["Max DD"] = f"{_max_dd(self.equity)*100:.2f}%"
        s["Calmar"] = round(s["calmar"], 3); del s["calmar"]
        s["Win Rate"] = f"{_win_rate(tr)*100:.1f}%" if len(tr)>0 else "N/A"
        s["Profit Factor"] = round(_profit_factor(tr), 2) if len(tr)>0 else "N/A"
        s["Trades"] = len(tr); s["Cost"] = f"{self.total_cost*100:.2f}%"
        return s

class BacktestEngine:
    def __init__(self, strategy=None, commission=0.0003, stamp=0.001, slippage=0.001,
                 long_only=True, min_commission=None, price_limit=0.10):
        self.strategy = strategy; self.commission = commission
        self.stamp = stamp; self.slippage = slippage; self.long_only = long_only
        self.min_commission = min_commission if min_commission is not None else _DEFAULT_MIN_COMMISSION
        self.price_limit = price_limit

    def run(self, df, code="stock"):
        if self.strategy is None: raise ValueError("No strategy set")
        signals = self.strategy.generate_signals(df)
        positions = signals.shift(1).fillna(0).astype(int)
        if self.long_only: positions = positions.clip(lower=0)
        asset_ret = df["Close"].pct_change().fillna(0)
        # Apply daily price limits (涨跌停)
        asset_ret = asset_ret.clip(lower=-self.price_limit, upper=self.price_limit)
        strat_ret = positions * asset_ret; total_cost = 0.0
        changes = positions.diff().fillna(0)
        for i in range(1, len(df)):
            ch = changes.iloc[i]
            if ch == 0: continue
            cost = self.commission + self.slippage + (self.stamp if ch < 0 else 0)
            strat_ret.iloc[i] -= cost * abs(ch); total_cost += cost * abs(ch)
        equity = (1 + strat_ret).cumprod(); equity = equity / equity.iloc[0]
        bench = (1 + asset_ret).cumprod(); bench = bench / bench.iloc[0]
        return BacktestResult(equity=equity, returns=strat_ret, positions=positions,
                               benchmark_equity=bench, benchmark_returns=asset_ret,
                               trades=_extract_trades(positions, asset_ret), total_cost=total_cost)

def walk_forward(strategy_or_cls, df, param_grid=None, train_months=24, test_months=6, **strat_kwargs):
    """Walk-forward validation. Returns out-of-sample equity.

    Args:
        strategy_or_cls: Strategy instance OR class. If instance, reads its
            ``param_grid`` attribute unless overridden by the ``param_grid`` arg.
        df: OHLCV DataFrame.
        param_grid: Optional override dict (e.g. {"fast":[10,20], "slow":[40,50]}).
            Defaults to ``strategy_or_cls.param_grid`` if available.
        train_months: Training window in months.
        test_months: Testing window in months.
        **strat_kwargs: Passed to strategy constructor if ``strategy_or_cls`` is a class.
    """
    # Resolve strategy class + param grid
    if isinstance(strategy_or_cls, type):
        strategy_cls = strategy_or_cls
    else:
        strategy_cls = type(strategy_or_cls)

    if param_grid is None:
        param_grid = getattr(strategy_or_cls, "param_grid", None)
    if not param_grid:
        logger.warning("walk_forward: no param_grid available, returning None")
        return None

    results = []; n = len(df)
    start_idx = train_months * 21  # ~21 trading days/month
    test_len = test_months * 21
    while start_idx + test_len < n:
        train = df.iloc[:start_idx]; test = df.iloc[start_idx:start_idx + test_len]
        best_params = None; best_sharpe = -999
        from itertools import product
        for combo in product(*param_grid.values()):
            params = dict(zip(param_grid.keys(), combo))
            s = strategy_cls(**params, **strat_kwargs)
            r = BacktestEngine(s).run(train)
            sh = _sharpe(r.returns)
            if sh > best_sharpe: best_sharpe = sh; best_params = params
        if best_params:
            s = strategy_cls(**best_params, **strat_kwargs)
            r = BacktestEngine(s).run(test)
            results.append(r)
        start_idx += test_len
    if not results: return None
    combined_eq = pd.concat([r.equity for r in results])
    combined_eq = combined_eq[~combined_eq.index.duplicated(keep="first")]
    return {"equity": combined_eq, "n_windows": len(results)}

# --- Internal metrics ---
def _sharpe(r, rf=0.03): c=r.dropna(); return float((c.mean()-rf/TRADING_DAYS)/c.std()*np.sqrt(TRADING_DAYS)) if len(c)>1 and c.std()>0 else 0.0
def _sortino(r, rf=0.03): c=r.dropna(); d=c[c<0]; return float((c.mean()-rf/TRADING_DAYS)/d.std()*np.sqrt(TRADING_DAYS)) if len(d)>1 and d.std()>0 else 0.0
def _max_dd(eq): rm=eq.expanding().max(); return float(((eq-rm)/rm).min())
def _total_return(eq): return float(eq.iloc[-1]/eq.iloc[0]-1) if len(eq)>1 else 0.0
def _annual_return(eq): tr=_total_return(eq); yrs=len(eq)/TRADING_DAYS; return float((1+tr)**(1/yrs)-1) if yrs>0 and tr>-1 else 0.0
def _annual_vol(r): c=r.dropna(); return float(c.std()*np.sqrt(TRADING_DAYS)) if len(c)>1 else 0.0
def _win_rate(tr): c=tr.dropna(); return float((c>0).sum()/len(c)) if len(c)>0 else 0.0
def _profit_factor(tr): c=tr.dropna(); g=c[c>0].sum(); l=abs(c[c<0].sum()); return float(g/l) if l>0 else (float("inf") if g>0 else 0.0)
def _metrics(eq, ret, bench_ret, rf): return {"sharpe":_sharpe(ret,rf),"sortino":_sortino(ret,rf),"calmar":_annual_return(eq)/abs(_max_dd(eq)) if _max_dd(eq)!=0 else 0}

def _extract_trades(positions, returns):
    trades=[]; i=0
    while i<len(positions):
        if positions.iloc[i]==0: i+=1; continue
        direction=positions.iloc[i]; entry=i
        for j in range(i+1,len(positions)):
            if positions.iloc[j]!=direction: exit=j; break
        else: exit=len(positions)-1
        tr=(1+returns.iloc[entry:exit+1]*direction).prod()-1
        trades.append({"entry":positions.index[entry],"exit":positions.index[exit],"direction":int(direction),"return":tr,"days":exit-entry})
        i=exit+1
    return pd.DataFrame(trades) if trades else pd.DataFrame()

def oos_quality_check(strategy, df, train_ratio=0.7):
    """Split data into in-sample / out-of-sample and evaluate strategy on both.

    Returns a dict with IS Sharpe, OOS Sharpe, degradation, and verdict.
    Verdict: "OK" (OOS profitable), "WARN" (significant degradation), "FAIL" (OOS loses money).

    Use this before trusting a strategy's signal — a large gap between IS and OOS
    Sharpe suggests overfitting.
    """
    split_idx = int(len(df) * train_ratio)
    if split_idx < 60 or len(df) - split_idx < 20:
        return {"is_sharpe": 0, "oos_sharpe": 0, "degradation": 0, "verdict": "SKIP"}

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    engine_is = BacktestEngine(strategy, long_only=True)
    is_result = engine_is.run(train_df)

    engine_oos = BacktestEngine(strategy, long_only=True)
    oos_result = engine_oos.run(test_df)

    is_sharpe = _sharpe(is_result.returns)
    oos_sharpe = _sharpe(oos_result.returns)
    degradation = is_sharpe - oos_sharpe

    # Verdict logic
    if is_sharpe <= 0:
        verdict = "SKIP"  # strategy doesn't even work in-sample
    elif oos_sharpe < -0.5:
        verdict = "FAIL"
    elif degradation > is_sharpe * 0.5 and is_sharpe > 0:
        verdict = "WARN"  # lost >50% of IS Sharpe in OOS
    else:
        verdict = "OK"

    return {
        "is_sharpe": round(is_sharpe, 3),
        "oos_sharpe": round(oos_sharpe, 3),
        "degradation": round(degradation, 3),
        "verdict": verdict,
    }


def drawdown_series(eq):
    rm=eq.expanding().max(); return (eq-rm)/rm
