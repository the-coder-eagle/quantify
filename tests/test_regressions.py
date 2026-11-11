"""Deterministic regressions for backtests, optimizer inputs and IC direction."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.backtest import BacktestEngine, _extract_trades, walk_forward
from core.cross_section import compute_rank_ic, ic_weighted_score, rank_all_stocks
from core.optimizer import EnsembleWeight, ParamScanner


@pytest.mark.parametrize(
    'positions,returns,expected,days',
    [
        ([0, 1, 1, 0], [0, .1, .2, -.5], [.32], [2]),
        ([1, -1, -1, 0], [.1, -.2, -.1, .5], [.1, .32], [1, 2]),
        ([0, 0, 1], [0, 0, .2], [.2], [1]),
        ([1, 0, 1], [.1, -.5, .2], [.1, .2], [1, 1]),
    ],
)
def test_trade_runs_do_not_include_exit_or_skip_reversal(positions, returns, expected, days):
    trades = _extract_trades(pd.Series(positions), pd.Series(returns))
    assert trades['return'].tolist() == pytest.approx(expected)
    assert trades['days'].tolist() == days
    assert (trades['exit'] - trades['entry'] + 1).tolist() == days


def test_no_positions_no_trades():
    assert _extract_trades(pd.Series([0, 0]), pd.Series([0, .1])).empty


class AlwaysLong:
    param_grid = {'signal': [1]}

    def __init__(self, signal=1):
        self.signal = signal

    def generate_signals(self, df):
        return pd.Series(self.signal, index=df.index)


@pytest.mark.parametrize('rows', [63, 64])
def test_walk_forward_compounds_windows_and_includes_exact_final_window(rows):
    df = pd.DataFrame({'Close': 100 * 1.01 ** np.arange(rows)},
                      index=pd.bdate_range('2025-01-01', periods=rows))
    result = walk_forward(AlwaysLong, df, train_months=1, test_months=1)
    windows = [BacktestEngine(AlwaysLong()).run(df.iloc[a:a + 21]) for a in (21, 42)]
    expected = (1 + pd.concat([w.returns for w in windows])).cumprod()
    assert result['n_windows'] == 2
    pd.testing.assert_series_equal(result['equity'], expected)
    assert result['equity'].iloc[21] == pytest.approx(result['equity'].iloc[20])
    assert result['equity'].iloc[-1] == pytest.approx(windows[0].equity.iloc[-1] ** 2)


@pytest.mark.parametrize('train,test', [(0, 1), (1, 0), (-1, 1)])
def test_walk_forward_rejects_nonpositive_windows(train, test):
    with pytest.raises(ValueError, match='must be positive'):
        walk_forward(AlwaysLong, pd.DataFrame(), train_months=train, test_months=test)


def test_scanner_penalizes_drawdown_in_actual_scan(monkeypatch):
    class Candidate:
        def __init__(self, dd):
            self.dd = dd

    def run(engine, df):
        return SimpleNamespace(summary=lambda: {
            'Sharpe': 1.0, 'Max DD': f'{engine.strategy.dd}%',
            'Win Rate': '50%', 'Profit Factor': 2, 'Trades': 2,
        })

    monkeypatch.setattr(BacktestEngine, 'run', run)
    results = ParamScanner().scan_single(None, 'candidate', Candidate, {'dd': [-40, -10]})
    assert results[0]['params'] == {'dd': -10}
    assert results[0]['max_dd'] == -.1
    assert results[0]['score'] > results[1]['score']


def test_precomputed_sharpes_produce_distinct_weights():
    weights = EnsembleWeight().update_from_sharpes({'good': 2., 'bad': -1.})
    assert weights['good'] > weights['bad']
    assert sum(weights.values()) == pytest.approx(1.)


def test_daily_returns_still_use_rolling_sharpe():
    perf = {'a': [.01, -.01, .02, .01, .03], 'b': [-.01, .01, -.02, -.01, -.03]}
    ew = EnsembleWeight()
    expected = ew.update_from_sharpes({k: ew._rolling_sharpe(v) for k, v in perf.items()})
    assert ew.update(perf) == expected


def test_scheduler_passes_sharpes_not_single_return_samples(monkeypatch):
    import scheduler
    monkeypatch.setattr(scheduler.TradingCalendar, 'should_run', lambda *a, **kw: True)
    monkeypatch.setattr(scheduler, 'market_snapshot', lambda **kw: pd.DataFrame({'code': ['1']}))
    monkeypatch.setattr(scheduler.DataManager, 'get', lambda *a: pd.DataFrame({'Close': np.ones(60)}))
    monkeypatch.setattr(scheduler, 'ALL_STRATEGY_CLASSES', [('good', None), ('bad', None)])
    monkeypatch.setattr(scheduler.ParamScanner, 'scan_all', lambda *a: {
        '000001': {'strategies': {'good': [{'sharpe': 2}], 'bad': [{'sharpe': -1}]}}
    })
    monkeypatch.setattr(scheduler.ParamScanner, 'save', lambda *a: None)
    captured = {}
    monkeypatch.setattr(scheduler.EnsembleWeight, 'save', lambda self: captured.update(self.get_weights()))
    monkeypatch.setattr(scheduler, 'run_factor_pipeline', lambda **kw: None)
    assert scheduler.run_optimization(force=True)
    assert captured['good'] > captured['bad']


def test_negative_ic_reverses_direction():
    assert ic_weighted_score({'f': 2.}, {'f': -.1}) == -2.
    assert ic_weighted_score({'f': -2.}, {'f': -.1}) == 2.
    stocks = [{'code': str(i), 'f': float(i)} for i in range(10)]
    assert rank_all_stocks(stocks, {'f': -.1}).iloc[0]['code'] == '0'


@pytest.mark.parametrize('constant_factor', [True, False])
def test_constant_values_have_zero_ic(constant_factor):
    constant, varying = pd.Series(np.ones(10)), pd.Series(np.arange(10))
    args = (constant, varying) if constant_factor else (varying, constant)
    assert compute_rank_ic(*args) == 0.
