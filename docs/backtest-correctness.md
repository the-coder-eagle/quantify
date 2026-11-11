# Backtest and scoring conventions

- Trade statistics group consecutive nonzero positions. A trade's `entry` and
  `exit` are its first and last held bars, inclusive; `days` is the number of
  held bars. A reversal begins a new trade on that bar. Trade returns are gross
  of costs, while the equity curve and strategy returns include configured costs.
- Walk-forward evaluation includes every complete test window, including one
  ending exactly at the final observation. Net returns compound across windows.
  Each window still starts flat and initializes indicators independently; this
  is not a continuous-position simulation with training-period warm-up.
- Optimizer drawdowns use the signed convention (zero or negative). A deeper
  drawdown lowers the score when other metrics are equal.
- `EnsembleWeight.update()` consumes daily returns. `update_from_sharpes()`
  consumes already-computed Sharpe estimates. The scheduler uses the latter
  for average best-candidate scan Sharpes. These are **in-sample** estimates,
  not independent out-of-sample evidence of profitability.
- IC magnitude controls factor weight, while negative IC reverses factor
  direction. Zero IC retains the raw factor direction with the existing weight
  floor. Constant factor/return cross-sections have zero IC rather than NaN.

Run deterministic regression coverage with:

```sh
python -m pytest tests/test_regressions.py -q
```

Run the full suite with `python -m pytest tests/ -v`. Some existing factor tests
call external market/news services and can take longer or depend on availability.
