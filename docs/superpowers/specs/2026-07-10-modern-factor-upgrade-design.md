# Modern Factor Upgrade — 现代A股多因子升级

## Overview

将 Quantify Pro v4 的 5 个基础技术策略（均线交叉/MACD/动量/量确认/反转）全面升级为 10 类 30+ 子因子的现代多因子体系，加入 IC 动态加权、横截面排名、北向资金、融资融券、行业动量等 A 股特有因子。

**约束**：用户资金 ~1200 元，单只持仓（100 股/手），akshare 数据源，T+1 交易。

---

## Factor System: 10 Categories, 30+ Sub-factors

| Factor Group | Sub-factors | Frequency | Source |
|-------------|-------------|:--:|------|
| **Value** | PE_TTM, PB, PS_TTM, Dividend Yield | Daily(price) + Quarterly(financials) | akshare fundamentals |
| **Momentum** | 1M(-1M skip), 3M, 6M, 12M return | Daily | OHLCV |
| **Quality** | ROE, Gross Margin, D/A Ratio, Operating CF/Profit | Quarterly | akshare financials |
| **Volatility** | 20D/60D Vol, Amihud Illiquidity, Beta | Daily | OHLCV |
| **Size** | Float Market Cap, Total Market Cap | Daily | Market data |
| **North-bound** | Holding %, 5D Net Inflow, 20D Net Inflow, 60D Inflow, Holding Market Cap | Daily | akshare hsgt |
| **Sector Momentum** | SW Sector 5D/20D Return, Within-Sector Rank | Daily | akshare board |
| **Technical** | RSI Deviation, MACD Histogram, BB %B, MA Slope, Volume Ratio | Daily | OHLCV (old strategy code reused) |
| **Sentiment** | AI News Sentiment, Analyst Rating Change | On-demand | akshare + DeepSeek |
| **Margin** | Margin Balance Change, Margin Buy Ratio, Short Balance | Daily | akshare margin |

---

## Architecture: Three-Layer Pipeline

### After Market Close (scheduler.py)

```
1. universe.filter()
   ├── Exclude ST/delisted/suspended
   ├── Price range filter (2~12 CNY)
   ├── Liquidity filter (daily turnover > 10M CNY)
   └── Market cap filter (> 1B CNY)

2. factors.compute_all(universe)
   ├── Fundamental factors (PE/PB/ROE/Gross Margin)
   ├── Momentum factors (1M/3M/6M/12M)
   ├── Volatility factors (20D/60D/Amihud)
   ├── North-bound factors (holdings/inflow)
   ├── Margin factors (balance change/buy ratio)
   ├── Sector momentum (sector 5D/20D return)
   └── Technical factors (RSI/MACD/BB/MA/Volume)

3. cross_section.rank(all_factors)
   ├── Sector neutralization (SW sector)
   ├── Z-score normalization
   ├── Rank IC calculation (60-day rolling)
   ├── IC weighted → composite score
   └── Save to data_cache/rankings/{date}.parquet
```

### Intraday (app.py)

```
load_ranking(today)
    ↓
analyze_stock(code)
    ├── Query snapshot → stock percentile in market/sector
    ├── Real-time north-bound data
    ├── 30+ factor weighted → composite score
    ├── Regime adjustment
    ├── Position size + stop loss
    └── Return signal
```

---

## File Changes

| File | Action | Est. Lines |
|------|:--:|:--:|
| `core/factors.py` | **New** | ~500 |
| `core/cross_section.py` | **New** | ~250 |
| `core/universe.py` | **New** | ~120 |
| `core/data.py` | **Extend** | Add north-bound/margin/fundamentals/sector interfaces |
| `core/strategies.py` | **Refactor** | Old classes → technical factor functions, keep ALL_STRATEGIES compat |
| `core/analyzer.py` | **Refactor** | Factor scoring replaces strategy voting, API compatible |
| `core/optimizer.py` | **Extend** | Factor IC weighting replaces ParamScanner |
| `scheduler.py` | **Extend** | Add factor computation + cross-section ranking |
| `app.py` | **Minimal** | Tab 2 voting → factor radar + score |
| `tests/test_strategies.py` | **Rewrite** | → `tests/test_factors.py` |
| `tests/test_analyzer.py` | **Tweak** | Keep API compatible |

---

## Known Risks

| Risk | Mitigation |
|------|-----------|
| akshare API interfaces may be broken | Each data source has fallback; failure returns NaN, doesn't block pipeline |
| Fundamental factors lag 2-5 months | Tag `data_date` + freshness; stale period IC weight halved |
| 1200 CNY = 1 stock, cross-section ranking overkill | Ranking results reused: percentile display informs decisions |
| ~25 tests need rewrite | New tests cover factor computation + IC logic, coverage ≥ current |

---

## Key Design Decisions

1. **Offline compute + online read**: Cross-section ranking runs in scheduler, app loads snapshot — avoids real-time compute on 5000 stocks
2. **IC-weighted factor aggregation**: Daily Rank IC over 60-day rolling window, softmax weights, minimum 5% per factor group
3. **Sector neutralization**: Within SW sector Z-score, avoids sector bias in ranking
4. **Old strategies → Technical factor group**: Code reused, not wasted; interpretability preserved via factor attribution display
5. **API compatibility**: `analyze_stock()` returns same dict shape, `app.py` changes minimal
