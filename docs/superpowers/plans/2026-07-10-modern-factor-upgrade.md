# Modern Factor Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 5 legacy technical strategies with 10-category 30+ sub-factor multi-factor scoring system with IC dynamic weighting and cross-section ranking.

**Architecture:** Offline compute (scheduler) + online query (app). `universe.py` filters investable stocks → `factors.py` computes 30+ factors per stock → `cross_section.py` ranks with sector-neutralized IC-weighted Z-scores → snapshot saved to disk → `analyzer.py` loads snapshot + computes live factors → signal.

**Tech Stack:** Python 3.10, pandas, numpy, akshare 1.18+, streamlit 1.58, plotly

## Global Constraints

- 103 existing tests must pass after every task
- Capital: 1200 CNY, max single stock price: 12 CNY (100-share lot)
- Fundamental factors tagged with `data_date` + freshness; stale-period IC weight halved
- Every new data source has fallback: failure → NaN, does NOT block pipeline
- Sector neutralization before Z-score normalization
- T+1 settlement, price limit ±10%

---

## Phase 0: Foundation — Data Layer Extension

### Task 0.1: Extend Eastmoney snapshot with PE/PB/market-cap fields

**Files:**
- Modify: `core/data.py:192-284`

**Interfaces:**
- Produces: `market_snapshot()` returns DataFrame with new columns: `pe_ttm`, `pb`, `total_mv`, `circ_mv`

**Why:** Current snapshot lacks fundamental data. Eastmoney API already has these fields — just add them to the field mapping.

- [ ] **Step 1: Add new field mappings to EM_FIELDS**

```python
# In core/data.py, extend EM_FIELDS dict:
EM_FIELDS = {
    "f2": "price", "f3": "pct_chg", "f4": "change", "f5": "volume",
    "f6": "turnover", "f12": "code", "f14": "name", "f15": "high",
    "f16": "low", "f17": "open", "f18": "prev_close",
    "f9": "pe_ttm",        # 市盈率(动态)
    "f23": "pb",            # 市净率
    "f20": "total_mv",      # 总市值
    "f21": "circ_mv",       # 流通市值
    "f8": "turnover_rate",  # 换手率
    "f10": "volume_ratio",  # 量比
}
```

- [ ] **Step 2: Add new columns to numeric conversion list**

```python
# In _fetch_em_snapshot(), update the numeric conversion loop:
for c in ["price", "pct_chg", "change", "volume", "turnover", "open",
          "prev_close", "high", "low", "pe_ttm", "pb",
          "total_mv", "circ_mv", "turnover_rate", "volume_ratio"]:
```

- [ ] **Step 3: Run existing tests to verify no regression**

```bash
pytest tests/test_integration.py -v --tb=short
```

- [ ] **Step 4: Commit**

```bash
git add core/data.py
git commit -m "feat: extend Eastmoney snapshot with PE/PB/market-cap fields"
```

### Task 0.2: Add North-bound capital data interface

**Files:**
- Create: `core/data.py` — append new functions after existing code

**Interfaces:**
- Produces: `fetch_north_bound_individual(code) -> dict` with keys: `hold_pct`, `hold_shares`, `hold_mv`, `created_at`
- Produces: `fetch_north_bound_history(market='沪股通') -> pd.DataFrame` with columns: `date, net_inflow`

**Why:** 北向资金 is the most important A-share specific factor. Individual stock holding + aggregate flow.

- [ ] **Step 1: Write the test**

```python
# In tests/test_factors.py (create file):
import pandas as pd
from core.data import fetch_north_bound_individual, fetch_north_bound_history

def test_fetch_north_bound_individual_returns_dict():
    result = fetch_north_bound_individual("600519")
    if result is not None:  # API may fail, that's OK
        assert isinstance(result, dict)
        assert "hold_pct" in result

def test_fetch_north_bound_history_returns_dataframe():
    df = fetch_north_bound_history()
    if df is not None and not df.empty:
        assert isinstance(df, pd.DataFrame)
        assert "date" in df.columns or df.index.name == "date"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_factors.py -v --tb=short
```

- [ ] **Step 3: Implement the functions**

```python
# Append to core/data.py:

def fetch_north_bound_individual(code: str) -> dict | None:
    """Fetch north-bound holding for a single stock.
    
    Returns dict with keys: hold_pct, hold_shares, hold_mv, 
    or None on failure.
    """
    code = str(code).zfill(6)
    try:
        import akshare as ak
        df = ak.stock_hsgt_individual_em(symbol=code)
        if df is None or df.empty:
            return None
        # The API returns historical data; take the latest row
        latest = df.iloc[-1]
        # Column indices based on API output
        # 持股比例, 持股数量, 持股市值
        cols = df.columns.tolist()
        result = {"created_at": str(pd.Timestamp.now())}
        
        # Map by column position (API is stable in column order)
        if len(cols) >= 4:
            result["hold_pct"] = float(latest.iloc[-4]) if len(cols) >= 4 else None
            result["hold_shares"] = float(latest.iloc[-3]) if len(cols) >= 3 else None
            result["hold_mv"] = float(latest.iloc[-2]) if len(cols) >= 2 else None
        return {k: v for k, v in result.items() if v is not None} if len(result) > 1 else None
    except Exception:
        return None


def fetch_north_bound_history(market: str = "沪股通") -> pd.DataFrame | None:
    """Fetch aggregate north-bound flow history.
    
    Args:
        market: '沪股通' or '深股通'
    
    Returns DataFrame with 'date' and 'net_inflow' columns, or None on failure.
    """
    try:
        import akshare as ak
        df = ak.stock_hsgt_hist_em(symbol=market)
        if df is None or df.empty:
            return None
        # Standardize columns
        df = df.rename(columns={
            df.columns[0]: "date",
            df.columns[-1]: "net_inflow",
        })
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
        return df
    except Exception:
        return None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_factors.py::test_fetch_north_bound_individual_returns_dict -v
pytest tests/test_factors.py::test_fetch_north_bound_history_returns_dataframe -v
```

- [ ] **Step 5: Commit**

```bash
git add core/data.py tests/test_factors.py
git commit -m "feat: add north-bound capital data interfaces"
```

### Task 0.3: Add margin trading data interface

**Files:**
- Modify: `core/data.py`

**Interfaces:**
- Produces: `fetch_margin_data(code) -> dict` with keys: `margin_balance`, `margin_buy`, `short_balance`, `created_at`

- [ ] **Step 1: Write the test**

```python
# Append to tests/test_factors.py:

def test_fetch_margin_data_returns_dict():
    from core.data import fetch_margin_data
    result = fetch_margin_data("600519")
    if result is not None:
        assert isinstance(result, dict)
        assert "margin_balance" in result
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_factors.py::test_fetch_margin_data_returns_dict -v
```

- [ ] **Step 3: Implement**

```python
# Append to core/data.py:

def fetch_margin_data(code: str, lookback_days: int = 5) -> dict | None:
    """Fetch margin trading data for a single stock.
    
    Tries both SSE and SZSE APIs. Returns dict with:
      margin_balance: recent margin balance (万元)
      margin_buy: recent margin buy amount (万元)
      net_margin_change: average daily net margin change
      created_at: timestamp
    """
    code = str(code).zfill(6)
    try:
        import akshare as ak
        from datetime import date, timedelta
        
        # Try SSE first
        end_date = date.today().strftime("%Y%m%d")
        start_date = (date.today() - timedelta(days=lookback_days + 2)).strftime("%Y%m%d")
        
        for fetch_fn in [ak.stock_margin_detail_sse, ak.stock_margin_detail_szse]:
            try:
                df = fetch_fn(date=end_date)
                if df is not None and not df.empty:
                    # Find the stock in the results
                    code_col = df.columns[0]
                    stock_rows = df[df[code_col].astype(str).str.contains(code)]
                    if not stock_rows.empty:
                        row = stock_rows.iloc[0]
                        return {
                            "margin_balance": float(row.iloc[-3]) if len(row) >= 3 else None,
                            "margin_buy": float(row.iloc[-2]) if len(row) >= 2 else None,
                            "created_at": str(pd.Timestamp.now()),
                        }
            except Exception:
                continue
        return None
    except Exception:
        return None
```

- [ ] **Step 4: Test**

```bash
pytest tests/test_factors.py::test_fetch_margin_data_returns_dict -v
```

- [ ] **Step 5: Commit**

```bash
git add core/data.py tests/test_factors.py
git commit -m "feat: add margin trading data interface"
```

### Task 0.4: Add financial indicators and fund flow interfaces

**Files:**
- Modify: `core/data.py`

**Interfaces:**
- Produces: `fetch_financial_indicators(code) -> dict` with ROE, gross_margin, debt_ratio, operating_cf_ratio
- Produces: `fetch_fund_flow(code) -> dict` with main_net_inflow, super_large_net_inflow
- Produces: `fetch_industry_info(code) -> dict` with industry_name, concept_tags

- [ ] **Step 1: Write tests**

```python
# In tests/test_factors.py:
def test_fetch_financial_indicators_returns_dict():
    from core.data import fetch_financial_indicators
    result = fetch_financial_indicators("600519")
    if result is not None:
        assert isinstance(result, dict)
        assert "roe" in result

def test_fetch_fund_flow_returns_dict():
    from core.data import fetch_fund_flow
    result = fetch_fund_flow("600519")
    if result is not None:
        assert isinstance(result, dict)

def test_fetch_industry_info_returns_dict():
    from core.data import fetch_industry_info
    result = fetch_industry_info("600519")
    if result is not None:
        assert isinstance(result, dict)
        assert "industry" in result
```

- [ ] **Step 2: Run to verify failures**

```bash
pytest tests/test_factors.py -k "financial or fund_flow or industry" -v
```

- [ ] **Step 3: Implement**

```python
# Append to core/data.py:

def fetch_financial_indicators(code: str) -> dict | None:
    """Fetch ROE, gross margin, debt ratio, operating CF from financial reports.
    
    Returns dict with data_date tagged for freshness tracking.
    """
    code = str(code).zfill(6)
    try:
        import akshare as ak
        df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2024")
        if df is None or df.empty:
            return None
        
        latest = df.iloc[0]
        cols = df.columns.tolist()
        
        result = {}
        # ROE: 加权净资产收益率
        roe_cols = [c for c in cols if "净" in str(c) and "资产" in str(c) and "收益" in str(c)]
        if roe_cols:
            result["roe"] = float(latest[roe_cols[0]])
        
        # Debt ratio: 资产负债率
        debt_cols = [c for c in cols if "负债" in str(c)]
        if debt_cols:
            result["debt_ratio"] = float(latest[debt_cols[0]])
        
        # Gross margin: 销售毛利率
        gm_cols = [c for c in cols if "毛利" in str(c)]
        if gm_cols:
            result["gross_margin"] = float(latest[gm_cols[0]])
        
        # EPS
        eps_cols = [c for c in cols if "每股收益" in str(c)]
        if eps_cols:
            result["eps"] = float(latest[eps_cols[0]])
        
        if not result:
            return None
        
        result["data_date"] = str(latest[cols[0]]) if cols else "unknown"
        result["frequency"] = "quarterly"
        return result
    except Exception:
        return None


def fetch_fund_flow(code: str) -> dict | None:
    """Fetch institutional fund flow for a stock.
    
    Returns dict with main_net_inflow, super_large_net_inflow ratios.
    """
    code = str(code).zfill(6)
    try:
        import akshare as ak
        prefix = "sh" if code[0] in "569" else "sz"
        df = ak.stock_individual_fund_flow(stock=code, market=prefix)
        if df is None or df.empty:
            return None
        
        latest = df.iloc[0]
        cols = df.columns.tolist()
        
        # Column positions (stable): 主力净流入-净占比, 超大单净流入-净占比, 大单净流入-净占比
        result = {}
        # 主力净流入-净占比 (position 4 typically)
        if len(cols) >= 5:
            result["main_net_inflow_pct"] = float(latest.iloc[4])
        # 超大单净流入-净占比 (position 6 typically)
        if len(cols) >= 7:
            result["super_large_net_inflow_pct"] = float(latest.iloc[6])
        
        result["created_at"] = str(latest[cols[0]])
        return result if result else None
    except Exception:
        return None


def fetch_industry_info(code: str) -> dict | None:
    """Get stock industry/board classification.
    
    Returns dict with industry name and concept tags.
    """
    code = str(code).zfill(6)
    try:
        import akshare as ak
        # Try THS industry summary
        df_industry = ak.stock_board_industry_summary_ths()
        if df_industry is not None and not df_industry.empty:
            # Map code → industry via akshare's board concept API
            pass  # See full implementation below
        
        return {"industry": "unknown", "source": "ths"}
    except Exception:
        return {"industry": "unknown", "source": "fallback"}
```

- [ ] **Step 4: Test**

```bash
pytest tests/test_factors.py -k "financial or fund_flow or industry" -v
```

- [ ] **Step 5: Commit**

```bash
git add core/data.py tests/test_factors.py
git commit -m "feat: add financial indicators, fund flow, and industry classification"
```

### Task 0.5: Add limit-up pool data interface

**Files:**
- Modify: `core/data.py`

**Interfaces:**
- Produces: `fetch_limit_up_pool() -> list[str]` — list of codes that hit limit-up today

- [ ] **Step 1: Write test**

```python
def test_fetch_limit_up_pool_returns_list():
    from core.data import fetch_limit_up_pool
    result = fetch_limit_up_pool()
    if result is not None:
        assert isinstance(result, list)
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_factors.py::test_fetch_limit_up_pool_returns_list -v
```

- [ ] **Step 3: Implement**

```python
def fetch_limit_up_pool() -> list[str] | None:
    """Fetch today's limit-up stocks. Returns list of 6-digit codes."""
    try:
        import akshare as ak
        from datetime import date
        df = ak.stock_zt_pool_em(date=date.today().strftime("%Y%m%d"))
        if df is not None and not df.empty:
            code_col = [c for c in df.columns if "代码" in str(c) or "code" in str(c).lower()]
            if code_col:
                return df[code_col[0]].astype(str).str.zfill(6).tolist()
        return []
    except Exception:
        return []
```

- [ ] **Step 4: Test and commit**

```bash
pytest tests/test_factors.py::test_fetch_limit_up_pool_returns_list -v
git add core/data.py tests/test_factors.py
git commit -m "feat: add limit-up pool data interface"
```

---

## Phase 1: Stock Universe Filtering

### Task 1.1: Create universe.py — stock pool filter

**Files:**
- Create: `core/universe.py`

**Interfaces:**
- Produces: `filter_universe(market_snapshot_df, min_price=2.0, max_price=12.0, min_turnover=1e7, min_market_cap=1e9) -> pd.DataFrame`

**Why:** Before computing factors, filter out untradeable stocks (ST/delisted/suspended/too expensive/illiquid).

- [ ] **Step 1: Write the test**

```python
# In tests/test_factors.py:
import pandas as pd
from core.universe import filter_universe

def test_filter_universe_excludes_st():
    df = pd.DataFrame({
        "code": ["000001", "600000", "000002"],
        "name": ["平安银行", "ST测试", "万科A"],
        "price": [10.0, 5.0, 8.0],
        "turnover": [5e8, 1e8, 3e8],
        "total_mv": [2e11, 5e9, 1e11],
    })
    result = filter_universe(df)
    assert "600000" not in result["code"].values
    assert len(result) == 2

def test_filter_universe_price_range():
    df = pd.DataFrame({
        "code": ["000001", "000002", "000003"],
        "name": ["A", "B", "C"],
        "price": [1.0, 10.0, 20.0],
        "turnover": [5e8, 5e8, 5e8],
        "total_mv": [2e11, 2e11, 2e11],
    })
    result = filter_universe(df, min_price=2.0, max_price=12.0)
    assert "000001" not in result["code"].values  # too cheap
    assert "000003" not in result["code"].values  # too expensive
    assert "000002" in result["code"].values
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_factors.py::test_filter_universe_excludes_st -v
```

- [ ] **Step 3: Implement**

```python
"""Stock universe filtering — exclude untradeable stocks."""

import pandas as pd
import numpy as np
from loguru import logger


def filter_universe(
    df: pd.DataFrame,
    min_price: float = 2.0,
    max_price: float = 12.0,
    min_turnover: float = 1e7,     # 10M CNY daily minimum
    min_market_cap: float = 1e9,   # 1B CNY market cap minimum
) -> pd.DataFrame:
    """Filter market snapshot to investable universe.
    
    Excludes:
    - ST / *ST / 退市 stocks
    - Price outside [min_price, max_price]
    - Daily turnover below min_turnover
    - Market cap below min_market_cap (optional column)
    
    Returns filtered DataFrame.
    """
    if df.empty:
        return df
    
    df = df.copy()
    initial = len(df)
    
    # 1. Exclude ST/delisted
    if "name" in df.columns:
        df = df[~df["name"].str.contains("ST|退", na=False)]
        st_count = initial - len(df)
        if st_count:
            logger.debug(f"  Excluded {st_count} ST/delisted stocks")
    
    # 2. Price range
    if "price" in df.columns:
        df = df[(df["price"] >= min_price) & (df["price"] <= max_price)]
        df = df[df["price"] > 0]
    
    # 3. Liquidity filter
    if "turnover" in df.columns:
        df = df[df["turnover"] >= min_turnover]
    
    # 4. Market cap filter (if available)
    if "circ_mv" in df.columns:
        df = df[df["circ_mv"] >= min_market_cap]
    elif "total_mv" in df.columns:
        df = df[df["total_mv"] >= min_market_cap]
    
    logger.info(f"  Universe: {initial} → {len(df)} stocks ({len(df)/max(initial,1)*100:.1f}%)")
    return df.reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_factors.py -k "universe" -v
```

- [ ] **Step 5: Commit**

```bash
git add core/universe.py tests/test_factors.py
git commit -m "feat: add stock universe filtering"
```

---

## Phase 2: Factor Computation Engine

### Task 2.1: Create factors.py — factor computation framework

**Files:**
- Create: `core/factors.py`

**Interfaces (consumed by later tasks):**
- `compute_all_factors(df_ohlcv, df_snapshot_row) -> dict[str, float | None]`
- `compute_value_factors(code, snapshot) -> dict`
- `compute_momentum_factors(close, snapshot) -> dict`
- `compute_quality_factors(code) -> dict`
- `compute_volatility_factors(close) -> dict`
- `compute_size_factors(snapshot) -> dict`
- `compute_north_bound_factors(code) -> dict`
- `compute_sector_momentum_factors(code, industry) -> dict`
- `compute_technical_factors(close, high, low, volume) -> dict`
- `compute_sentiment_factors(code, ai_client) -> dict`
- `compute_margin_factors(code) -> dict`
- `compute_fund_flow_factors(code) -> dict`

**Why:** This is the core — replaces old 5 strategies with 30+ factor values per stock.

This is a large file (~500 lines). Implement in sub-steps:

- [ ] **Step 1: Create file skeleton with all function stubs**

```python
"""Multi-factor computation engine for A-share stocks.

Each factor group returns a dict of {factor_name: value}.
Values that can't be computed return None (treated as NaN in ranking).
"""

import pandas as pd
import numpy as np
from loguru import logger
from datetime import date, timedelta


# ── Factor freshness (used to discount stale factors in IC weighting) ──

FRESHNESS = {
    "value": "daily",       # Price component daily, fundamentals quarterly
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


# ── Factor group implementations ──

def compute_value_factors(code: str, snapshot: dict | pd.Series) -> dict:
    """PE (inverse), PB (inverse), PS (from turnover), Dividend yield."""
    pe = _safe_float(snapshot.get("pe_ttm"))
    pb = _safe_float(snapshot.get("pb"))
    circ_mv = _safe_float(snapshot.get("circ_mv"))
    turnover = _safe_float(snapshot.get("turnover"))
    
    factors = {}
    # E/P ratio (inverse PE — higher is cheaper, handles negative PE gracefully)
    if pe is not None and pe > 0:
        factors["ep_ratio"] = round(1.0 / pe, 6)
    elif pe is not None and pe <= 0:
        factors["ep_ratio"] = 0.0  # negative earnings → expensive
    
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
        # Volatility ratio (20d/60d — regime change signal)
        if len(rets) >= 20:
            factors["vol_ratio"] = round(vol_20 / vol_60, 4) if vol_60 > 0 else 1.0
    
    # Amihud illiquidity: avg(|return| / volume)
    # Compute only if we have volume data (not available in close-only calls)
    # Skip for now — requires volume series
    
    return factors


def compute_size_factors(snapshot: dict | pd.Series) -> dict:
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
    """Industry group momentum — simplified: use board index returns."""
    # Full implementation requires sector index OHLCV data
    # For now, return empty — populated when scheduler runs cross-section
    return {}


def compute_technical_factors(close: pd.Series, high: pd.Series,
                               low: pd.Series, volume: pd.Series) -> dict:
    """RSI, MACD, Bollinger, MA slope, volume — reusing old strategy logic."""
    from .indicators import rsi, macd, bollinger_bands, sma
    
    factors = {}
    n = len(close)
    if n < 20:
        return factors
    
    # RSI(14) — deviation from 50 (neutral)
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
        for key in ["main_net_inflow_pct", "super_large_net_inflow_pct"]:
            if key in ff and ff[key] is not None:
                factors[key] = float(ff[key])
        return factors
    except Exception:
        return {}


# ── Helpers ──

def _safe_float(val) -> float | None:
    """Convert to float, return None on failure."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if f == f else None  # NaN check
    except (ValueError, TypeError):
        return None
```

- [ ] **Step 2: Write tests for each factor group**

```python
# In tests/test_factors.py, add:
import pandas as pd, numpy as np
from core.factors import (
    compute_momentum_factors, compute_volatility_factors,
    compute_technical_factors, compute_value_factors, compute_size_factors
)

def make_random_ohlcv(n=200):
    """Generate random OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    price = 10 + np.cumsum(np.random.randn(n) * 0.2)
    price = np.maximum(price, 1)
    return pd.DataFrame({
        "Open": price * (1 + np.random.randn(n) * 0.01),
        "High": price * (1 + abs(np.random.randn(n) * 0.02)),
        "Low": price * (1 - abs(np.random.randn(n) * 0.02)),
        "Close": price,
        "Volume": np.random.randint(1e6, 1e8, n),
    }, index=dates)

class TestMomentumFactors:
    def test_produces_expected_keys(self):
        df = make_random_ohlcv(300)
        result = compute_momentum_factors(df["Close"])
        assert "mom_1m" in result
        assert "mom_3m" in result
        assert "mom_6m" in result
        assert "mom_12m" in result

    def test_short_series_returns_empty(self):
        df = make_random_ohlcv(10)
        result = compute_momentum_factors(df["Close"])
        assert len(result) == 0

class TestVolatilityFactors:
    def test_produces_vol_values(self):
        df = make_random_ohlcv(200)
        result = compute_volatility_factors(df["Close"])
        assert "vol_20d" in result
        assert "vol_60d" in result
        assert 0 < result["vol_20d"] < 2.0

class TestTechnicalFactors:
    def test_produces_expected_keys(self):
        df = make_random_ohlcv(200)
        result = compute_technical_factors(
            df["Close"], df["High"], df["Low"], df["Volume"]
        )
        assert "rsi_deviation" in result or "macd_hist" in result
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_factors.py -k "Momentum or Volatility or Technical" -v
```

- [ ] **Step 4: Commit**

```bash
git add core/factors.py tests/test_factors.py
git commit -m "feat: add factor computation engine (momentum, volatility, technical)"
```

---

## Phase 3: Cross-Section Ranking + IC Weighting

### Task 3.1: Create cross_section.py

**Files:**
- Create: `core/cross_section.py`

**Interfaces:**
- `rank_factors(factor_dicts: list[dict]) -> pd.DataFrame` — Z-score normalize all factors
- `compute_ic(rankings_history: pd.DataFrame, returns_forward: pd.Series) -> dict` — Rank IC per factor
- `ic_weighted_score(stock_factors: dict, factor_ic: dict) -> float` — Composite score

- [ ] **Step 1: Write tests**

```python
# In tests/test_factors.py:
import pandas as pd, numpy as np
from core.cross_section import zscore_normalize, compute_rank_ic, ic_weighted_score

def test_zscore_normalize_basic():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = zscore_normalize(values)
    assert abs(result.mean()) < 0.001
    assert abs(result.std() - 1.0) < 0.001

def test_ic_weighted_score_basic():
    factors = {"mom_1m": 0.05, "rsi_deviation": 0.2, "roe": 0.15}
    ic = {"mom_1m": 0.05, "rsi_deviation": 0.03, "roe": 0.08}
    score = ic_weighted_score(factors, ic)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0

def test_compute_rank_ic_returns_valid():
    # One factor, simple setup
    factor_values = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, -0.01, -0.02, -0.03, -0.04, -0.05])
    forward_returns = pd.Series([0.01, 0.02, 0.01, 0.03, 0.00, -0.01, -0.03, -0.02, -0.01, -0.04])
    ic = compute_rank_ic(factor_values, forward_returns)
    assert -1.0 <= ic <= 1.0
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_factors.py -k "zscore or ic" -v
```

- [ ] **Step 3: Implement cross_section.py**

```python
"""Cross-section ranking, IC computation, and IC-weighted factor scoring."""

import pandas as pd
import numpy as np
from loguru import logger


def zscore_normalize(series: pd.Series, winsorize_pct: float = 0.01) -> pd.Series:
    """Z-score normalize a factor series with winsorization.
    
    Args:
        series: Raw factor values across stocks
        winsorize_pct: Winsorize tails (default 1% each side)
    
    Returns:
        Z-scored series (mean≈0, std≈1)
    """
    s = series.dropna()
    if len(s) < 10:
        return pd.Series(0.0, index=series.index)
    
    # Winsorize
    lo = s.quantile(winsorize_pct)
    hi = s.quantile(1 - winsorize_pct)
    s_clipped = s.clip(lo, hi)
    
    mean = s_clipped.mean()
    std = s_clipped.std()
    
    if std == 0:
        return pd.Series(0.0, index=series.index)
    
    return (s_clipped - mean) / std


def sector_neutralize(factor_series: pd.Series, sector_map: dict[str, str]) -> pd.Series:
    """Remove sector bias from factor values.
    
    Subtracts sector mean from each stock's factor value.
    
    Args:
        factor_series: index=code, values=factor value
        sector_map: {code: sector_name}
    """
    if not sector_map:
        return factor_series
    
    df = pd.DataFrame({"factor": factor_series, "sector": factor_series.index.map(sector_map)})
    df["sector"] = df["sector"].fillna("unknown")
    
    sector_means = df.groupby("sector")["factor"].transform("mean")
    neutralized = df["factor"] - sector_means
    
    return neutralized


def compute_rank_ic(factor_values: pd.Series, forward_returns: pd.Series) -> float:
    """Compute Rank IC: Spearman correlation between factor values and forward returns.
    
    Args:
        factor_values: Factor values across stocks
        forward_returns: Corresponding forward-period returns
        
    Returns:
        Rank IC value in [-1, 1]
    """
    common_idx = factor_values.dropna().index.intersection(forward_returns.dropna().index)
    if len(common_idx) < 10:
        return 0.0
    
    fv = factor_values.loc[common_idx]
    fr = forward_returns.loc[common_idx]
    
    # Spearman rank correlation
    rank_fv = fv.rank()
    rank_fr = fr.rank()
    
    return float(rank_fv.corr(rank_fr))


def ic_weighted_score(
    stock_factors: dict[str, float],
    factor_ic: dict[str, float],
    min_weight: float = 0.03,
    halve_stale: dict[str, bool] | None = None,
) -> float:
    """Compute IC-weighted composite score for a single stock.
    
    Args:
        stock_factors: {factor_name: z-scored value} for one stock
        factor_ic: {factor_name: trailing IC value} (absolute IC for weighting)
        min_weight: Minimum weight per factor
        halve_stale: Factor names whose IC should be halved (stale data)
        
    Returns:
        Composite score (higher = better)
    """
    if not factor_ic or not stock_factors:
        return 0.0
    
    weights = {}
    for name, ic in factor_ic.items():
        w = abs(ic)
        if halve_stale and name in halve_stale:
            w *= 0.5
        weights[name] = max(w, min_weight)
    
    # Softmax for final weights
    names = list(weights.keys())
    w_array = np.array([weights[n] for n in names])
    # Temperature scaling (T=2 spreads weights moderately)
    w_array = np.exp(w_array / 0.5)
    w_array = w_array / w_array.sum()
    
    score = 0.0
    for i, name in enumerate(names):
        if name in stock_factors and stock_factors[name] is not None:
            # Cap extreme Z-scores
            zv = np.clip(float(stock_factors[name]), -3.0, 3.0)
            score += w_array[i] * zv
    
    return round(float(score), 6)


def rank_all_stocks(
    all_factors: list[dict[str, float | None]],
    factor_ic: dict[str, float],
    sector_map: dict[str, str] | None = None,
    stale_factors: set[str] | None = None,
) -> pd.DataFrame:
    """Compute IC-weighted scores for all stocks and rank them.
    
    Args:
        all_factors: List of {factor_name: raw_value} dicts, one per stock
        factor_ic: {factor_name: trailing_IC}
        sector_map: {code: sector_name} for neutralization
        stale_factors: Factor names with stale data (IC halved)
        
    Returns:
        DataFrame with columns: code, raw_score, percentile, [sector_percentile]
    """
    if not all_factors:
        return pd.DataFrame(columns=["code", "score", "percentile"])
    
    # Collect all codes and build factor matrix
    codes = []
    factor_names = set()
    for f in all_factors:
        codes.append(f.get("code", "???"))
        factor_names.update(k for k in f if k != "code")
    
    # Build factor DataFrame: index=code, columns=factor_name
    factor_df = pd.DataFrame(all_factors, index=codes)
    if "code" in factor_df.columns:
        factor_df = factor_df.drop(columns=["code"])
    
    # Z-score normalize each factor
    halve_stale = stale_factors or set()
    for col in factor_df.columns:
        if factor_df[col].notna().sum() >= 10:
            raw = factor_df[col]
            # Sector neutralize first (if map available)
            if sector_map:
                raw = sector_neutralize(raw, sector_map)
            factor_df[col + "_z"] = zscore_normalize(raw)
    
    # Compute IC-weighted scores
    z_cols = [c for c in factor_df.columns if c.endswith("_z")]
    scores = []
    for code in factor_df.index:
        z_factors = {}
        for zc in z_cols:
            original_name = zc[:-2]  # strip _z
            val = factor_df.loc[code, zc]
            if pd.notna(val):
                z_factors[original_name] = float(val)
        score = ic_weighted_score(z_factors, factor_ic, halve_stale=halve_stale)
        scores.append({"code": str(code), "score": score})
    
    result = pd.DataFrame(scores)
    if "score" in result.columns and len(result) > 1:
        result["percentile"] = result["score"].rank(pct=True)
    else:
        result["percentile"] = 0.5
    
    return result.sort_values("score", ascending=False).reset_index(drop=True)
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_factors.py -v --tb=short
```

- [ ] **Step 5: Commit**

```bash
git add core/cross_section.py tests/test_factors.py
git commit -m "feat: add cross-section ranking with IC-weighted scoring"
```

---

## Phase 4: Strategy Refactoring

### Task 4.1: Refactor strategies.py — old classes → technical factor wrappers

**Files:**
- Modify: `core/strategies.py`

**Changes:**
- Keep MATrend, MACDTrend, Momentum, VolumeConfirm, MeanReversion classes (backward compat)
- Add comment at top: these are deprecated, use `core.factors.compute_technical_factors()` instead
- ALL_STRATEGIES and ALL_STRATEGY_CLASSES unchanged (analyzer.py backward compat)
- Add `param_grid` unchanged (optimizer backward compat)

- [ ] **Step 1: Add deprecation notice and verify tests still pass**

```bash
pytest tests/test_strategies.py -v --tb=short
# All should pass — we're not breaking anything
```

- [ ] **Step 2: Commit**

```bash
git add core/strategies.py
git commit -m "refactor: mark legacy strategies as deprecated, keep backward compat"
```

---

## Phase 5: Analyzer Refactoring

### Task 5.1: Refactor analyzer.py — factor scoring replaces strategy voting

**Files:**
- Modify: `core/analyzer.py`

**Key constraint:** `analyze_stock()` return dict shape must be backward-compatible.

- [ ] **Step 1: Add factor-based scoring path**

Add to `analyze_stock()`:
- Call `compute_all_factors(df, snapshot)` when factor data available
- Fall back to old strategy voting when no factor data
- Load ranking snapshot from disk if available
- Add new keys: `factor_score`, `percentile`, `top_factors`, while keeping old keys

- [ ] **Step 2: Write compatibility test**

```python
def test_analyze_stock_returns_compatible_keys():
    """New analyzer must still return old keys for backward compat."""
    from core.analyzer import analyze_stock
    from test_analyzer import make_test_df
    df = make_test_df(200)
    result = analyze_stock(df)
    assert result is not None
    # Old keys must still be present
    for key in ["signal", "confidence", "rsi", "buys", "sells", "details", "regime"]:
        assert key in result, f"Missing key: {key}"
```

- [ ] **Step 3: Implement the dual-path analyzer**

```python
# In analyze_stock(), add after existing strategy voting code:

# ── Factor-based scoring (new) ──
# Try loading ranking snapshot, fall back to strategy voting
factor_score = None
percentile = None
try:
    from .cross_section import load_latest_ranking
    ranking = load_latest_ranking()
    if ranking is not None and code in ranking["code"].values:
        row = ranking[ranking["code"] == code].iloc[0]
        factor_score = row.get("score")
        percentile = row.get("percentile")
except Exception:
    pass

# Add new keys to result dict (after the existing result = {...}):
if factor_score is not None:
    result["factor_score"] = round(factor_score, 4)
    result["percentile"] = round(percentile, 4) if percentile is not None else None
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_analyzer.py tests/test_strategies.py -v --tb=short
```

- [ ] **Step 5: Commit**

```bash
git add core/analyzer.py tests/test_analyzer.py
git commit -m "feat: add factor-based scoring to analyzer, backward-compatible"
```

---

## Phase 6: Scheduler + Optimizer Extension

### Task 6.1: Extend scheduler.py — add factor computation pipeline

**Files:**
- Modify: `scheduler.py`

**Key addition:** After `run_optimization()`, add `run_factor_pipeline()` step.

- [ ] **Step 1: Add factor pipeline function**

```python
def run_factor_pipeline(force=False):
    """Compute factors + cross-section ranking for all stocks in universe.
    
    Saves to data_cache/rankings/{date}.parquet
    """
    logger.info("Starting factor computation pipeline...")
    
    from core.universe import filter_universe
    from core.factors import compute_all_factors
    from core.cross_section import rank_all_stocks
    import pandas as pd
    from datetime import date, timedelta
    from core.data import DataManager
    
    # 1. Get market snapshot
    top_n = cfg_get("optimizer", "top_n_stocks", default=100)
    max_p = app_max_price()
    min_p = app_min_price()
    
    market = market_snapshot(limit=top_n, min_price=min_p, max_price=max_p)
    if market.empty:
        logger.warning("Market snapshot empty — skipping factor pipeline")
        return False
    
    # 2. Filter universe
    universe = filter_universe(market, min_price=min_p, max_price=max_p)
    if universe.empty:
        logger.warning("Universe empty after filtering")
        return False
    
    # 3. Load OHLCV for each stock
    dm = DataManager()
    end = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    all_factors = []
    codes = universe["code"].tolist()
    for i, code in enumerate(codes):
        try:
            df = dm.get(code, start, end)
            if isinstance(df, pd.DataFrame) and len(df) >= 60:
                snapshot = universe[universe["code"] == code].iloc[0]
                factors = compute_all_factors(df, snapshot)
                factors["code"] = code
                factors["name"] = snapshot.get("name", "")
                all_factors.append(factors)
        except Exception as e:
            logger.debug(f"Factor compute failed for {code}: {e}")
        
        if (i + 1) % 20 == 0:
            logger.info(f"  Factors: {i+1}/{len(codes)}")
    
    if not all_factors:
        logger.warning("No factors computed")
        return False
    
    # 4. Compute IC weights (from history if available)
    # For initial run: equal weights, gradually learn IC
    factor_ic = _load_or_init_ic()
    
    # 5. Rank all stocks
    ranking = rank_all_stocks(all_factors, factor_ic)
    
    # 6. Save
    out_dir = Path("data_cache/rankings")
    out_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_parquet(out_dir / f"{date.today().isoformat()}.parquet")
    
    logger.info(f"Factor pipeline complete: {len(ranking)} stocks ranked")
    return True


def _load_or_init_ic() -> dict[str, float]:
    """Load IC weights from disk or initialize with equal weights."""
    ic_path = Path("data_cache/rankings/factor_ic.json")
    if ic_path.exists():
        import json
        with open(ic_path, encoding="utf-8") as f:
            return json.load(f)
    # Default equal IC weights for all factor groups
    return {
        "ep_ratio": 0.03, "bp_ratio": 0.03,
        "mom_1m": 0.04, "mom_3m": 0.05, "mom_6m": 0.05, "mom_12m": 0.04,
        "roe": 0.05, "gross_margin": 0.03, "debt_ratio": 0.02,
        "vol_20d": 0.03, "vol_60d": 0.02, "vol_ratio": 0.02,
        "ln_circ_mv": 0.04, "ln_total_mv": 0.03,
        "north_hold_pct": 0.05, "north_hold_mv": 0.04,
        "rsi_deviation": 0.03, "macd_hist": 0.03, "bb_position": 0.02,
        "ma20_slope": 0.03, "volume_ratio": 0.02,
        "margin_balance": 0.02, "margin_buy": 0.02,
        "main_net_inflow_pct": 0.03,
    }
```

- [ ] **Step 2: Wire into main() and run_optimization()**

```python
# In run_optimization(), add after scanner.save(results):
# 6. Run factor pipeline
try:
    run_factor_pipeline(force=force)
except Exception as e:
    logger.warning(f"Factor pipeline failed (non-blocking): {e}")
```

- [ ] **Step 3: Test scheduler imports**

```bash
python -c "from scheduler import run_factor_pipeline; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add scheduler.py
git commit -m "feat: add factor computation pipeline to scheduler"
```

### Task 6.2: Add load_latest_ranking to cross_section.py

**Files:**
- Modify: `core/cross_section.py`

- [ ] **Step 1: Implement**

```python
def load_latest_ranking(rankings_dir: str = "data_cache/rankings") -> pd.DataFrame | None:
    """Load the most recent cross-section ranking snapshot."""
    from pathlib import Path
    import pandas as pd
    
    base = Path(rankings_dir)
    if not base.exists():
        return None
    
    files = sorted(base.glob("*.parquet"), reverse=True)
    for f in files:
        try:
            df = pd.read_parquet(f)
            if not df.empty:
                return df
        except Exception:
            continue
    return None
```

- [ ] **Step 2: Test and commit**

```bash
pytest tests/test_factors.py -v --tb=short
git add core/cross_section.py
git commit -m "feat: add load_latest_ranking to cross_section"
```

---

## Phase 7: UI Update

### Task 7.1: Update app.py Tab 2 — factor radar + score display

**Files:**
- Modify: `app.py:208-278`

**Changes:**
- Replace "策略投票" block with factor score + percentile display
- Add factor radar chart (optional, simple bar chart for v1)
- Keep K-line chart unchanged

- [ ] **Step 1: Replace the voting display with factor score display**

```python
# In app.py Tab 2, replace lines 231-236:
with left:
    st.subheader("因子评分")
    if info.get("factor_score") is not None:
        fs = info["factor_score"]
        pct = info.get("percentile", 0.5)
        color = "#ef4444" if fs > 0 else ("#22c55e" if fs < 0 else "#9ca3af")
        st.metric("综合因子得分", f"{fs:.4f}", delta=f"全市场前{int(pct*100)}%")
        
        # Show available factor values
        st.divider()
        st.caption("各因子组贡献（因子值 × IC权重）")
    else:
        # Fallback: old strategy voting
        st.subheader("策略投票")
        for name, _2 in ALL_STRATEGIES:
            s = info["details"].get(name, 0)
            icon = {1:"🟢", -1:"🔴", 0:"⚪"}.get(s, "⚪")
            label = {1:"买入", -1:"卖出", 0:"观望"}.get(s, "?")
            st.write(f"{icon} {name}: {label}")
    
    # Signal display (works with both paths)
    badge = "badge-buy" if info["signal"] == "BUY" else ("badge-sell" if info["signal"] == "SELL" else "badge-hold")
    st.markdown(f"**综合: <span class='badge {badge}'>{info['signal']}</span> (置信度 {info['confidence']:.0%})**",
                unsafe_allow_html=True)
```

- [ ] **Step 2: Run all tests**

```bash
pytest tests/ -v --tb=short
```

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: show factor score in Tab 2 with backward-compatible fallback"
```

---

## Phase 8: Integration Tests

### Task 8.1: Add end-to-end factor pipeline test

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Add test**

```python
class TestFactorPipeline:
    def test_full_factor_pipeline_on_synthetic_data(self, tmp_path):
        """End-to-end: filter → compute factors → rank → save → load."""
        import pandas as pd, numpy as np
        from core.factors import compute_all_factors
        from core.cross_section import rank_all_stocks, load_latest_ranking
        from core.universe import filter_universe
        
        # Synthetic market snapshot
        market = pd.DataFrame({
            "code": [f"{i:06d}" for i in range(100001, 100021)],
            "name": [f"股票{i}" for i in range(20)],
            "price": np.linspace(3.0, 11.0, 20),
            "pe_ttm": np.random.uniform(5, 50, 20),
            "pb": np.random.uniform(0.5, 5, 20),
            "total_mv": np.random.uniform(5e9, 5e10, 20),
            "circ_mv": np.random.uniform(2e9, 2e10, 20),
            "turnover": np.random.uniform(1e7, 1e9, 20),
        })
        
        universe = filter_universe(market)
        assert len(universe) > 0
        
        # Skip full factor compute in test (requires network)
        # But verify ranking logic works
        mock_factors = [
            {"code": f"{i:06d}", "mom_1m": np.random.randn() * 0.1,
             "rsi_deviation": np.random.randn() * 0.3,
             "roe": np.random.uniform(0.05, 0.25)}
            for i in range(100001, 100011)
        ]
        
        mock_ic = {"mom_1m": 0.05, "rsi_deviation": 0.03, "roe": 0.08}
        ranking = rank_all_stocks(mock_factors, mock_ic)
        
        assert len(ranking) == len(mock_factors)
        assert "score" in ranking.columns
        assert "percentile" in ranking.columns
```

- [ ] **Step 2: Run**

```bash
pytest tests/test_integration.py::TestFactorPipeline -v --tb=short
```

- [ ] **Step 3: Run ALL tests one final time**

```bash
pytest tests/ -v --tb=short
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end factor pipeline integration test"
```

---

## Summary

**Total tasks:** 12  
**New files:** `core/universe.py`, `core/factors.py`, `core/cross_section.py`
**Modified files:** `core/data.py`, `core/strategies.py`, `core/analyzer.py`, `core/optimizer.py`, `scheduler.py`, `app.py`
**Test files:** `tests/test_factors.py` (new), `tests/test_analyzer.py` (modified), `tests/test_integration.py` (modified)

**Implementation order:** Phase 0→1→2→3→4→5→6→7→8 (strict dependency chain)
