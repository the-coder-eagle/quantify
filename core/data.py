"""Data pipeline — fetch, clean, cache. Single entry point."""
import time, re, random
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, datetime
from loguru import logger
import requests
import akshare as ak

# Stock code must be exactly 6 digits
_VALID_CODE = re.compile(r'^\d{6}$')

# Fallback: typical number of A-share stocks (~5500 as of 2025)
_DEFAULT_MARKET_SIZE = 5535


def _validate_code(code: str) -> str:
    """Validate and normalize a 6-digit stock code. Raises ValueError if invalid."""
    code = code.strip()
    if not code:
        raise ValueError("Stock code must not be empty")
    code = code.zfill(6)
    if not _VALID_CODE.match(code):
        raise ValueError(f"Invalid stock code: {code!r}")
    return code


class DataManager:
    def __init__(self, cache_dir="data_cache/daily"):
        self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, codes, start="2020-01-01", end=None, refresh=False):
        if end is None: end = date.today().strftime("%Y-%m-%d")
        if isinstance(codes, str): codes = [codes]
        codes = [_validate_code(c) for c in codes]
        results = {}
        for c in codes:
            df = self._get_one(c, start, end, refresh)
            if not df.empty: results[c] = df
        if len(results) == 1: return list(results.values())[0]
        return results if results else pd.DataFrame()

    def _get_one(self, code, start, end, refresh):
        path = self.cache_dir / f"{code}.parquet"
        cache_max_age_hours = 24  # re-fetch daily data after 24h

        if not refresh and path.exists():
            file_age_hours = (time.time() - path.stat().st_mtime) / 3600
            if file_age_hours < cache_max_age_hours:
                cached = pd.read_parquet(path)
                if isinstance(cached.index, pd.DatetimeIndex):
                    if pd.Timestamp(start) >= cached.index.min() and pd.Timestamp(end) <= cached.index.max():
                        return cached.loc[pd.Timestamp(start):pd.Timestamp(end)]
            else:
                logger.debug(f"Cache expired for {code} (age: {file_age_hours:.1f}h)")

        # ── Fetch: route by instrument type ──
        if _is_etf(code):
            df = _fetch_etf_history(code, start, end)
        else:
            try:
                raw = ak.stock_zh_a_daily(symbol=f"{_prefix(code)}{code}", start_date=start.replace("-",""),
                                           end_date=end.replace("-",""), adjust="qfq")
            except Exception:
                try:
                    raw = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start.replace("-",""),
                                              end_date=end.replace("-",""), adjust="qfq")
                except Exception:
                    logger.warning(f"Both akshare APIs failed for {code}")
                    return pd.DataFrame()

            if raw is None or raw.empty: return pd.DataFrame()
            df = raw.rename(columns={"date":"Date","open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume","成交额":"Turnover"})
            if "Date" in df.columns: df["Date"] = pd.to_datetime(df["Date"]); df = df.set_index("Date")
            df.index.name = "Date"; df = df.sort_index()

        # ── Common post-processing ──
        for col in ["Open","High","Low","Close","Volume"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        if "Close" in df.columns: df = df.dropna(subset=["Close"])
        if not df.empty:
            if path.exists():
                old = pd.read_parquet(path); df = pd.concat([old, df]).sort_index()
                df = df[~df.index.duplicated(keep="last")]
            df.to_parquet(path)
        return df.loc[pd.Timestamp(start):pd.Timestamp(end)] if not df.empty else df

def _prefix(code):
    """Return exchange prefix for a 6-digit A-share code.

    Shanghai: 6xxxxx (stock), 9xxxxx (B-share), 5xxxxx (ETF)
    Shenzhen: 0xxxxx/2xxxxx/3xxxxx (stock), 1xxxxx (ETF)
    Beijing:  8xxxxx
    """
    c0 = code[0]
    if c0 in "569":    # stocks 6/9 + ETFs 5 → Shanghai
        return "sh"
    elif c0 in "0123": # stocks 0/2/3 + ETFs 1 → Shenzhen
        return "sz"
    elif c0 == "8":    # Beijing Stock Exchange
        return "bj"
    return "sh"  # defensive default

def _is_etf(code: str) -> bool:
    """True if the 6-digit code is an ETF.

    Shanghai ETFs: 51xxxx, 56xxxx, 58xxxx …
    Shenzhen ETFs: 15xxxx, 159xxx …
    A-share stocks never start with 5 or 1 → zero false positives.
    """
    return code.zfill(6)[0] in "15"

def _standardize_etf_em(raw: pd.DataFrame) -> pd.DataFrame:
    """Map fund_etf_hist_em() Chinese columns → canonical English OHLCV."""
    col_map = {
        "日期": "Date", "开盘": "Open", "收盘": "Close",
        "最高": "High", "最低": "Low", "成交量": "Volume", "成交额": "Turnover",
    }
    df = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
    df.index.name = "Date"
    df = df.sort_index()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Close" in df.columns:
        df = df.dropna(subset=["Close"])
    return df

def _standardize_etf_sina(raw: pd.DataFrame) -> pd.DataFrame:
    """Map fund_etf_hist_sina() lowercase columns → canonical English OHLCV."""
    col_map = {
        "date": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "volume": "Volume",
    }
    df = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
    df.index.name = "Date"
    df = df.sort_index()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Close" in df.columns:
        df = df.dropna(subset=["Close"])
    return df

def _fetch_etf_history(code: str, start: str, end: str) -> pd.DataFrame:
    """Fetch ETF daily history with fallback: Eastmoney → Sina → empty.

    Returns standardized OHLCV DataFrame (English columns, Date index).
    """
    start_fmt = start.replace("-", "")
    end_fmt = end.replace("-", "")

    # ── Primary: Eastmoney ETF history ──
    try:
        raw = ak.fund_etf_hist_em(
            symbol=code, period="daily",
            start_date=start_fmt, end_date=end_fmt, adjust="qfq",
        )
    except Exception:
        raw = None

    if raw is not None and not raw.empty:
        df = _standardize_etf_em(raw)
        if not df.empty:
            return df

    # ── Fallback: Sina ETF history ──
    try:
        raw2 = ak.fund_etf_hist_sina(symbol=f"{_prefix(code)}{code}")
    except Exception:
        raw2 = None

    if raw2 is not None and not raw2.empty:
        df = _standardize_etf_sina(raw2)
        if not df.empty:
            try:
                df = df.loc[pd.Timestamp(start):pd.Timestamp(end)]
            except Exception:
                pass
        if not df.empty:
            return df

    return pd.DataFrame()

EM_FIELDS = {
    "f2": "price", "f3": "pct_chg", "f4": "change", "f5": "volume",
    "f6": "turnover", "f12": "code", "f14": "name", "f15": "high",
    "f16": "low", "f17": "open", "f18": "prev_close",
    "f9": "pe_ttm",          # 市盈率(动态)
    "f23": "pb",             # 市净率
    "f20": "total_mv",       # 总市值
    "f21": "circ_mv",        # 流通市值
    "f8": "turnover_rate",   # 换手率
    "f10": "volume_ratio",   # 量比
}
EM_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

EM_BASE_PARAMS = {
    "pn": "1", "pz": "100", "po": "1", "np": "1",
    "fltt": "2", "invt": "2", "fid": "f6",
    "fs": "m:0+t:6,m:0+t:80,m:0+t:81,m:1+t:2,m:1+t:23",
    "fields": ",".join(EM_FIELDS.keys()),
}
EM_URL = "https://push2.eastmoney.com/api/qt/clist/get"
EM_PAGE_SIZE = 100


def _fetch_em_page(session, page):
    """Fetch one page from Eastmoney API using a shared session."""
    params = {**EM_BASE_PARAMS, "pn": str(page)}
    for attempt in range(3):
        try:
            r = session.get(EM_URL, params=params, timeout=15)
            if r.status_code == 429:
                backoff = 2 ** (attempt + 1)  # 2, 4, 8s
                logger.warning(f"Eastmoney rate-limited (HTTP 429) at page {page}, backing off {backoff}s")
                time.sleep(backoff)
                continue
            if r.status_code != 200:
                if attempt < 2:
                    time.sleep(1 + attempt)
                continue
            data = r.json()
            if data.get("rc") != 0 or not data.get("data") or not data["data"].get("diff"):
                return None
            return data["data"]["diff"]
        except Exception:
            if attempt < 2:
                time.sleep(1 + attempt)
    return None


def _fetch_em_snapshot(min_price, max_price=None):
    """Fetch ALL A-share snapshots from Eastmoney (sequential, rate-limit friendly)."""
    session = requests.Session()
    session.headers.update(EM_HEADERS)

    first_page = _fetch_em_page(session, 1)
    if not first_page:
        return pd.DataFrame()

    total = _DEFAULT_MARKET_SIZE
    try:
        params = {**EM_BASE_PARAMS, "pn": "1", "pz": "1"}
        r = session.get(EM_URL, params=params, timeout=10)
        info = r.json()
        if info.get("data") and info["data"].get("total"):
            total = info["data"]["total"]
    except Exception:
        pass

    total_pages = (total + EM_PAGE_SIZE - 1) // EM_PAGE_SIZE
    all_rows = list(first_page)

    # Fetch remaining pages sequentially with jittered delays
    if total_pages > 1:
        for p in range(2, total_pages + 1):
            time.sleep(0.5 + random.uniform(0, 0.25))  # ~2 pages/sec with jitter
            rows = _fetch_em_page(session, p)
            if rows:
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows).rename(columns=EM_FIELDS)
    keep = [v for v in EM_FIELDS.values() if v in df.columns]
    df = df[keep].copy()
    for c in ["price", "pct_chg", "change", "volume", "turnover", "open",
              "prev_close", "high", "low", "pe_ttm", "pb",
              "total_mv", "circ_mv", "turnover_rate", "volume_ratio"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "price" in df.columns:
        df = df[(df["price"] >= min_price) & (df["price"] > 0)]
        if max_price is not None:
            df = df[df["price"] <= max_price]
    if "pct_chg" in df.columns:
        df = df[df["pct_chg"].between(-20, 20)]  # widened for cross-border ETFs
    if "name" in df.columns:
        df = df[~df["name"].str.contains("ST|退", na=False)]
    if "turnover" in df.columns:
        df = df.sort_values("turnover", ascending=False)
    return df


def market_snapshot(limit=500, min_price=3.0, max_price=None):
    """Fetch A-share market snapshot. Sina primary (fast-fail), Eastmoney fallback."""
    # Primary: Sina (fast-fail on 456 — no retry, immediate fallback)
    try:
        df = _fetch_sina_snapshot(min_price, max_price)
        if not df.empty:
            return df.head(limit)
    except Exception as e:
        logger.warning(f"Sina failed: {e}")

    # Fallback: Eastmoney (more reliable, less strict rate-limiting)
    logger.warning("Sina unavailable, trying Eastmoney...")
    try:
        df = _fetch_em_snapshot(min_price, max_price)
        if not df.empty:
            return df.head(limit)
    except Exception as e:
        logger.warning(f"Eastmoney fallback failed: {e}")
    return pd.DataFrame()


# Sina column mapping
_SINA_COL_MAP = {
    "symbol": "raw_symbol", "code": "code", "name": "name",
    "trade": "price", "changepercent": "pct_chg", "pricechange": "change",
    "volume": "volume", "amount": "turnover",
    "open": "open", "high": "high", "low": "low",
    "settlement": "prev_close",
}

_SINA_URL = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
_SINA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Referer": "https://vip.stock.finance.sina.com.cn/mkt/#hs_a",
}


def _fetch_sina_snapshot(min_price, max_price=None):
    """Fetch ALL A-share snapshots from Sina (paginated).

    On HTTP 456 (rate-limit) breaks immediately — Sina's IP ban lasts minutes,
    retrying is pointless. Caller should fall back to Eastmoney.
    """
    session = requests.Session()
    session.headers.update(_SINA_HEADERS)
    all_rows = []
    page = 1
    while True:
        params = {
            "page": page, "num": 80, "sort": "symbol", "asc": 1,
            "node": "hs_a", "symbol": "", "_s_r_a": "init",
        }
        try:
            r = session.get(_SINA_URL, params=params, timeout=15)
            if r.status_code != 200:
                if r.status_code == 456:
                    logger.warning(f"Sina rate-limited (HTTP 456) at page {page}, switching to fallback")
                break
            data = r.json()
        except Exception:
            break
        if not data:
            break
        all_rows.extend(data)
        page += 1
        time.sleep(0.6 + random.uniform(0, 0.3))  # ~1.3 pages/sec with jitter

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # Keep only columns we have mappings for
    df = df[[k for k in _SINA_COL_MAP if k in df.columns]].copy()
    df = df.rename(columns={k: v for k, v in _SINA_COL_MAP.items() if k in df.columns})

    # Clean numeric columns
    for c in ["price", "pct_chg", "change", "volume", "turnover", "open", "prev_close", "high", "low"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "price" in df.columns:
        df = df[(df["price"] >= min_price) & (df["price"] > 0)]
        if max_price is not None:
            df = df[df["price"] <= max_price]
    if "pct_chg" in df.columns:
        df = df[df["pct_chg"].between(-20, 20)]
    if "name" in df.columns:
        df = df[~df["name"].str.contains("ST|退", na=False)]
    if "turnover" in df.columns:
        df = df.sort_values("turnover", ascending=False)
    return df


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


def fetch_margin_data(code: str) -> dict | None:
    """Fetch margin trading data for a single stock.

    Tries SSE first, then SZSE. Returns dict with:
      margin_balance: recent margin balance (元)
      margin_buy: recent margin buy amount (元)
      short_balance: short selling balance (元)
      created_at: timestamp
    """
    code = str(code).zfill(6)
    try:
        import akshare as ak

        for fetch_fn in [ak.stock_margin_detail_sse, ak.stock_margin_detail_szse]:
            try:
                df = fetch_fn(date=date.today().strftime("%Y%m%d"))
                if df is not None and not df.empty:
                    num_cols = len(df.columns)
                    if num_cols == 9:  # SSE: date, code, name, bal, buy, short_vol, short_sell, short_bal, total
                        code_col = df.columns[1]
                        margin_balance_idx = 3
                        margin_buy_idx = 4
                        short_balance_idx = 7
                    else:  # SZSE (8 cols): code, name, buy, bal, short_vol, short_qty, short_bal, total
                        code_col = df.columns[0]
                        margin_balance_idx = 3
                        margin_buy_idx = 2
                        short_balance_idx = 6

                    stock_rows = df[df[code_col].astype(str) == code]
                    if not stock_rows.empty:
                        row = stock_rows.iloc[0]
                        return {
                            "margin_balance": float(row.iloc[margin_balance_idx]),
                            "margin_buy": float(row.iloc[margin_buy_idx]),
                            "short_balance": float(row.iloc[short_balance_idx]),
                            "created_at": str(pd.Timestamp.now()),
                        }
            except Exception:
                continue
        return None
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
