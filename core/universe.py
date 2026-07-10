"""Stock universe filtering -- exclude untradeable stocks."""

import pandas as pd
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
    - ST / *ST / delisted stocks
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

    logger.info(f"  Universe: {initial} -> {len(df)} stocks ({len(df)/max(initial,1)*100:.1f}%)")
    return df.reset_index(drop=True)
