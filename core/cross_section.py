"""Cross-section ranking, IC computation, and IC-weighted factor scoring.

This module receives factor values from factors.py for all stocks and does:
1. Sector neutralization (within-sector Z-score)
2. Z-score normalization with winsorization
3. Rank IC computation (Spearman correlation of factor values vs forward returns)
4. IC-weighted composite scoring (softmax, min weight, stale-factor halving)
5. rank_all_stocks() — the master function that ties everything together
"""

import pandas as pd
import numpy as np


def zscore_normalize(series: pd.Series, winsorize_pct: float = 0.01) -> pd.Series:
    """Z-score normalize a factor series with winsorization.

    Args:
        series: Raw factor values across stocks
        winsorize_pct: Winsorize tails (default 1% each side)

    Returns:
        Z-scored series (mean approx 0, std approx 1)
    """
    s = series.dropna()
    if len(s) < 10:
        return pd.Series(0.0, index=series.index)

    # Winsorize — clip extreme values to reduce outlier impact
    lo = s.quantile(winsorize_pct)
    hi = s.quantile(1 - winsorize_pct)
    s_clipped = s.clip(lo, hi)

    mean = s_clipped.mean()
    std = s_clipped.std()

    if std == 0:
        return pd.Series(0.0, index=series.index)

    return (s_clipped - mean) / std


def sector_neutralize(
    factor_series: pd.Series, sector_map: dict[str, str] | None
) -> pd.Series:
    """Remove sector bias from factor values.

    Subtracts sector mean from each stock's factor value. Stocks whose code
    is not in ``sector_map`` are grouped into an "unknown" sector.

    Args:
        factor_series: index=code, values=factor value
        sector_map: {code: sector_name} or None

    Returns:
        Sector-neutralized series
    """
    if not sector_map:
        return factor_series

    df = pd.DataFrame(
        {"factor": factor_series, "sector": factor_series.index.map(sector_map)}
    )
    df["sector"] = df["sector"].fillna("unknown")

    sector_means = df.groupby("sector")["factor"].transform("mean")
    neutralized = df["factor"] - sector_means

    return neutralized


def compute_rank_ic(
    factor_values: pd.Series, forward_returns: pd.Series
) -> float:
    """Compute Rank IC: Spearman correlation between factor values and forward returns.

    Args:
        factor_values: Factor values across stocks
        forward_returns: Corresponding forward-period returns

    Returns:
        Rank IC value in [-1, 1]; 0.0 if fewer than 10 common observations
    """
    common_idx = factor_values.dropna().index.intersection(
        forward_returns.dropna().index
    )
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

    Each factor's weight starts as |IC|, optionally halved if stale.
    A ``min_weight`` floor is applied. Raw weights are softmax-normalised
    (temperature T=0.5), then the score is the dot product of those weights
    with the stock's Z-scored factor values (clipped to [-3, 3]).

    Args:
        stock_factors: {factor_name: z-scored value} for one stock
        factor_ic: {factor_name: trailing IC value}
        min_weight: Minimum raw weight per factor before softmax
        halve_stale: {factor_name: True} for factors whose IC should be halved

    Returns:
        Composite score (higher = better); 0.0 if either dict is empty
    """
    if not factor_ic or not stock_factors:
        return 0.0

    # Build raw weights with min-weight floor
    weights = {}
    for name, ic in factor_ic.items():
        w = abs(ic)
        if halve_stale and name in halve_stale and halve_stale[name]:
            w *= 0.5
        weights[name] = max(w, min_weight)

    # Softmax for final weights (temperature=0.5 for moderate spreading)
    names = list(weights.keys())
    w_array = np.array([weights[n] for n in names])
    w_array = np.exp(w_array / 0.5)
    w_array = w_array / w_array.sum()

    score = 0.0
    for i, name in enumerate(names):
        if name in stock_factors and stock_factors[name] is not None:
            # Cap extreme Z-scores to avoid a single factor dominating
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
        all_factors: List of {factor_name: raw_value} dicts, one per stock.
                     Each dict may include a ``"code"`` key identifying the stock.
        factor_ic: {factor_name: trailing_IC}
        sector_map: {code: sector_name} for neutralization (optional)
        stale_factors: Set of factor names with stale data (IC halved)

    Returns:
        DataFrame with columns: code, score, percentile — sorted descending by score
    """
    if not all_factors:
        return pd.DataFrame(columns=["code", "score", "percentile"])

    # Collect codes and factor names
    codes = []
    factor_names: set[str] = set()
    for f in all_factors:
        codes.append(f.get("code", "???"))
        factor_names.update(k for k in f if k != "code")

    # Build factor DataFrame: index=code, columns=factor_name
    factor_df = pd.DataFrame(all_factors, index=codes)
    if "code" in factor_df.columns:
        factor_df = factor_df.drop(columns=["code"])

    # Z-score normalize each factor (with optional sector neutralization)
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
            original_name = zc[:-2]  # strip "_z"
            val = factor_df.loc[code, zc]
            if pd.notna(val):
                z_factors[original_name] = float(val)

        # Build halve_stale dict for ic_weighted_score
        halve_dict: dict[str, bool] = {}
        for name in z_factors:
            halve_dict[name] = name in halve_stale

        score = ic_weighted_score(
            z_factors, factor_ic, halve_stale=halve_dict
        )
        scores.append({"code": str(code), "score": score})

    result = pd.DataFrame(scores)
    if "score" in result.columns and len(result) > 1:
        result["percentile"] = result["score"].rank(pct=True)
    else:
        result["percentile"] = 0.5

    return result.sort_values("score", ascending=False).reset_index(drop=True)


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
