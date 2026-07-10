#!/usr/bin/env python
"""scheduler.py — Run daily optimization after market close.

Usage:
    python scheduler.py              # Run once (checks if trading day)
    python scheduler.py --force      # Run even on non-trading days / weekends
    python scheduler.py --daemon     # Run continuously, sleep until next trigger

On a server, use crontab instead of --daemon:
    30 15 * * 1-5 cd /path/to/quantify && python scheduler.py
"""
import sys, os, time, json
from datetime import date, datetime, time, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from loguru import logger

from core import DataManager, market_snapshot, ALL_STRATEGY_CLASSES
from core.optimizer import TradingCalendar, ParamScanner, EnsembleWeight, load_latest_weights
from config import app_max_price, app_min_price, get as cfg_get


def run_optimization(force=False):
    """Run a full optimization cycle. Returns True if results were saved."""
    cal = TradingCalendar()

    if not cal.should_run(force=force):
        return False

    logger.info("Starting daily optimization...")

    # 1. Fetch market snapshot (top N by turnover, within price range)
    top_n = cfg_get("optimizer", "top_n_stocks", default=50)
    max_p = app_max_price()
    min_p = app_min_price()

    logger.info(f"Fetching market snapshot (top {top_n}, price {min_p}-{max_p})...")
    market = market_snapshot(limit=top_n, min_price=min_p, max_price=max_p)

    if market.empty:
        logger.warning("Market snapshot empty — skipping optimization")
        return False

    codes = market["code"].dropna().astype(str).str.zfill(6).tolist()
    logger.info(f"Scanning {len(codes)} stocks...")

    # 2. Load historical data for each stock
    dm = DataManager()
    end = date.today().strftime("%Y-%m-%d")
    lookback = cfg_get("optimizer", "lookback_days", default=250)
    start = (date.today() - timedelta(days=lookback)).strftime("%Y-%m-%d")

    stock_data = {}
    for i, code in enumerate(codes):
        try:
            df = dm.get(code, start, end)
            if isinstance(df, pd.DataFrame) and len(df) >= 60:
                stock_data[code] = df
        except Exception as e:
            logger.warning(f"Data fetch failed for {code}: {e}")

        if (i + 1) % 10 == 0:
            logger.info(f"  Data: {i+1}/{len(codes)}")

    if not stock_data:
        logger.warning("No stock data fetched — skipping optimization")
        return False

    logger.info(f"Loaded data for {len(stock_data)} stocks")

    # 3. Run parameter scan
    scanner = ParamScanner()
    results = scanner.scan_all(stock_data)

    # 4. Compute ensemble weights from scan results
    ew = EnsembleWeight()
    # Build per-strategy returns from scan results
    perf = {}
    for name, _ in ALL_STRATEGY_CLASSES:
        # Use average Sharpe across all stocks as the performance signal
        sharpes = []
        for code, info in results.items():
            top = info.get("strategies", {}).get(name, [])
            if top:
                sharpes.append(top[0].get("sharpe", 0))
        perf[name] = [np.mean(sharpes)] if sharpes else [0.0]

    weights = ew.update(perf)
    logger.info(f"Ensemble weights: {json.dumps(weights, ensure_ascii=False)}")

    # 5. Save results
    scanner.save(results)
    ew.save()

    # 6. Run factor pipeline
    try:
        run_factor_pipeline(force=force)
    except Exception as e:
        logger.warning(f"Factor pipeline failed (non-blocking): {e}")

    logger.info("Optimization complete!")
    return True


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


def _next_trigger(now: datetime) -> datetime:
    """Calculate next sensible wake-up time to avoid busy-looping on idle days."""
    today = now.date()
    weekday = now.weekday()  # 0=Mon .. 6=Sun

    # If before market close on a weekday, check again at 15:05 today
    if weekday < 5 and now.time() < time(15, 5):
        return now.replace(hour=15, minute=5, second=0, microsecond=0)

    # Otherwise, find next weekday at 15:05
    days_ahead = 1
    next_date = today + timedelta(days=days_ahead)
    while next_date.weekday() >= 5:  # Skip weekends
        days_ahead += 1
        next_date = today + timedelta(days=days_ahead)

    return datetime.combine(next_date, time(15, 5))


def daemon_mode():
    """Run continuously, triggering optimization after market close on trading days."""
    logger.info("Scheduler daemon started. Waiting for next trigger...")
    cal = TradingCalendar()

    while True:
        now = datetime.now()
        if cal.should_run():
            logger.info("Triggering optimization...")
            run_optimization(force=True)
            # Sleep until tomorrow 15:05
            tomorrow = now + timedelta(days=1)
            next_run = tomorrow.replace(hour=15, minute=5, second=0, microsecond=0)
        else:
            next_run = _next_trigger(now)
            logger.debug(f"Not time yet. Next check: {next_run.strftime('%Y-%m-%d %H:%M')}")

        sleep_seconds = max(60, (next_run - datetime.now()).total_seconds())
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantify Daily Optimizer")
    parser.add_argument("--force", action="store_true", help="Run even on non-trading days")
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    args = parser.parse_args()

    if args.daemon:
        daemon_mode()
    else:
        success = run_optimization(force=args.force)
        if success:
            print("Optimization completed successfully.")
        else:
            print("Skipped — not a trading day, already run, or no data.")
        sys.exit(0 if success else 1)
