"""Shared test fixtures."""
import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a random walk."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_rows, freq="B")
    close = 10.0 + np.cumsum(rng.normal(0, 0.2, n_rows))
    close = np.maximum(close, 1.0)  # price can't go negative
    high = close + rng.uniform(0, 0.5, n_rows)
    low = close - rng.uniform(0, 0.5, n_rows)
    low = np.maximum(low, 0.01)
    open_price = close + rng.uniform(-0.3, 0.3, n_rows)
    volume = rng.integers(1_000_000, 10_000_000, n_rows)
    return pd.DataFrame(
        {"Open": open_price, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """200-row synthetic OHLCV for normal testing."""
    return _make_ohlcv(200)


@pytest.fixture
def sample_ohlcv_short() -> pd.DataFrame:
    """30-row synthetic OHLCV — triggers insufficient-data paths."""
    return _make_ohlcv(30)
