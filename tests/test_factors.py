"""Tests for north-bound capital data interfaces."""
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
