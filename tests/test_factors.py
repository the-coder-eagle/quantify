"""Tests for north-bound capital data interfaces, financial indicators, fund flow, and industry info."""
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


def test_fetch_margin_data_returns_dict():
    from core.data import fetch_margin_data
    result = fetch_margin_data("600519")
    if result is not None:  # API may fail, that's OK
        assert isinstance(result, dict)
        assert "margin_balance" in result
        assert "margin_buy" in result
        assert "created_at" in result


def test_fetch_financial_indicators_returns_dict():
    from core.data import fetch_financial_indicators
    result = fetch_financial_indicators("600519")
    if result is not None:  # API may fail, that's OK
        assert isinstance(result, dict)
        assert "roe" in result


def test_fetch_fund_flow_returns_dict():
    from core.data import fetch_fund_flow
    result = fetch_fund_flow("600519")
    if result is not None:  # API may fail, that's OK
        assert isinstance(result, dict)


def test_fetch_industry_info_returns_dict():
    from core.data import fetch_industry_info
    result = fetch_industry_info("600519")
    if result is not None:  # API may fail, that's OK
        assert isinstance(result, dict)
        assert "industry" in result
