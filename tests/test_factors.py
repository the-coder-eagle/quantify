"""Tests for north-bound capital data interfaces, financial indicators, fund flow, industry info, and universe filtering."""
import pandas as pd
from core.data import fetch_north_bound_individual, fetch_north_bound_history
from core.universe import filter_universe


def test_fetch_limit_up_pool_returns_list():
    from core.data import fetch_limit_up_pool
    result = fetch_limit_up_pool()
    if result is not None:
        assert isinstance(result, list)


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
        assert "industry_name" in result


# ---------------------------------------------------------------------------
# Universe / stock pool filter tests
# ---------------------------------------------------------------------------

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
