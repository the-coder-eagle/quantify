"""Tests for config module — loading, defaults, typed accessors."""
from config import (
    get, app_capital, app_risk_pct, app_max_price, app_min_price,
    app_scan_default, research_capital,
)


class TestConfigLoading:
    def test_imports_succeed(self):
        """Config module loads without error."""
        assert True

    def test_app_capital(self):
        assert app_capital() == 1200.0

    def test_app_risk_pct(self):
        assert app_risk_pct() == 0.02

    def test_app_max_price(self):
        assert app_max_price() == 12.0

    def test_app_min_price(self):
        assert app_min_price() == 2.0

    def test_app_scan_default(self):
        assert app_scan_default() == 500

    def test_research_capital(self):
        assert research_capital() == 100000.0

    def test_nested_get(self):
        val = get("costs", "min_commission")
        assert val == 5.0

    def test_get_default(self):
        val = get("nonexistent", "key", default=42)
        assert val == 42

    def test_regime_config(self):
        vh = get("regime", "vol_change_high", default=1.5)
        vl = get("regime", "vol_change_low", default=0.7)
        assert isinstance(vh, (int, float))
        assert isinstance(vl, (int, float))

    def test_optimizer_config(self):
        top_n = get("optimizer", "top_n_stocks", default=50)
        sw = get("optimizer", "score_weights", "sharpe", default=0.4)
        assert isinstance(top_n, int)
        assert isinstance(sw, (int, float))

    def test_ai_agent_config(self):
        max_rescan = get("ai_agent", "max_force_rescan_per_day", default=3)
        assert max_rescan == 3
