"""Tests for core.risk — position_size, stop_loss, kelly, check_stops."""
from core.risk import (
    position_size,
    kelly_fraction,
    stop_loss_price,
    check_stops,
    volatility_target_shares,
)


class TestPositionSize:
    def test_lot_rounding(self):
        # 100k equity, ¥50 price, 2% risk, 5% stop → value = 40,000 → 800 shares
        shares = position_size(100_000, 50, risk_pct=0.02, stop_pct=0.05, lot=100)
        assert shares == 800
        assert shares % 100 == 0

    def test_zero_price(self):
        assert position_size(100_000, 0, 0.02, 0.05) == 0

    def test_zero_stop(self):
        assert position_size(100_000, 50, 0.02, 0) == 0

    def test_small_position_rounds_to_zero(self):
        # Very small position should round to 0 lots
        shares = position_size(1000, 500, risk_pct=0.01, stop_pct=0.05, lot=100)
        assert shares == 0


class TestStopLoss:
    def test_long_stop(self):
        stop = stop_loss_price(100, atr_val=2.0, multiplier=2.0, direction=1)
        assert stop == 96.0

    def test_short_stop(self):
        stop = stop_loss_price(100, atr_val=2.0, multiplier=2.0, direction=-1)
        assert stop == 104.0


class TestCheckStops:
    def test_fixed_stop_triggers(self):
        exit_flag, reason = check_stops(entry=100, current=94, highest=105, atr_val=2.0, holding_days=5)
        assert exit_flag
        assert "固定止损" in reason

    def test_no_stop_triggered(self):
        exit_flag, reason = check_stops(entry=100, current=102, highest=105, atr_val=2.0, holding_days=5)
        assert not exit_flag

    def test_time_stop(self):
        # current=111 keeps price above all price-based stops; only time trips
        exit_flag, reason = check_stops(entry=100, current=111, highest=111, atr_val=1.0, holding_days=35)
        assert exit_flag
        assert "时间" in reason


class TestKelly:
    def test_basic(self):
        f = kelly_fraction(win_prob=0.6, avg_win=0.1, avg_loss=0.05, fraction=0.5)
        assert 0 < f <= 0.25

    def test_bad_inputs(self):
        assert kelly_fraction(0, 0.1, 0.05) == 0.0
        assert kelly_fraction(0.6, 0, 0.05) == 0.0


class TestVolatilityTarget:
    def test_basic(self):
        shares = volatility_target_shares(100_000, 50, target_vol=0.20, recent_vol=0.30, lot=100)
        assert shares >= 0
        assert shares % 100 == 0

    def test_zero_vol(self):
        assert volatility_target_shares(100_000, 50, recent_vol=0) == 0
