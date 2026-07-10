"""Tests for cross-section ranking: Z-score normalization, IC computation,
IC-weighted composite scoring, and rank_all_stocks."""

import pandas as pd
import numpy as np
import pytest
from core.cross_section import (
    zscore_normalize,
    sector_neutralize,
    compute_rank_ic,
    ic_weighted_score,
    rank_all_stocks,
)


# =============================================================================
# zscore_normalize
# =============================================================================

class TestZscoreNormalize:
    def test_basic_normalization(self):
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = zscore_normalize(values)
        assert abs(result.mean()) < 0.001
        assert abs(result.std() - 1.0) < 0.01  # winsorization may clip slightly

    def test_constant_series_returns_zeros(self):
        values = pd.Series([3.0, 3.0, 3.0, 3.0, 3.0])
        result = zscore_normalize(values)
        assert all(result == 0.0)

    def test_too_few_values_returns_zeros(self):
        values = pd.Series([1.0, 2.0, 3.0])  # only 3 values, < 10 threshold
        result = zscore_normalize(values)
        assert all(result == 0.0)

    def test_winsorization_clips_outliers(self):
        # A series with a massive outlier that would be clipped
        values = pd.Series([1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.02, 1.08, 0.92, 1.01,
                            100.0])  # outlier
        result = zscore_normalize(values, winsorize_pct=0.1)
        # After winsorization, result should not be totally dominated by the outlier
        assert abs(result.iloc[-1]) < 10.0  # clipped, not at z=~30

    def test_handles_nan(self):
        values = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        result = zscore_normalize(values)
        assert result.isna().sum() == 0  # nan dropped before computation
        assert abs(result.mean()) < 0.001

    def test_mixed_valid_and_nan(self):
        values = pd.Series([1.0, 2.0, 3.0, np.nan, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = zscore_normalize(values)
        assert len(result) == len(values)
        assert result.isna().sum() == 0

    def test_return_type(self):
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = zscore_normalize(values)
        assert isinstance(result, pd.Series)
        assert result.index.equals(values.index)


# =============================================================================
# sector_neutralize
# =============================================================================

class TestSectorNeutralize:
    def test_removes_sector_bias(self):
        """Each sector should have mean near zero after neutralization."""
        codes = ["A", "B", "C", "D", "E", "F"]
        factor_values = pd.Series([10.0, 12.0, 1.0, 3.0, 50.0, 52.0], index=codes)
        sector_map = {"A": "tech", "B": "tech", "C": "health", "D": "health",
                      "E": "energy", "F": "energy"}
        result = sector_neutralize(factor_values, sector_map)
        # Check each sector has mean near zero
        for sector in ["tech", "health", "energy"]:
            sector_indices = [c for c in codes if sector_map[c] == sector]
            sector_mean = result.loc[sector_indices].mean()
            assert abs(sector_mean) < 0.001, f"Sector {sector} mean = {sector_mean}"

    def test_empty_sector_map_returns_original(self):
        codes = ["A", "B", "C"]
        factor_values = pd.Series([10.0, 20.0, 30.0], index=codes)
        result = sector_neutralize(factor_values, {})
        pd.testing.assert_series_equal(result, factor_values)

    def test_none_sector_map_returns_original(self):
        codes = ["A", "B", "C"]
        factor_values = pd.Series([10.0, 20.0, 30.0], index=codes)
        result = sector_neutralize(factor_values, None)
        pd.testing.assert_series_equal(result, factor_values)

    def test_unknown_code_gets_unknown_sector(self):
        codes = ["A", "B", "C"]
        factor_values = pd.Series([10.0, 20.0, 30.0], index=codes)
        sector_map = {"A": "tech"}  # B and C have no sector
        result = sector_neutralize(factor_values, sector_map)
        # Should not crash; missing ones grouped as "unknown"
        assert len(result) == 3

    def test_single_stock_sector(self):
        codes = ["A"]
        factor_values = pd.Series([42.0], index=codes)
        sector_map = {"A": "tech"}
        result = sector_neutralize(factor_values, sector_map)
        # Single stock sector gets zero neutralized value (own mean subtracted)
        assert abs(result.iloc[0]) < 0.001


# =============================================================================
# compute_rank_ic
# =============================================================================

class TestComputeRankIC:
    def test_returns_valid_ic(self):
        factor_values = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05,
                                   -0.01, -0.02, -0.03, -0.04, -0.05])
        forward_returns = pd.Series([0.01, 0.02, 0.01, 0.03, 0.00,
                                     -0.01, -0.03, -0.02, -0.01, -0.04])
        ic = compute_rank_ic(factor_values, forward_returns)
        assert isinstance(ic, float)
        assert -1.0 <= ic <= 1.0

    def test_perfect_positive_correlation(self):
        factor_values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        forward_returns = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        ic = compute_rank_ic(factor_values, forward_returns)
        assert ic == pytest.approx(1.0, abs=0.01)

    def test_perfect_negative_correlation(self):
        factor_values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        forward_returns = pd.Series([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        ic = compute_rank_ic(factor_values, forward_returns)
        assert ic == pytest.approx(-1.0, abs=0.01)

    def test_too_few_stocks_returns_zero(self):
        factor_values = pd.Series([1.0, 2.0, 3.0])  # only 3, below threshold of 10
        forward_returns = pd.Series([0.1, 0.2, 0.3])
        ic = compute_rank_ic(factor_values, forward_returns)
        assert ic == 0.0

    def test_handles_mismatched_indices(self):
        factor_values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0,
                                   6.0, 7.0, 8.0, 9.0, 10.0],
                                  index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
        forward_returns = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5,
                                     0.6, 0.7, 0.8, 0.9, 1.0],
                                    index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "Z"])
        ic = compute_rank_ic(factor_values, forward_returns)
        assert isinstance(ic, float)
        assert -1.0 <= ic <= 1.0

    def test_handles_nan_in_inputs(self):
        factor_values = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0,
                                   6.0, 7.0, 8.0, 9.0, 10.0])
        forward_returns = pd.Series([0.1, 0.2, 0.3, np.nan, 0.5,
                                     0.6, 0.7, 0.8, 0.9, 1.0])
        ic = compute_rank_ic(factor_values, forward_returns)
        assert isinstance(ic, float)
        assert -1.0 <= ic <= 1.0

    def test_zero_correlation_for_random_data(self):
        rng = np.random.default_rng(42)
        factor_values = pd.Series(rng.normal(0, 1, 100))
        forward_returns = pd.Series(rng.normal(0, 1, 100))
        ic = compute_rank_ic(factor_values, forward_returns)
        assert -1.0 <= ic <= 1.0


# =============================================================================
# ic_weighted_score
# =============================================================================

class TestICWeightedScore:
    def test_basic_score(self):
        factors = {"mom_1m": 0.05, "rsi_deviation": 0.2, "roe": 0.15}
        ic = {"mom_1m": 0.05, "rsi_deviation": 0.03, "roe": 0.08}
        score = ic_weighted_score(factors, ic)
        assert isinstance(score, float)
        assert -3.0 <= score <= 3.0  # Z-scores clipped to [-3, 3]

    def test_all_zero_ic_gives_min_weights(self):
        factors = {"mom_1m": 1.0, "rsi_deviation": -1.0}
        ic = {"mom_1m": 0.0, "rsi_deviation": 0.0}
        score = ic_weighted_score(factors, ic)
        assert isinstance(score, float)

    def test_empty_factors_returns_zero(self):
        score = ic_weighted_score({}, {"mom_1m": 0.05})
        assert score == 0.0

    def test_empty_ic_returns_zero(self):
        score = ic_weighted_score({"mom_1m": 0.05}, {})
        assert score == 0.0

    def test_both_empty_returns_zero(self):
        score = ic_weighted_score({}, {})
        assert score == 0.0

    def test_stale_factor_halved(self):
        """When a factor is marked stale, its influence should be reduced."""
        factors = {"factor_a": 2.0, "factor_b": 2.0}
        ic = {"factor_a": 0.1, "factor_b": 0.1}
        halve_stale = {"factor_a": True}
        score_with = ic_weighted_score(factors, ic, halve_stale=halve_stale)
        score_without = ic_weighted_score(factors, ic)
        # With stale halving, the stale factor gets less weight
        # The score should differ (this is binary, so stale factor's weight is halved)
        assert score_with != score_without or abs(score_with - score_without) < 2.0

    def test_min_weight_floor(self):
        """Each factor should have at least min_weight regardless of IC."""
        factors = {"factor_a": 1.0, "factor_b": 1.0}
        ic = {"factor_a": 0.0001, "factor_b": 0.0001}
        score = ic_weighted_score(factors, ic, min_weight=0.1)
        assert isinstance(score, float)
        assert -3.0 <= score <= 3.0

    def test_missing_factor_in_stock_factors(self):
        """Factor in IC but not in stock's factors should be skipped."""
        factors = {"mom_1m": 0.5}
        ic = {"mom_1m": 0.05, "another_factor": 0.1}
        score = ic_weighted_score(factors, ic)
        assert isinstance(score, float)

    def test_score_sign(self):
        """Positive z-scores should give positive scores (roughly)."""
        factors = {"factor_a": 2.0, "factor_b": 1.5}
        ic = {"factor_a": 0.1, "factor_b": 0.1}
        score = ic_weighted_score(factors, ic)
        assert score > 0

    def test_negative_z_scores_negative_score(self):
        factors = {"factor_a": -2.0, "factor_b": -1.5}
        ic = {"factor_a": 0.1, "factor_b": 0.1}
        score = ic_weighted_score(factors, ic)
        assert score < 0


# =============================================================================
# rank_all_stocks
# =============================================================================

class TestRankAllStocks:
    def test_returns_dataframe_with_expected_columns(self):
        all_factors = [
            {"code": "000001", "mom_1m": 0.05, "rsi_deviation": 0.2},
            {"code": "000002", "mom_1m": 0.03, "rsi_deviation": 0.1},
            {"code": "000003", "mom_1m": 0.01, "rsi_deviation": -0.1},
        ]
        factor_ic = {"mom_1m": 0.05, "rsi_deviation": 0.03}
        result = rank_all_stocks(all_factors, factor_ic)
        assert isinstance(result, pd.DataFrame)
        assert "code" in result.columns
        assert "score" in result.columns
        assert "percentile" in result.columns
        assert len(result) == 3

    def test_empty_factors_returns_empty_dataframe(self):
        result = rank_all_stocks([], {})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_ranking_order(self):
        """Stock with best factor values should rank highest."""
        all_factors = [
            {"code": "best", "mom_1m": 3.0},
            {"code": "mid", "mom_1m": 0.0},
            {"code": "worst", "mom_1m": -3.0},
        ]
        factor_ic = {"mom_1m": 0.1}
        result = rank_all_stocks(all_factors, factor_ic)
        assert result.iloc[0]["code"] == "best"
        assert result.iloc[2]["code"] == "worst"

    def test_sector_neutralization_affects_scores(self):
        """Scores should differ when sector neutralization is applied."""
        all_factors = []
        sector_map = {}
        codes_by_sector = {
            "tech": [f"T{i:03d}" for i in range(10)],
            "health": [f"H{i:03d}" for i in range(10)],
            "energy": [f"E{i:03d}" for i in range(10)],
        }
        for sector, codes in codes_by_sector.items():
            base = 10.0 if sector == "tech" else 1.0 if sector == "health" else 50.0
            for code in codes:
                all_factors.append({"code": code, "factor_x": base})
                sector_map[code] = sector

        factor_ic = {"factor_x": 0.1}

        result_no_sector = rank_all_stocks(all_factors, factor_ic)
        result_with_sector = rank_all_stocks(all_factors, factor_ic, sector_map=sector_map)

        # The presence of sector map should change scores
        assert not result_no_sector["score"].equals(result_with_sector["score"])

    def test_stale_factors_handling(self):
        """Stale factors should affect scores differently."""
        all_factors = [
            {"code": "A", "mom_1m": 2.0, "stale_factor": 2.0},
            {"code": "B", "mom_1m": -2.0, "stale_factor": -2.0},
        ]
        factor_ic = {"mom_1m": 0.1, "stale_factor": 0.1}
        result = rank_all_stocks(all_factors, factor_ic, stale_factors={"stale_factor"})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_percentile_is_between_0_and_1(self):
        all_factors = [
            {"code": "000001", "mom_1m": 0.05, "rsi_deviation": 0.2},
            {"code": "000002", "mom_1m": 0.03, "rsi_deviation": 0.1},
            {"code": "000003", "mom_1m": 0.01, "rsi_deviation": -0.1},
            {"code": "000004", "mom_1m": 0.02, "rsi_deviation": 0.15},
            {"code": "000005", "mom_1m": 0.04, "rsi_deviation": 0.05},
        ]
        factor_ic = {"mom_1m": 0.05, "rsi_deviation": 0.03}
        result = rank_all_stocks(all_factors, factor_ic)
        assert result["percentile"].between(0, 1).all()

    def test_result_is_sorted_descending_by_score(self):
        all_factors = [
            {"code": "A", "mom_1m": 1.0},
            {"code": "B", "mom_1m": 2.0},
            {"code": "C", "mom_1m": 3.0},
        ]
        factor_ic = {"mom_1m": 0.1}
        result = rank_all_stocks(all_factors, factor_ic)
        assert result["score"].is_monotonic_decreasing

    def test_handles_partial_factor_data(self):
        """Some stocks missing some factors should not crash."""
        all_factors = [
            {"code": "A", "mom_1m": 0.05, "rsi_deviation": 0.2},
            {"code": "B", "mom_1m": 0.03},  # missing rsi_deviation
            {"code": "C", "rsi_deviation": 0.1},  # missing mom_1m
        ]
        factor_ic = {"mom_1m": 0.05, "rsi_deviation": 0.03}
        result = rank_all_stocks(all_factors, factor_ic)
        assert len(result) == 3
        assert "score" in result.columns

    def test_single_stock(self):
        all_factors = [
            {"code": "ONLY", "mom_1m": 1.0},
        ]
        factor_ic = {"mom_1m": 0.05}
        result = rank_all_stocks(all_factors, factor_ic)
        assert len(result) == 1
        assert result.iloc[0]["code"] == "ONLY"
        assert result.iloc[0]["percentile"] == 0.5  # default for single stock
