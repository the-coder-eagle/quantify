"""
Quick scan: find investable stocks under 10 RMB with 1000 RMB capital.
100 shares minimum × price ≤ 1000 → max price = 10 RMB.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from core.data import market_snapshot, DataManager, enrich_valuation

CAPITAL = 1000
MAX_PRICE = 10.0   # 100 shares × 10 = 1000
MIN_PRICE = 2.0    # skip penny stocks
MIN_TURNOVER = 5e7 # 50M CNY daily minimum for liquidity

print("=" * 70)
print(f"  1000元资金扫描 — 单价 ≤ {MAX_PRICE}元，成交额 ≥ {MIN_TURNOVER/1e6:.0f}M")
print("=" * 70)

# 1. Fetch market snapshot
print("\n[1/3] 获取市场快照...")
df = market_snapshot(limit=800, min_price=MIN_PRICE)

if df.empty:
    print("ERROR: 未获取到任何数据")
    sys.exit(1)

print(f"  获取到 {len(df)} 只股票（价格 ≥ {MIN_PRICE}元）")

# 2. Filter
print(f"\n[2/3] 筛选条件：价格 ≤ {MAX_PRICE}元，成交额 ≥ {MIN_TURNOVER/1e6:.0f}M...")

# Already filtered by min_price in market_snapshot
mask = (df["price"] <= MAX_PRICE)
if "turnover" in df.columns:
    mask = mask & (df["turnover"] >= MIN_TURNOVER)

candidates = df[mask].copy()

# Sort by turnover (liquidity first)
if "turnover" in candidates.columns:
    candidates = candidates.sort_values("turnover", ascending=False)

print(f"  筛选出 {len(candidates)} 只候选股票")

# Enrich top candidates with PE/PB (Baidu source) — the Sina snapshot lacks
# valuation fields, and Eastmoney's push2 clist may be unreachable. Only the
# top-N by turnover are enriched (full-market per-stock calls would be too slow).
if "pe_ttm" not in candidates.columns or candidates["pe_ttm"].isna().any():
    print(f"  补齐候选股估值数据（百度源，并发 {min(30, len(candidates))} 只）...")
    before = int(candidates["pe_ttm"].notna().sum()) if "pe_ttm" in candidates.columns else 0
    enrich_valuation(candidates, top_n=30)
    after = int(candidates["pe_ttm"].notna().sum())
    print(f"  估值补齐: {before} → {after} 只 (PE/PB)")


# 3. Display candidates with key metrics
print(f"\n[3/3] 候选股票Top 30：")
print("-" * 100)
header = f"{'代码':<8} {'名称':<10} {'现价':>8} {'涨跌幅':>8} {'成交额(亿)':>12} {'换手率':>8} {'PE(TTM)':>10} {'PB':>8}"
print(header)
print("-" * 100)

for _, row in candidates.head(30).iterrows():
    code = str(row.get("code", "")).zfill(6)
    name = str(row.get("name", ""))[:8]
    price = row.get("price", np.nan)
    pct = row.get("pct_chg", np.nan)
    turnover = row.get("turnover", np.nan)
    turnover_rate = row.get("turnover_rate", np.nan)
    pe = row.get("pe_ttm", np.nan)
    pb = row.get("pb", np.nan)

    print(f"{code:<8} {name:<10} {price:>8.2f} {pct:>+7.2f}% {turnover/1e8:>11.2f}  {turnover_rate:>7.2f}% {pe:>9.1f}  {pb:>7.2f}")

# 4. Simple ranking: combine several factors
print(f"\n{'='*70}")
print("  综合评分排名（流动性 + 估值 + 动量）")
print(f"{'='*70}")

def rank_candidates(df):
    """Rank by: turnover (40%), low PE (20%), low PB (20%), positive momentum (20%)"""
    df = df.copy()
    scores = pd.DataFrame(index=df.index)

    # 1) Turnover rank (higher better) - weight 40%
    if "turnover" in df.columns:
        scores["turnover_rank"] = df["turnover"].rank(pct=True) * 0.40

    # 2) PE rank (lower better) - weight 20%
    if "pe_ttm" in df.columns:
        pe_clean = df["pe_ttm"].clip(lower=0, upper=500)
        scores["pe_rank"] = (1 - pe_clean.rank(pct=True, na_option="bottom")) * 0.20

    # 3) PB rank (lower better) - weight 20%
    if "pb" in df.columns:
        pb_clean = df["pb"].clip(lower=0, upper=50)
        scores["pb_rank"] = (1 - pb_clean.rank(pct=True, na_option="bottom")) * 0.20

    # 4) Momentum (price change) - weight 20%, prefer moderate positive
    if "pct_chg" in df.columns:
        pct_clean = df["pct_chg"].clip(-10, 10)
        scores["momentum_rank"] = pct_clean.rank(pct=True, na_option="bottom") * 0.20

    df["score"] = scores.sum(axis=1)
    return df.sort_values("score", ascending=False)


ranked = rank_candidates(candidates)

print(f"\n{'代码':<8} {'名称':<10} {'现价':>8} {'涨跌幅':>8} {'成交额(亿)':>12} {'PE':>8} {'PB':>8} {'综合分':>8}")
print("-" * 85)
for _, row in ranked.head(15).iterrows():
    code = str(row.get("code", "")).zfill(6)
    name = str(row.get("name", ""))[:8]
    price = row.get("price", np.nan)
    pct = row.get("pct_chg", np.nan)
    turnover = row.get("turnover", np.nan)
    pe = row.get("pe_ttm", np.nan)
    pb = row.get("pb", np.nan)
    score = row.get("score", 0)

    print(f"{code:<8} {name:<10} {price:>8.2f} {pct:>+7.2f}% {turnover/1e8:>11.2f}  {pe:>7.1f} {pb:>7.2f} {score:>8.4f}")

# 5. Allocation recommendation
# Define quality universe for allocation
has_pe = "pe_ttm" in ranked.columns and ranked["pe_ttm"].notna().any()
has_pb = "pb" in ranked.columns and ranked["pb"].notna().any()
if has_pe and has_pb:
    quality = ranked[(ranked["pe_ttm"] > 0) & (ranked["pb"] > 0)].copy()
    if quality.empty:
        quality = ranked.copy()
else:
    quality = ranked.copy()

print(f"\n{'='*70}")
print("  1000元资金分配建议（整数规划优化）")
print("  约束: 100股起买, 100股整数倍, 总额≤1000元")
print(f"{'='*70}")

# ── Greedy: pick top-scored stocks, 100 shares each, until budget exhausted ──
# Then add a "best single-stock" alternative

def greedy_allocate(candidates_df, budget=CAPITAL, min_shares=100):
    """Pick top stocks greedily: 100 shares each, until budget runs out."""
    plan = []
    remaining = budget
    for _, row in candidates_df.iterrows():
        price = row["price"]
        if pd.isna(price) or price <= 0:
            continue
        cost = min_shares * price
        if cost <= remaining:
            plan.append({
                "code": str(row.get("code", "")).zfill(6),
                "name": str(row.get("name", "")),
                "price": price,
                "shares": min_shares,
                "cost": cost,
                "pct_chg": row.get("pct_chg", 0),
                "turnover": row.get("turnover", 0),
                "pe_ttm": row.get("pe_ttm", np.nan) if "pe_ttm" in candidates_df.columns else np.nan,
                "score": row.get("score", 0),
            })
            remaining -= cost
        if len(plan) >= 5:
            break
    return plan, remaining

# Plan A: Top-scored (turnover-weighted)
plan_a, rem_a = greedy_allocate(quality)

# Plan B: Best value — sort by low price × score (cheaper good stocks first)
if "score" in quality.columns:
    quality_b = quality.copy()
    quality_b["value_score"] = quality_b["score"] / quality_b["price"].clip(lower=0.5)
    quality_b = quality_b.sort_values("value_score", ascending=False)
else:
    quality_b = quality
plan_b, rem_b = greedy_allocate(quality_b)

# Plan C: Equal-weighted bank/energy blue chips (safe)
blue_chips = quality[quality["name"].str.contains("银行|工|农|建|中行|招商|石油|石化|宝钢|神华|中煤|中铁|中冶|交行|光大|华夏|民生|浦发|平安|太保|人寿", na=False)]
if blue_chips.empty:
    blue_chips = quality
plan_c, rem_c = greedy_allocate(blue_chips)

print(f"\n  {'方案A (动量优先)':─^60}")
if plan_a:
    for i, a in enumerate(plan_a):
        pct = a["cost"]/CAPITAL*100
        print(f"  {i+1}. {a['code']} {a['name']:<8}  ￥{a['price']:.2f} x {a['shares']}股 = {a['cost']:.0f}元 ({pct:.0f}%)  涨跌:{a['pct_chg']:+.2f}%")
    print(f"  → 共{len(plan_a)}只, 投入{sum(a['cost'] for a in plan_a):.0f}元, 剩余{rem_a:.0f}元")
else:
    print("  无法构建组合")

print(f"\n  {'方案B (性价比优先)':─^60}")
if plan_b:
    for i, a in enumerate(plan_b):
        pct = a["cost"]/CAPITAL*100
        print(f"  {i+1}. {a['code']} {a['name']:<8}  ￥{a['price']:.2f} x {a['shares']}股 = {a['cost']:.0f}元 ({pct:.0f}%)  涨跌:{a['pct_chg']:+.2f}%")
    total_b = sum(a["cost"] for a in plan_b)
    if len(plan_b) >= 3:
        print(f"  → 共{len(plan_b)}只, 投入{total_b:.0f}元, 剩余{rem_b:.0f}元 [推荐: 分散度高]")
    else:
        print(f"  → 共{len(plan_b)}只, 投入{total_b:.0f}元, 剩余{rem_b:.0f}元")
else:
    print("  无法构建组合")

print(f"\n  {'方案C (蓝筹防御)':─^60}")
if plan_c:
    for i, a in enumerate(plan_c):
        pct = a["cost"]/CAPITAL*100
        print(f"  {i+1}. {a['code']} {a['name']:<8}  ￥{a['price']:.2f} x {a['shares']}股 = {a['cost']:.0f}元 ({pct:.0f}%)  涨跌:{a['pct_chg']:+.2f}%")
    total_c = sum(a["cost"] for a in plan_c)
    print(f"  → 共{len(plan_c)}只, 投入{total_c:.0f}元, 剩余{rem_c:.0f}元")
else:
    print("  无法构建组合")

print(f"\n[免责声明] 以上为量化模型自动筛选，仅供参考，不构成投资建议。")
print(f"   股市有风险，投资需谨慎。建议分散投资，设置止损。")
