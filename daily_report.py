"""
Daily Market Report Generator / 每日复盘报告生成器

Run after market close (15:00+) to generate an analysis report.
Saves to results/reports/YYYY-MM-DD.md

Usage:
    python daily_report.py              # Generate today's report
    python daily_report.py 2026-07-03   # Generate for specific date
"""
import sys, os, json
from datetime import date, datetime, timedelta

import pandas as pd
import numpy as np
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import DataManager, ALL_STRATEGIES, analyze_stock

# ============================================================
# CONFIG
# ============================================================
WATCH_CODES = [
    "000001","600036","600519","000858","300750","000333","600900",
    "600276","002594","000725","601318","600030","000651","600809",
    "300760","600031","002475","603259","601012","688981",
]
STOCK_NAMES = {
    "000001":"平安银行","600036":"招商银行","600519":"贵州茅台","000858":"五粮液",
    "300750":"宁德时代","000333":"美的集团","600900":"长江电力","600276":"恒瑞医药",
    "002594":"比亚迪","000725":"京东方A","601318":"中国平安","600030":"中信证券",
    "000651":"格力电器","600809":"山西汾酒","300760":"迈瑞医疗","600031":"三一重工",
    "002475":"立讯精密","603259":"药明康德","601012":"隆基绿能","688981":"中芯国际",
}
from config import research_capital, app_risk_pct
CAPITAL = research_capital()  # 研究报告使用较大的模拟资金
RISK_PCT = app_risk_pct()
STRATEGIES = ALL_STRATEGIES
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "reports")


def analyze_one(df, code=""):
    """Run all strategies on one stock."""
    return analyze_stock(df, CAPITAL, RISK_PCT, code=code)


def generate_report(report_date=None):
    """Generate a full daily report."""
    if report_date is None:
        report_date = date.today()
    elif isinstance(report_date, str):
        report_date = date.fromisoformat(report_date)

    dm = DataManager()
    end_str = report_date.strftime("%Y-%m-%d")
    start_str = (report_date - timedelta(days=365)).strftime("%Y-%m-%d")

    results = []
    errors = []

    print(f"Generating report for {report_date}...")
    for i, code in enumerate(WATCH_CODES):
        try:
            df = dm.get(code, start_str, end_str)
            if isinstance(df, pd.DataFrame) and len(df) >= 60:
                info = analyze_one(df, code)
                info["code"] = code
                info["name"] = STOCK_NAMES.get(code, "?")
                results.append(info)
            else:
                errors.append(f"{code}: data insufficient")
                logger.warning(f"{code}: 数据不足 ({len(df) if isinstance(df, pd.DataFrame) else 'N/A'} 行)")
        except Exception as e:
            errors.append(f"{code}: {e}")
            logger.error(f"{code}: {e}")
        print(f"  [{i+1}/{len(WATCH_CODES)}] {code} OK")

    # Sort: BUY first, then SELL, then HOLD
    sort_order = {"BUY": 0, "SELL": 1, "HOLD": 2}
    results.sort(key=lambda r: (sort_order.get(r["signal"], 3), -r["confidence"]))

    # Build report
    lines = []
    lines.append(f"# Quantify 每日复盘报告")
    lines.append(f"## {report_date.isoformat()} ({['周一','周二','周三','周四','周五','周六','周日'][report_date.weekday()]})")
    lines.append("")

    # Summary stats
    buys = [r for r in results if r["signal"] == "BUY"]
    sells = [r for r in results if r["signal"] == "SELL"]
    holds = [r for r in results if r["signal"] == "HOLD"]
    lines.append(f"## 概览")
    lines.append(f"- 扫描股票: {len(results)} 只")
    lines.append(f"- 买入信号: {len(buys)} 只 (置信度>=60%)")
    lines.append(f"- 卖出信号: {len(sells)} 只")
    lines.append(f"- 观望: {len(holds)} 只")
    if errors:
        lines.append(f"- 数据异常: {len(errors)} 只")
    lines.append("")

    # Buy signals
    if buys:
        lines.append("## 买入信号")
        lines.append("| 代码 | 名称 | 现价 | RSI | MA20 | MA50 | 置信度 | 建议股数 | 止损价 |")
        lines.append("|------|------|------|-----|------|------|--------|----------|--------|")
        for r in [x for x in buys if x["confidence"] >= 0.4]:
            lines.append(f"| {r['code']} | {r['name']} | {r['price']} | {r['rsi']} | "
                        f"{r['ma20']} | {r['ma50']} | {r['confidence']:.0%} | "
                        f"{r['shares'] or '-'} | {r['stop_loss'] or '-'} |")

    # Sell signals
    if sells:
        lines.append("")
        lines.append("## 卖出信号")
        lines.append("| 代码 | 名称 | 现价 | RSI | 置信度 |")
        lines.append("|------|------|------|-----|--------|")
        for r in sells:
            lines.append(f"| {r['code']} | {r['name']} | {r['price']} | {r['rsi']} | {r['confidence']:.0%} |")

    # Detailed strategy votes
    lines.append("")
    lines.append("## 详细投票")
    lines.append("| 代码 | 均线 | MACD | 动量 | 量确认 | 反转 | 结论 |")
    lines.append("|------|------|------|------|--------|------|------|")
    for r in results:
        d = r["details"]
        def v(k): return "买" if d.get(k,0)==1 else ("卖" if d.get(k,0)==-1 else "-")
        lines.append(f"| {r['code']} | {v('趋势均线(20/50)')} | {v('MACD交叉(12/26/9)')} | "
                    f"{v('动量(60日)')} | {v('量确认(20/50)')} | {v('反转(BB+RSI)')} | "
                    f"**{r['signal']}** |")

    if errors:
        lines.append("")
        lines.append("## 数据异常")
        for e in errors:
            lines.append(f"- {e}")

    lines.append("")
    lines.append("---")
    lines.append(f"*报告由 Quantify 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("*⚠️ 仅供学习参考，不构成投资建议*")

    # Save to file
    os.makedirs(REPORT_DIR, exist_ok=True)
    filename = f"{report_date.isoformat()}.md"
    filepath = os.path.join(REPORT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also save JSON for the app to display
    json_path = os.path.join(REPORT_DIR, f"{report_date.isoformat()}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "date": report_date.isoformat(),
            "summary": {"total": len(results), "buys": len(buys),
                        "sells": len(sells), "holds": len(holds)},
            "buy_signals": [{"code": r["code"], "name": r["name"], "price": r["price"],
                             "signal": r["signal"], "confidence": r["confidence"],
                             "shares": r["shares"], "stop_loss": r["stop_loss"],
                             "rsi": r["rsi"]} for r in results if r["signal"] == "BUY"],
            "sell_signals": [{"code": r["code"], "name": r["name"], "price": r["price"],
                              "confidence": r["confidence"]} for r in results if r["signal"] == "SELL"],
            "all_results": [{"code": r["code"], "name": r["name"], "price": r["price"],
                             "signal": r["signal"], "confidence": r["confidence"],
                             "rsi": r["rsi"]} for r in results],
        }, f, ensure_ascii=False, indent=2)

    print(f"\nReport saved: {filepath}")
    print(f"JSON saved:  {json_path}")
    logger.info(f"报告已生成: {filepath}")
    return filepath


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    path = generate_report(target)
    print(f"\nDone: {path}")
