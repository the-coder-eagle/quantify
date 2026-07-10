"""
Quantify v4 — Professional Semi-Automated Trading Terminal
core/: data, indicators, strategies, backtest, risk, regime
"""
import sys, os, json, warnings, re as _re
from datetime import date, datetime, time, timedelta
import pandas as pd, numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import (DataManager, market_snapshot, sma, ema, rsi, macd, bollinger_bands,
                   atr, ALL_STRATEGIES, analyze_stock, BacktestEngine, drawdown_series,
                   walk_forward, position_size, stop_loss_price, detect_regime,
                   regime_risk_multiplier, load_latest_weights, load_latest_params,
                   AIAgent)
from ai import AIClient
from news import fetch as fetch_news, sentiment as news_sentiment

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Quantify Pro", page_icon="", layout="wide")
st.markdown("""<style>
.card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px 18px;margin:6px 0;}
.card-buy{border-left:4px solid #ef4444}.card-sell{border-left:4px solid #22c55e}.card-hold{border-left:4px solid #9ca3af}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600}
.badge-buy{background:#fef2f2;color:#dc2626}.badge-sell{background:#f0fdf4;color:#16a34a}.badge-hold{background:#f3f4f6;color:#6b7280}
</style>""", unsafe_allow_html=True)

from config import app_capital, app_risk_pct, app_max_price, app_min_price, app_scan_default

# Runtime config — loaded from config/settings.yaml
CAPITAL = app_capital()
RISK_PCT = app_risk_pct()
DEFAULT_MAX_PRICE = app_max_price()
DEFAULT_MIN_PRICE = app_min_price()
DEFAULT_SCAN = app_scan_default()

MARKET_NAMES = {
    "茅台":"600519","贵州茅台":"600519","五粮液":"000858","宁德":"300750","宁德时代":"300750",
    "比亚迪":"002594","平安":"601318","中国平安":"601318","招商银行":"600036","招行":"600036",
    "美的":"000333","美的集团":"000333","格力":"000651","格力电器":"000651","伊利":"600887",
    "中国银行":"601988","中行":"601988","宝钢":"600019","宝钢股份":"600019",
}

# ============================================================
# Optimizer state (loaded at startup from latest results)
# ============================================================
_opt_params = load_latest_params() or {}
_opt_weights = load_latest_weights() or {}
_opt_date = "未优化"
# Find latest optimization date
import glob as _glob
_opt_dirs = sorted(_glob.glob("results/optimizer/20*"), reverse=True)
if _opt_dirs:
    _opt_date = os.path.basename(_opt_dirs[0])

# ============================================================
# Helpers
# ============================================================
def is_trading_now():
    now = datetime.now()
    return now.weekday() < 5 and time(9,30) <= now.time() <= time(15,0)

@st.cache_data(ttl=120 if is_trading_now() else 600)
def load_market(limit, min_price, max_price):
    return market_snapshot(limit, min_price, max_price)

@st.cache_data(ttl=120 if is_trading_now() else 600)
def load_history(code, days=250):
    dm = DataManager()
    end = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    return dm.get(code, start, end)

def analyze(code):
    """Full analysis: strategies + regime + position sizing.

    Uses optimizer results (strategy_overrides + weights) when available.
    """
    df = load_history(code)
    if isinstance(df, dict):
        df = df.get(code.zfill(6), pd.DataFrame())

    # Build strategy_overrides from optimizer results
    overrides = None
    weights = None
    if _opt_params:
        overrides = {}
        for sname, best in _opt_params.items():
            if "params" in best:
                overrides[sname] = best["params"]
    if _opt_weights:
        weights = _opt_weights

    info = analyze_stock(df, CAPITAL, RISK_PCT,
                         strategy_overrides=overrides,
                         strategy_weights=weights)
    if info is None:
        return None
    info["code"] = code
    info["df"] = df
    return info

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("Quantify Pro")
    st.caption("半自动量化交易决策系统")
    in_trade = is_trading_now()
    st.caption(f"{'🟢 交易中' if in_trade else '⚫ 已收盘'} | {date.today()}")

    with st.expander("设置", expanded=False):
        scan_n = st.slider("扫描数量", 100, 5000, DEFAULT_SCAN, 100)
        min_p = st.slider("最低股价", 1.0, 50.0, DEFAULT_MIN_PRICE, 0.5)
        max_p = st.slider("最高股价", 5.0, 200.0, DEFAULT_MAX_PRICE, 1.0,
                         help=f"过滤高价股。{CAPITAL:.0f}元资金最多买{CAPITAL/100:.0f}元/股（1手=100股）")
        ai_key = st.text_input("DeepSeek Key", type="password", placeholder="sk-...")

    if st.button("刷新数据", width="stretch"):
        st.cache_data.clear(); st.rerun()

    st.divider()
    st.caption(f"上次优化: {_opt_date}")
    st.caption(f"优化策略: {'✓' if _opt_params else '默认参数'}")
    st.caption("数据: 新浪财经 | core/模块化架构")

ai_client = None
if ai_key.strip():
    try:
        ai_client = AIClient(api_key=ai_key.strip())
    except Exception as e:
        logger.warning(f"AIClient 初始化失败: {e}")

# ============================================================
# LOAD
# ============================================================
with st.spinner("加载全市场数据..."):
    market = load_market(scan_n, min_p, max_p)

# ============================================================
# TABS
# ============================================================
t1, t2, t3, t4, t5, t6 = st.tabs([
    "全市场扫描", "个股分析", "AI决策助手", "策略验证", "交易信号", "追踪记录"
])

# ============================================================
# TAB 1
# ============================================================
with t1:
    if market.empty:
        st.warning("数据加载失败，点击侧栏「刷新数据」")
    else:
        if "pct_chg" in market.columns:
            up = (market["pct_chg"] > 0).sum(); dn = (market["pct_chg"] < 0).sum()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("上涨", f"{up}只"); c2.metric("下跌", f"{dn}只")
            c3.metric("平均涨跌", f"{market['pct_chg'].mean():+.2f}%")
            c4.metric("样本", f"{len(market)}只")

        kw = st.text_input("搜索", placeholder="代码/名称").strip()
        flt = st.selectbox("筛选", ["全部","涨幅>5%","涨幅>2%","跌幅>2%","跌幅>5%"])

        disp = market.copy()
        if kw:
            safe_kw = _re.escape(kw.upper())
            m = disp["code"].str.contains(safe_kw, na=False)
            if "name" in disp.columns:
                m |= disp["name"].str.contains(kw, na=False)
            disp = disp[m]
        if "pct_chg" in disp.columns:
            if ">5%" in flt: disp = disp[disp["pct_chg"] > 5]
            elif ">2%" in flt: disp = disp[disp["pct_chg"] > 2]
            elif "跌幅>2%" in flt: disp = disp[disp["pct_chg"] < -2]
            elif "跌幅>5" in flt: disp = disp[disp["pct_chg"] < -5]

        cols_show = []
        for cn, en in [("代码","code"),("名称","name"),("最新价","price"),("涨跌%","pct_chg"),
                        ("涨跌额","change"),("今开","open"),("最高","high"),("最低","low"),
                        ("昨收","prev_close"),("成交额(亿)","turnover")]:
            if en in disp.columns: cols_show.append((cn, en))

        tbl = disp[[en for _, en in cols_show]].copy()
        tbl.columns = [cn for cn, _ in cols_show]
        if "成交额(亿)" in tbl.columns: tbl["成交额(亿)"] = (tbl["成交额(亿)"] / 1e8).round(2)
        if "涨跌%" in tbl.columns: tbl["涨跌%"] = tbl["涨跌%"].round(2)

        def color_pct(v):
            try: fv = float(v); return "color:#ef4444;font-weight:600" if fv > 0 else ("color:#22c55e" if fv < 0 else "")
            except (ValueError, TypeError): return ""

        styled = tbl.style.map(color_pct, subset=["涨跌%"] if "涨跌%" in tbl.columns else [])
        fmt = {cn: st.column_config.NumberColumn(format=f) for cn, f in
               [("最新价","%.2f"),("涨跌%","%+.2f"),("涨跌额","%+.2f"),
                ("今开","%.2f"),("最高","%.2f"),("最低","%.2f"),("昨收","%.2f"),("成交额(亿)","%.2f")] if cn in tbl.columns}
        st.dataframe(styled, width="stretch", height=700, column_config=fmt)

# ============================================================
# TAB 2: Individual Stock Analysis
# ============================================================
with t2:
    code = st.text_input("股票代码", "600519").strip().zfill(6)
    if st.button("分析", type="primary") or st.session_state.get("t2c") == code:
        st.session_state["t2c"] = code
        info = analyze(code)
        if info is None:
            st.warning("数据不足（需60个交易日）")
        else:
            df = info["df"]; c = df["Close"]; price = info["price"]
            reg = info["regime"]

            # Header
            st.header(f"{code}")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("最新价", f"{price:.2f}")
            m2.metric("RSI(14)", f"{info['rsi']}")
            m3.metric("市场状态", reg["regime"].replace("_"," "))
            m4.metric("波动率(年)", f"{reg.get('vol_20',0)*100:.1f}%")
            m5.metric("趋势强度", f"{reg.get('trend_strength',0)*100:.1f}%")
            st.divider()

            left, right = st.columns([0.9, 1.5])
            with left:
                st.subheader("因子评分")
                if info.get("factor_score") is not None:
                    fs = info["factor_score"]
                    pct = info.get("percentile", 0.5)
                    color = "#ef4444" if fs > 0 else ("#22c55e" if fs < 0 else "#9ca3af")
                    st.metric("综合因子得分", f"{fs:.4f}", delta=f"全市场前{int(pct*100)}%")

                    # Show available factor values
                    st.divider()
                    st.caption("各因子组贡献（因子值 × IC权重）")
                else:
                    # Fallback: old strategy voting
                    st.subheader("策略投票")
                    for name, _2 in ALL_STRATEGIES:
                        s = info["details"].get(name, 0)
                        icon = {1:"🟢", -1:"🔴", 0:"⚪"}.get(s, "⚪")
                        label = {1:"买入", -1:"卖出", 0:"观望"}.get(s, "?")
                        st.write(f"{icon} {name}: {label}")

                badge = "badge-buy" if info["signal"] == "BUY" else ("badge-sell" if info["signal"] == "SELL" else "badge-hold")
                st.markdown(f"**综合: <span class='badge {badge}'>{info['signal']}</span> (置信度 {info['confidence']:.0%})**",
                            unsafe_allow_html=True)

                # Regime-aware risk adjustment
                adj_risk = RISK_PCT * regime_risk_multiplier(reg)
                if info["signal"] == "BUY" and info["shares"] > 0:
                    st.info(f"建议买入 {info['shares']} 股 | 止损 {info['stop']} | "
                            f"风险调整: {RISK_PCT*100:.0f}%→{adj_risk*100:.1f}% (状态: {reg['regime']})")

                # Regime info
                st.divider()
                st.caption(f"价格在 MA20 {reg.get('price_vs_ma20','?')} | "
                          f"波动变化: {reg.get('vol_change',1):.1f}x")

            with right:
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                                     row_heights=[0.55, 0.2, 0.25])
                fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
                                              close=df["Close"], name="K线",
                                              increasing_line_color="#ef4444", decreasing_line_color="#22c55e"),
                              row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=sma(c, 20), line=dict(color="#3b82f6", width=1.5), name="MA20"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=sma(c, 50), line=dict(color="#f59e0b", width=1.5), name="MA50"), row=1, col=1)
                bb = bollinger_bands(c, 20, 2)
                fig.add_trace(go.Scatter(x=df.index, y=bb["upper"], line=dict(color="#9ca3af", width=0.8, dash="dot"),
                                          showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=bb["lower"], fill="tonexty",
                                          fillcolor="rgba(156,163,175,0.08)",
                                          line=dict(color="#9ca3af", width=0.8, dash="dot"), showlegend=False), row=1, col=1)
                vc = ["#ef4444" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#22c55e" for i in range(len(df))]
                fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vc, showlegend=False), row=2, col=1)
                m = macd(c)
                fig.add_trace(go.Scatter(x=df.index, y=m["macd"], line=dict(color="#3b82f6", width=1), name="MACD"), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=m["signal"], line=dict(color="#f59e0b", width=1), name="Signal"), row=3, col=1)
                mcs = ["#ef4444" if v >= 0 else "#22c55e" for v in m["histogram"]]
                fig.add_trace(go.Bar(x=df.index, y=m["histogram"], marker_color=mcs, showlegend=False), row=3, col=1)
                fig.update_layout(height=550, xaxis_rangeslider_visible=False,
                                   legend=dict(orientation="h", yanchor="bottom", y=1.01),
                                   margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ============================================================
# TAB 3: AI Decision Assistant
# ============================================================
with t3:
    st.header("AI 决策助手")
    if "chat" not in st.session_state: st.session_state.chat = []

    # Build market context
    ctx = ""
    up = dn = 0
    top5 = ""
    if not market.empty and "pct_chg" in market.columns:
        up = int((market["pct_chg"] > 0).sum()); dn = int((market["pct_chg"] < 0).sum())
        top5 = ", ".join(f"{r['code']}{r.get('name','')}({r.get('pct_chg',0):+.1f}%)"
                         for _, r in market.head(5).iterrows())
        ctx = f"全市场{len(market)}只 | 上涨{up}下跌{dn} | 成交额前5: {top5}"

    with st.expander("AI 可见数据", expanded=False):
        st.code(ctx or "等待行情数据...", language="")

    msg = st.chat_input("例如: 分析茅台、推荐低价股、解释RSI、我的仓位合理吗...")
    if msg:
        st.session_state.chat.append({"role": "user", "content": msg})
        if ai_client:
            # Initialize AIAgent with tool access
            agent = AIAgent(ai_client)

            # Detect stock mentions for extra context
            extra = ""
            import re
            codes_found = re.findall(r'\b(\d{6})\b', msg)
            for name, cd in MARKET_NAMES.items():
                if name in msg: codes_found.append(cd)
            for cd in list(set(codes_found))[:3]:
                info = analyze(cd)
                if info:
                    extra += f"\n{cd}: ¥{info['price']} RSI{info['rsi']} {info['signal']}(买{info['buys']}卖{info['sells']}) 状态:{info['regime']['regime']}"

            market_ctx = (f"全市场{len(market)}只 | 上涨{up}下跌{dn} | "
                         f"资金{CAPITAL:.0f}元 | 最高可买{CAPITAL/100:.0f}元/股 | "
                         f"成交额前5: {top5}" if not market.empty else "等待数据...")

            with st.spinner("AI 分析中（可使用回测/参数调整工具）..."):
                resp = agent.chat(msg, market_context=market_ctx, stock_info=extra)
        else:
            codes_found = [c for c in ["600519","000858","300750","000001","002594"] if c in msg]
            if codes_found:
                info = analyze(codes_found[0])
                resp = f"离线分析 {codes_found[0]}: ¥{info['price']} RSI{info['rsi']} {info['signal']}信号" if info else "数据不足"
            else:
                resp = "输入股票代码获取分析，或输入 DeepSeek Key 启用 AI。"
        st.session_state.chat.append({"role": "assistant", "content": resp})

    for m in st.session_state.chat:
        st.chat_message(m["role"]).write(m["content"])

# ============================================================
# TAB 4: Strategy Validation
# ============================================================
with t4:
    st.header("策略验证")
    c1, c2 = st.columns([1, 1])
    with c1: bt_code = st.text_input("代码", "600519", key="btc").strip().zfill(6)
    with c2: bt_method = st.selectbox("验证方式", ["标准回测", "Walk-Forward验证"])

    if st.button("运行验证", type="primary"):
        df = load_history(bt_code, 500)
        if len(df) >= 100:
            strat = ALL_STRATEGIES[0][1]  # MA Trend as default
            engine = BacktestEngine(strat, long_only=True)
            result = engine.run(df, bt_code)
            s = result.summary()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sharpe", f"{s['Sharpe']:.3f}")
            c2.metric("最大回撤", s["Max DD"])
            c3.metric("胜率", s["Win Rate"])
            c4.metric("交易次数", str(s["Trades"]))

            if bt_method == "Walk-Forward验证":
                st.info("Walk-Forward: 在训练集上优化参数，在测试集上验证。如果OOS表现远差于IS，策略过拟合。")
                try:
                    wf = walk_forward(strat, df, train_months=12, test_months=3)
                    if wf:
                        wf_ret = (wf["equity"].iloc[-1] - 1) * 100
                        st.metric(f"WF验证 ({wf['n_windows']}窗口)", f"{wf_ret:+.1f}%")
                        st.caption("正值=策略在未知数据上盈利，负值=可能过拟合")
                except Exception as e:
                    st.warning(f"Walk-Forward未完成: {e}")

            # Equity curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity.values,
                                      name="策略", line=dict(color="#3b82f6", width=2)))
            fig.add_trace(go.Scatter(x=result.benchmark_equity.index, y=result.benchmark_equity.values,
                                      name="买入持有", line=dict(color="#d1d5db", width=1, dash="dash")))
            fig.add_hline(y=1.0, line_dash="dot")
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

            dd = drawdown_series(result.equity) * 100
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy",
                                       line=dict(color="#ef4444", width=1.5)))
            fig2.update_layout(height=200, yaxis_title="回撤%", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

# ============================================================
# TAB 5: Trading Signals
# ============================================================
with t5:
    st.header("交易信号")

    if market.empty:
        st.warning("请先加载市场数据")
    else:
        # Use market data for scan candidates (top N by turnover)
        codes_all = market["code"].dropna().astype(str).str.zfill(6).tolist()
        scan_count = st.slider("扫描数量", 20, min(300, len(codes_all)), 50, 10,
                               help="从全市场成交额最高的股票中扫描")
        codes_scan = codes_all[:scan_count]

        c_left, c_right = st.columns([1, 3])
        with c_left:
            do_scan = st.button("扫描信号", type="primary", width="stretch")
        with c_right:
            st.caption(f"将从成交额 TOP {scan_count} 中筛选买卖信号")

        if do_scan:
            results = []
            bar = st.progress(0)
            total = len(codes_scan)
            for i, cd in enumerate(codes_scan):
                info = analyze(cd)
                if info and info["signal"] != "HOLD" and info["confidence"] >= 0.4:
                    adj_risk = RISK_PCT * regime_risk_multiplier(info["regime"])
                    sh = position_size(CAPITAL, info["price"], adj_risk, 0.05)
                    results.append({"code": cd, "signal": info["signal"],
                                    "confidence": f"{info['confidence']:.0%}",
                                    "price": info["price"], "rsi": info["rsi"],
                                    "shares": sh, "stop": info["stop"],
                                    "regime": info["regime"]["regime"],
                                    "buys": info["buys"], "sells": info["sells"]})
                bar.progress((i + 1) / total)
            bar.empty()

            if results:
                buys = [r for r in results if r["signal"] == "BUY"]
                sells = [r for r in results if r["signal"] == "SELL"]
                st.success(f"扫描 {total} 只 → 🔴买入 {len(buys)} | 🟢卖出 {len(sells)}")

                results.sort(key=lambda x: float(x["confidence"].replace("%", "")), reverse=True)
                for a in results:
                    card_cls = "card-buy" if a["signal"] == "BUY" else "card-sell"
                    badge_cls = "badge-buy" if a["signal"] == "BUY" else "badge-sell"
                    emoji = "🔴" if a["signal"] == "BUY" else "🟢"
                    st.markdown(f"""<div class="card {card_cls}">
                    <b>{emoji} {a['code']}</b> <span class="badge {badge_cls}">{a['signal']}</span>
                    置信度{a['confidence']} | ¥{a['price']:.2f} | RSI{a['rsi']}
                    | 状态:{a['regime']} | 买{a['buys']}票卖{a['sells']}票
                    </div>""", unsafe_allow_html=True)
                    if a["signal"] == "BUY" and a["shares"] > 0:
                        st.success(f"建议 {a['shares']} 股 (约{a['shares']*a['price']:,.0f}元) | 止损 ¥{a['stop']}")
            else:
                st.info("无明确信号。市场可能处于震荡或观望状态。")

# ============================================================
# TAB 6: Tracking
# ============================================================
TRACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache", "tracking.json")
with t6:
    st.header("决策追踪")
    st.caption("记录每次决策 → 定期回顾 → 持续改进")

    # Load
    preds = []
    if os.path.exists(TRACK_FILE):
        try:
            with open(TRACK_FILE, "r", encoding="utf-8") as f:
                preds = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"跟踪文件读取失败: {e}")

    # Add
    cc1, cc2, cc3 = st.columns(3)
    with cc1: t_code = st.text_input("股票代码", key="tcd").strip().zfill(6)
    with cc2: t_advice = st.selectbox("方向", ["买入", "卖出", "观望"], key="tadv")
    with cc3: t_price = st.number_input("价格", 0.0, 10000.0, 0.0, 0.01, key="tpr")

    t_note = st.text_input("理由", placeholder="为什么做这个决定？").strip()
    if st.button("记录", type="primary"):
        preds.append({"date": date.today().isoformat(), "code": t_code,
                       "advice": t_advice, "price": t_price, "note": t_note})
        with open(TRACK_FILE, "w", encoding="utf-8") as f: json.dump(preds, f, ensure_ascii=False, indent=2)
        st.success("已记录"); st.rerun()

    # Check predictions
    if preds:
        checked = 0
        for p in preds:
            if p.get("checked"): continue
            try:
                df = load_history(p["code"], 30)
                if len(df) > 0:
                    cur = df["Close"].iloc[-1]
                    p["current"] = round(cur, 2)
                    p["pct"] = round((cur / p["price"] - 1) * 100, 2)
                    if p["advice"] == "买入": p["result"] = "✅" if p["pct"] > 0 else "❌"
                    elif p["advice"] == "卖出": p["result"] = "✅" if p["pct"] < 0 else "❌"
                    p["checked"] = True; checked += 1
            except Exception as e:
                logger.warning(f"预测验证失败 {p.get('code','?')}: {e}")
        if checked > 0:
            with open(TRACK_FILE, "w", encoding="utf-8") as f: json.dump(preds, f, ensure_ascii=False, indent=2)

        right = sum(1 for p in preds if p.get("result") == "✅")
        total_checked = sum(1 for p in preds if p.get("checked"))
        if total_checked > 0:
            c1, c2 = st.columns(2)
            c1.metric("已验证", str(total_checked))
            c2.metric("准确率", f"{right / total_checked * 100:.0f}%")

        for p in reversed(preds):
            days = (date.today() - date.fromisoformat(p["date"])).days
            emoji = "🔴" if p["advice"] == "买入" else ("🟢" if p["advice"] == "卖出" else "⚪")
            st.markdown(f"{emoji} **{p['code']}** {p['advice']} @{p['price']:.2f} ({days}天前) "
                        f"{'→ 现价¥'+str(p.get('current',''))+' ('+str(p.get('pct',''))+'%)' if p.get('current') else '→ 待验证...'}"
                        f" {' '+p.get('result','')}")

st.divider()
st.caption("Quantify Pro v4 | core/ modular | 半自动决策系统 | 仅供学习")
