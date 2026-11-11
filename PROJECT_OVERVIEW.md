# Quantify Pro v4 — 半自动 A 股量化交易系统

> 项目详细介绍 · 基于 2026-11-11 对归档源码的静态梳理（最后提交 `fb455e9`，2026-07-11）
>
> 项目目录：`quantify/`

---

## 0. 一句话概括

Quantify 是一个面向 A 股、以 **1200 元小额资金**为约束条件的 **半自动量化交易决策系统**：收盘后由调度器自动做参数寻优与多因子横截面排名，次日通过 Streamlit 界面给出「买/卖/观望 + 建议股数 + 止损价」，**最终由人确认并自行下单**，系统不连接任何券商接口。

它同时是一个量化学习项目：从数据获取、技术指标、策略回测、风险控制，一路搭到多因子、IC 加权、AI 代理和自动优化闭环。

---

## 1. 项目档案

| 项目 | 内容 |
|---|---|
| 名称 | Quantify / Quantify Pro v4 |
| 定位 | 半自动 A 股量化交易决策终端（学习 + 研究用途） |
| 项目目录 | `quantify/` |
| 归档日期 | 2026-07-20 |
| 当前状态 | **ARCHIVED**（pipeline 全部 FROZEN，所有 agent 禁用） |
| 技术栈 | Python 3.10+ / Streamlit / Plotly / pandas / numpy / akshare / DeepSeek API |
| 数据源 | 东方财富（主）、新浪财经（备）、百度（估值补全） |
| 代码规模 | 约 6,176 行 Python（含测试），14 个测试文件 |
| 测试基线 | 初始提交标注 103 tests passing；`.pytest_cache` 中 `lastfailed` 为空 |
| Git 历史 | 23 个提交，2026-07-10 11:16 ～ 2026-07-11 12:35 |
| 虚拟环境 | 归档时未保留 `.venv` |
| 数据缓存 | `data_cache/daily/` 中保留 **781 个** parquet 日线文件 |

---

## 2. 项目目标与「半自动」的含义

### 2.1 用户约束

配置文件 `config/settings.yaml` 直接写明了设计前提：

```yaml
app:
  capital: 1200.0      # 大学生小额资金
  risk_pct: 0.02       # 单笔风险 2%
  max_price: 12.0      # 1200 元 / 100 股 = 最高 12 元/股
  min_price: 2.0       # 过滤仙股
  scan_default: 500
```

A 股最小交易单位是 1 手 = 100 股，所以 1200 元资金只能买单价 ≤ 12 元的股票。系统的股票池、仓位计算、扫描器都围绕这个硬约束设计。

### 2.2 「半自动」的三层含义

1. **自动计算**：收盘后自动拉数据、跑参数扫描、计算因子、生成排名。
2. **自动建议**：次日界面自动给出 BUY / SELL / HOLD、置信度、建议股数、止损价。
3. **人工执行**：没有券商下单 API，没有实盘账户，最终决策和下单由用户自己完成。

项目里反复出现「仅供学习参考，不构成投资建议」的免责声明。

---

## 3. 系统全景

```text
┌─────────────────────── 收盘后：离线计算层（scheduler.py）───────────────────────┐
│                                                                                 │
│  akshare / 东方财富 / 新浪                                                       │
│        │                                                                        │
│        ▼                                                                        │
│  行情快照 market_snapshot()                                                      │
│        │                                                                        │
│        ├──► 股票池过滤 filter_universe()  （去 ST/退市、价格、流动性、市值）      │
│        │                                                                        │
│        ├──► 参数网格扫描 ParamScanner     → results/optimizer/<日期>/best_params  │
│        │        └──► 集成权重 EnsembleWeight → weights.json                      │
│        │                                                                        │
│        └──► 因子流水线 run_factor_pipeline()                                     │
│                 ├── 30+ 子因子 compute_all_factors()                             │
│                 ├── 横截面 Z-score / 行业中性                                     │
│                 ├── Rank IC 加权                                                  │
│                 └── 排名快照 → data_cache/rankings/<日期>.parquet                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼ 次日
┌─────────────────────── 盘中/盘后：在线查询层（app.py）──────────────────────────┐
│                                                                                 │
│  全市场扫描 │ 个股分析 │ AI 决策助手 │ 策略验证 │ 交易信号 │ 决策追踪            │
│                                                                                 │
│  因子得分 + 策略投票 + 市场状态 + 风险调整                                        │
│        │                                                                        │
│        ▼                                                                        │
│  BUY / SELL / HOLD + 置信度 + 建议股数 + 止损价                                   │
│        │                                                                        │
│        ▼                                                                        │
│  用户人工确认 → 自行下单（系统不接券商）                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

核心设计原则是 **离线计算、在线查询**：耗时的因子计算和参数扫描放在收盘后，第二天界面只读结果，响应快。

---

## 4. 目录结构与模块职责

```text
quantify/
├── app.py                  # Streamlit 主界面（6 个 Tab）
├── ai.py                   # DeepSeek API 客户端（OpenAI 兼容协议）
├── news.py                 # 东方财富新闻抓取 + 情绪分析
├── scheduler.py            # 收盘后自动优化 + 因子流水线
├── daily_report.py         # 20 只自选股的每日复盘报告
├── quick_scan.py           # 1000 元小资金低价股扫描与分配方案
├── start.bat               # 一键启动脚本（清端口、装依赖、开浏览器）
├── requirements.txt        # 运行依赖
├── config/
│   ├── settings.yaml       # 全部配置集中于此
│   └── __init__.py         # 带类型的配置读取器
├── core/
│   ├── data.py             # 数据获取、清洗、Parquet 缓存
│   ├── indicators.py       # SMA/EMA/RSI/MACD/BOLL/ATR
│   ├── strategies.py       # 5 个传统技术策略（已标记 deprecated）
│   ├── analyzer.py         # 单股综合分析：投票 + 状态 + 仓位
│   ├── backtest.py         # 回测引擎、指标、Walk-Forward、OOS 检查
│   ├── risk.py             # 仓位、凯利、波动率目标、多级止损
│   ├── regime.py           # 市场状态识别与风险乘数
│   ├── universe.py         # 可交易股票池过滤
│   ├── factors.py          # 11 类 30+ 子因子计算引擎
│   ├── cross_section.py    # Z-score、行业中性、Rank IC、IC 加权排名
│   ├── optimizer.py        # 交易日历、参数扫描、集成权重
│   └── ai_agent.py         # 带沙箱工具调用的 AI 代理
├── tests/                  # 14 个测试文件
├── data_cache/daily/       # 781 个日线 parquet 缓存
├── results/
│   ├── reports/            # 2026-07-05 的日报 md/json
│   ├── optimizer/          # ai_overrides.json（历史扫描结果目录已不在）
│   └── audit/              # AI 写操作审计日志目录（当前为空）
└── docs/superpowers/       # 自动优化器与多因子升级的设计/实施文档
```

### 模块行数一览

| 文件 | 行数 | 职责 |
|---|---:|---|
| `core/data.py` | 726 | 数据获取与缓存 |
| `app.py` | 531 | Streamlit 六页界面 |
| `core/ai_agent.py` | 462 | AI 工具代理与安全边界 |
| `core/optimizer.py` | 419 | 参数扫描与集成权重 |
| `core/factors.py` | 300 | 多因子计算 |
| `scheduler.py` | 262 | 收盘后调度 |
| `core/cross_section.py` | 242 | 横截面排名与 IC 加权 |
| `quick_scan.py` | 221 | 小资金扫描 |
| `core/backtest.py` | 181 | 回测引擎 |
| `daily_report.py` | 178 | 每日报告 |
| `core/analyzer.py` | 147 | 综合分析器 |
| `core/strategies.py` | 113 | 5 个传统策略 |
| `core/regime.py` | 112 | 市场状态 |
| `news.py` | 101 | 新闻与情绪 |
| `core/indicators.py` | 57 | 技术指标 |
| `core/universe.py` | 53 | 股票池过滤 |
| `core/risk.py` | 52 | 风险管理 |
| `config/__init__.py` | 62 | 配置加载 |
| `ai.py` | 31 | DeepSeek 客户端 |

---

## 5. 核心模块详解

### 5.1 数据层 `core/data.py`

#### 行情快照 `market_snapshot(limit, min_price, max_price)`

- **主源：东方财富** `push2` 接口。字段更全，包含 `pe_ttm`、`pb`、`total_mv`、`circ_mv`、`turnover_rate` 等。
- **备源：新浪财经** 分页接口。每页 80 条，按成交额降序，最多 25 页；请求间 `sleep(0.6 + jitter)`；遇到 HTTP 456（限流）立即停止，不无意义重试。
- 返回的 DataFrame 通过 `attrs["source"]` 标记数据来源，调用方可以知道哪些字段缺失。
- 新浪源缺少 PE/PB/市值，`enrich_valuation()` 会用百度估值接口并发补齐候选股（默认前 30 只、8 线程）。

#### 历史日线 `DataManager.get(codes, start, end, refresh)`

- 使用 akshare `stock_zh_a_daily`（新浪），失败后回退 `stock_zh_a_hist`（东方财富）。
- 统一为 `Date / Open / High / Low / Close / Volume`。
- **前复权 `qfq`**，避免分红拆股扭曲回测。
- 本地 Parquet 缓存，24 小时内直接读；过期后重新拉取并与旧数据合并、按索引去重。
- 支持 ETF（`51/56/58/15/159` 开头），有独立的东方财富/新浪字段标准化函数。
- 股票代码严格校验为 6 位数字。

#### A 股特色数据接口

| 函数 | 内容 |
|---|---|
| `fetch_limit_up_pool()` | 当日涨停池 |
| `fetch_north_bound_individual(code)` | 北向资金个股持股 |
| `fetch_north_bound_history()` | 北向资金历史 |
| `fetch_margin_data(code)` | 融资融券余额、融资买入 |
| `fetch_financial_indicators(code)` | ROE、毛利率、资产负债率、EPS（取最新一期） |
| `fetch_fund_flow(code)` | 主力/超大单净流入 |
| `fetch_industry_info(code)` | 行业分类 |
| `fetch_valuation_baidu(code)` | 百度 PE/PB 补全 |

所有接口都遵循同一个原则：**失败返回 `None` 或空，不阻塞主流程**。

#### 股票池过滤 `core/universe.py`

排除 ST / *ST / 退市股、价格区间外、成交额低于阈值、市值过小的股票，把全市场压缩到「买得起且能交易」的候选集。

---

### 5.2 技术指标与策略

#### 指标 `core/indicators.py`

`SMA`、`EMA`、`RSI(14)`、`MACD(12/26/9)`、`Bollinger Bands(20, 2)`、`ATR(14)`，全部向量化实现。

#### 5 个传统策略 `core/strategies.py`

| 策略 | 默认参数 | 逻辑 |
|---|---|---|
| `MATrend` 趋势-均线 | fast=20, slow=50, ma_type=sma | 快线上穿慢线做多，下穿做空 |
| `MACDTrend` 趋势-MACD | 12/26/9 | MACD 与信号线金叉/死叉 |
| `Momentum` 动量 | lookback=60 | 60 日收益 > 阈值做多 |
| `VolumeConfirm` 量确认 | 20/50, vol_mult=1.2 | 均线信号必须得到成交量放大确认 |
| `MeanReversion` 反转 | BB(20,2) + RSI(14) | 跌破下轨且 RSI<30 买入，反之卖出 |

模块顶部已标注 **deprecated**：新代码应使用 `core.factors.compute_technical_factors()`。但这些策略仍被保留用于「策略投票」、参数扫描和回测验证。

`ALL_STRATEGIES` 是默认实例列表，`ALL_STRATEGY_CLASSES` 是「名称 → 类」映射，供优化器用不同参数重新构造实例。

---

### 5.3 回测引擎 `core/backtest.py`

#### 真实性约束

回测不是简单的「信号 × 收益」，而是尽量贴近 A 股现实：

| 约束 | 实现 |
|---|---|
| **T+1** | 信号 `shift(1)` 后才变成持仓，当天信号不能当天成交 |
| **涨跌停** | 日收益裁剪到 ±10% |
| **佣金** | 万三（0.03%），最低 5 元 |
| **印花税** | 千一（0.1%），仅卖出 |
| **滑点** | 0.1% |
| **整手** | 仓位计算按 100 股向下取整 |
| **只做多** | `long_only=True`，空头信号被裁剪 |

#### 回测指标

`BacktestResult.summary()` 输出：

- 总收益、基准收益（买入持有）
- CAGR、年化波动率
- Sharpe、Sortino、Calmar
- 最大回撤
- 胜率、盈亏比（Profit Factor）
- 交易次数、总交易成本

#### 防过拟合工具

- `walk_forward()`：滚动训练/测试。默认训练 24 个月、测试 6 个月；界面里用 12/3 个月。每个窗口在训练集上网格搜索最优参数，再在测试集上验证，最后拼接 OOS 净值。
- `oos_quality_check()`：按 70/30 切分样本内/样本外，比较 Sharpe 退化程度，给出 `OK / WARN / FAIL / SKIP` 裁决。WARN 表示 OOS Sharpe 流失超过样本内的 50%。

这两点是项目里比较专业的部分：**先怀疑过拟合，再相信回测**。

---

### 5.4 风险管理与市场状态

#### 仓位与止损 `core/risk.py`

| 函数 | 作用 |
|---|---|
| `position_size()` | 固定比例风险：`equity × risk_pct / stop_pct`，按 100 股取整 |
| `kelly_fraction()` | 半凯利公式，上限 25% |
| `volatility_target_shares()` | 按近期波动率反向缩放仓位 |
| `stop_loss_price()` | ATR 止损价 |
| `check_stops()` | 固定止损、追踪止损、ATR 止损、时间止损四合一 |

其中凯利、波动率目标、`check_stops` 已实现并有测试，但**没有接入主分析流程**（见第 10 节）。

#### 市场状态 `core/regime.py`

把市场分成 5 类：

| 状态 | 判定 |
|---|---|
| `trending_up` | 价格 > MA20 > MA50（更长数据还要求 > MA200），且 20 日收益 > 0 |
| `trending_down` | 价格 < MA20 < MA50 < MA200，且 20 日收益 < 0 |
| `high_volatility` | 20 日波动率 > 1.5 × 60 日波动率 |
| `low_volatility` | 20 日波动率 < 0.7 × 60 日波动率，且 \|20 日收益\| < 3% |
| `ranging` | 以上都不满足 |

风险乘数：

- 高波动：仓位 × 0.5
- 低波动：仓位 × 1.2
- 波动率变化 > 1.3 倍：仓位 × 0.7

界面会把「风险调整前 2% → 调整后 1.x%」直接显示出来。

---

### 5.5 多因子引擎 `core/factors.py`

每个股票计算 11 组、30+ 个子因子。无法计算的值返回 `None`，在排名时当作 NaN 处理。

| 因子组 | 子因子 | 频率 | 来源 |
|---|---|---|---|
| 价值 Value | `ep_ratio`、`bp_ratio`、`ps_proxy` | 日频 | 快照 PE/PB/市值 |
| 动量 Momentum | `mom_1m`、`mom_3m`、`mom_6m`、`mom_12m`、`mom_1m_ex5d` | 日频 | OHLCV |
| 质量 Quality | `roe`、`gross_margin`、`debt_ratio`、`eps` | 季频 | 财报 |
| 波动 Volatility | `vol_20d`、`vol_60d`、`vol_ratio` | 日频 | OHLCV |
| 规模 Size | `ln_circ_mv`、`ln_total_mv` | 日频 | 快照市值 |
| 北向 North-bound | `north_hold_pct`、`north_hold_shares`、`north_hold_mv` | 日频 | 北向持股 |
| 行业动量 Sector | 设计中（当前为空实现） | 日频 | 板块指数 |
| 技术 Technical | `rsi_deviation`、`macd_hist`、`macd_above_signal`、`bb_position`、`bb_width`、`ma20_slope`、`volume_ratio` | 日频 | OHLCV |
| 情绪 Sentiment | `news_sentiment`、`sentiment_confidence` | 按需 | 新闻 + AI |
| 融资融券 Margin | `margin_balance`、`margin_buy` | 日频 | 两融 |
| 资金流 Fund Flow | `main_net_inflow`、`super_large_net_inflow` | 日频 | 资金流 |

模块还定义了因子新鲜度 `FRESHNESS`，用于在 IC 加权时对过期因子降权（例如财报因子按季度更新）。

---

### 5.6 横截面排名与 IC 加权 `core/cross_section.py`

这是 v4 升级的核心。对全市场股票做横截面比较，而不是单股看指标。

处理流程：

1. **去极值 + Z-score**：每个因子先按 1%/99% 分位缩尾，再标准化。
2. **行业中性**：减去同行业均值，避免「行业整体高估」被误判为个股因子强。
3. **Rank IC**：因子值与未来收益的 Spearman 秩相关；少于 10 个共同样本返回 0。
4. **IC 加权打分**：
   - 原始权重 = `|IC|`，下限 `min_weight=0.03`
   - 过期因子权重减半
   - softmax（温度 0.5）归一化
   - 个股得分 = Σ(权重 × 因子 Z 值)，Z 值裁剪到 `[-3, 3]`，防止单因子爆炸
5. **输出排名**：`code`、`score`、`percentile`，按得分降序，保存为每日 parquet 快照。

`load_latest_ranking()` 供 `analyzer.py` 在线读取最近一份排名。

> 注意：IC 更新机制目前只实现了「计算函数」和「默认 IC 表」，调度器并不会用未来收益回算并迭代 IC。详见第 10 节。

---

### 5.7 自动优化器 `core/optimizer.py`

包含三块：

#### TradingCalendar 交易日历

- 周一到周五
- 当前时间 ≥ 15:00
- 当天还没有跑过（结果目录不存在）
- 如果 akshare 能拉到交易日历，则进一步排除节假日
- `--force` 可跳过检查

#### ParamScanner 参数扫描

- 对每个策略的 `param_grid` 做笛卡尔积网格搜索
- 每个组合跑一次回测
- 综合评分：

```text
score = 0.4 × max(Sharpe, -2)
      + 0.3 × max(0, 1 + MaxDD)        # MaxDD 为负
      + 0.2 × max(WinRate, 0)
      + 0.1 × min(ProfitFactor, 5) / 5
```

- 每个策略、每只股票保留前 3 组参数
- 输出 `results/optimizer/<日期>/best_params.json` 与 `scan_report.json`

#### EnsembleWeight 集成权重

- 用近期策略 Sharpe 做 softmax（温度 2.0）
- 每个策略权重下限 5%
- 输出 `weights.json`

界面启动时读取最近一天的 `best_params.json` 和 `weights.json`，作为 `analyze_stock()` 的参数覆盖和投票权重。

---

### 5.8 AI 决策助手 `core/ai_agent.py` + `ai.py` + `news.py`

#### AI 客户端 `ai.py`

- 通过 OpenAI SDK 调用 DeepSeek 兼容接口
- 默认 `base_url=https://api.deepseek.com`
- 默认模型 `deepseek-v4-flash`
- 支持普通对话和强制 JSON 输出
- 失败指数退避重试 3 次

#### 工具沙箱

AI 不是自由发挥，而是被限制在一组工具里：

| 类型 | 工具 | 说明 |
|---|---|---|
| 读 | `run_backtest` | 跑指定股票 + 策略 + 参数的回测 |
| 读 | `get_market_snapshot` | 获取行情快照 |
| 读 | `get_stock_history` | 获取历史日线 |
| 读 | `get_optimization_results` | 读取优化结果 |
| 读 | `get_performance` | 读取策略表现 |
| 读 | `get_current_params` | 读取当前参数 |
| 写 | `adjust_strategy_params` | 调整策略参数（有硬边界） |
| 写 | `adjust_ensemble_weights` | 调整集成权重 |
| 写 | `add_to_watchlist` | 加入自选（最多 20 只） |
| 写 | `force_rescan` | 强制重扫（每天最多 3 次） |

硬编码安全边界：

```text
单策略参数范围       PARAM_BOUNDS（如 fast ∈ [5,60]、slow ∈ [20,120]）
单次权重调整上限      0.30
单策略权重下限        0.05
写操作频率            10 次/分钟
强制重扫              3 次/天
自选股上限            20 只
每轮工具调用          5 个
```

所有写操作写入 `results/audit/<日期>.jsonl`，包含时间、工具、参数、结果、理由。

AI 通过回复里的 `<function_call>{...}</function_call>` 文本块调用工具；代理解析后执行，再把结果发回模型生成最终回答。这不是原生 function calling，但实现了同样的「模型 → 工具 → 模型」闭环。

#### 新闻情绪 `news.py`

- 东方财富个股新闻
- 优先用 AI 判断情绪，返回 `score ∈ [-1,1]` 和置信度
- 关键词匹配只作降级展示，模块注释明确写着「不要用于交易信号」

---

### 5.9 决策分析器 `core/analyzer.py`

单股分析的统一入口，被 App、日报、扫描器共用。

流程：

1. 数据不足 60 行直接返回 `None`
2. 用 5 个策略生成最新信号
3. 按权重统计买入票 / 卖出票，计算置信度
4. 识别市场状态
5. 计算 RSI、ATR、MA20、MA50、布林带位置
6. 如果是 BUY：按「资金 × 风险比例 × 状态乘数」计算建议股数和止损价
7. 如果存在因子排名快照：附加 `factor_score` 和全市场百分位
8. 返回统一字典：信号、置信度、指标、状态、股数、止损、各策略投票明细

信号规则：

- 加权买入票 > 加权卖出票 → `BUY`
- 加权卖出票 > 加权买入票 → `SELL`
- 否则 `HOLD`

---

### 5.10 Streamlit 前端 `app.py`

侧边栏可调扫描数量、价格区间、输入 DeepSeek Key；启动时加载最近的优化参数与权重。

六个 Tab：

| Tab | 功能 |
|---|---|
| **全市场扫描** | 涨跌统计、代码/名称搜索、涨跌幅筛选、红涨绿跌表格 |
| **个股分析** | K 线 + MA20/MA50 + 布林带 + 成交量 + MACD 三联图；因子得分/百分位；策略投票；买卖信号；建议股数与止损 |
| **AI 决策助手** | 聊天式分析，自动识别 6 位代码和常见股票名，把行情上下文喂给 AI；AI 可调用工具；无 Key 时退化为离线分析 |
| **策略验证** | 标准回测 / Walk-Forward；显示 Sharpe、最大回撤、胜率、交易次数、净值曲线、回撤曲线 |
| **交易信号** | 从成交额 TOP N 中扫描，列出 BUY/SELL（置信度 ≥ 40%），附建议股数和止损 |
| **追踪记录** | 手动记录每次决策（日期、代码、方向、价格、理由），之后自动拉现价计算对错，统计准确率 |

数据源降级时会明确提示：东方财富不可达、已切换到新浪，PE/PB 等字段暂缺。

---

### 5.11 定时任务 `scheduler.py`

每天收盘后跑一次完整闭环：

1. 交易日检查（周末、节假日、已跑过、未到 15:00 都跳过）
2. 拉成交额 TOP 50、价格区间内的行情快照
3. 加载每只股票最近 250 天日线
4. `ParamScanner.scan_all()` 网格搜索
5. `EnsembleWeight.update()` 更新策略权重
6. 保存 `best_params.json`、`scan_report.json`、`weights.json`
7. 跑因子流水线（失败不阻塞优化结果）
8. 保存当日排名快照

运行方式：

```bash
python scheduler.py            # 跑一次，自动判断是否该跑
python scheduler.py --force    # 强制跑
python scheduler.py --daemon   # 常驻，自动等到下一个交易日 15:05
```

服务器上建议用 crontab：

```cron
30 15 * * 1-5 cd /path/to/quantify && python scheduler.py
```

---

### 5.12 每日复盘 `daily_report.py`

- 固定关注 20 只大盘/行业代表股（茅台、五粮液、宁德、比亚迪、招行等）
- 研究资金使用 `research_capital = 100,000` 元（不是实盘的 1200 元）
- 输出 `results/reports/YYYY-MM-DD.md` 和 `.json`
- 报告内容：概览、买入信号表、卖出信号表、五策略投票明细、数据异常
- 归档里保留了 `2026-07-05` 的样例报告：20 只中 5 只买入、15 只卖出

---

### 5.13 小资金扫描 `quick_scan.py`

针对 1000 元资金的独立命令行工具：

- 价格 2～10 元（100 股 × 10 元 = 1000 元）
- 日成交额 ≥ 5000 万，避免流动性陷阱
- 候选股按四因子排序：成交额 40% + 低 PE 20% + 低 PB 20% + 动量 20%
- 输出三套 100 股整数倍分配方案：
  - 方案 A：动量优先
  - 方案 B：性价比优先（得分/价格）
  - 方案 C：蓝筹防御（银行、能源、基建等）
- 每套最多 5 只，贪心分配直到预算用完

---

### 5.14 配置系统 `config/settings.yaml`

所有魔法数字集中管理，`config/__init__.py` 提供 `get()` 和带类型的读取函数。

| 配置段 | 关键项 |
|---|---|
| `app` | 资金 1200、单笔风险 2%、价格 2～12 元、默认扫描 500 |
| `watchlist` | 6 只默认自选 |
| `account` | 初始资金 1200、研究资金 100,000、默认止损 5% |
| `costs` | 佣金万三、最低 5 元、印花税千一、滑点 0.1% |
| `ai` | DeepSeek 地址、模型、API Key（留空） |
| `data` | 缓存目录、前复权 |
| `regime` | 高低波动阈值 1.5 / 0.7、低波动收益阈值 3% |
| `optimizer` | TOP 50、回看 250 天、最少 60 天、评分权重 |
| `ensemble` | 窗口 20 天、温度 2.0、最低权重 0.05 |
| `ai_agent` | 每日重扫 3 次、每分钟写 10 次、审计目录、每轮 5 个工具 |

---

## 6. 数据与产物现状（归档时）

| 项目 | 状态 |
|---|---|
| 日线缓存 | 781 个 parquet，覆盖大量 A 股 |
| 排名快照 | **不存在**（`data_cache/rankings/` 未生成） |
| 优化结果 | 只有 `results/optimizer/ai_overrides.json`，没有按日期的 `weights.json` / `best_params.json` |
| 日报 | 只有 `results/reports/2026-07-05.md/.json` |
| AI 审计 | `results/audit/` 为空 |
| 决策追踪 | `data_cache/tracking.json` 不存在 |
| 虚拟环境 | `.venv` 未保留 |
| 归档标记 | `.claude/project-context.md` 中 `pipeline.status: ARCHIVED`，`.claude/fief-config.yaml` 中所有 agent `enabled: false` |

`ai_overrides.json` 里还留着一次 AI 调整记录：

```json
{
  "params": {
    "趋势-均线": { "fast": 15, "slow": 45 }
  },
  "_last_modified": "2026-07-09T16:23:36",
  "_reason": ""
}
```

---

## 7. 测试与质量

- 14 个测试文件、约 1,800 行测试代码
- 初始提交信息标注 **103 tests passing**
- `.pytest_cache/v/cache/lastfailed` 为 `{}`，说明最后一次跑测试时没有失败项
- 覆盖范围：
  - 指标正确性（SMA/EMA/RSI/MACD/BOLL/ATR）
  - 5 个策略的信号生成
  - 回测引擎、回撤、Walk-Forward
  - 风险：仓位、止损、凯利、波动率目标
  - 市场状态识别与风险乘数
  - 因子计算全部子模块
  - 横截面：Z-score、行业中性、Rank IC、IC 加权、全市场排名
  - 优化器：交易日历、参数扫描、集成权重
  - AI 代理：参数边界、工具注册、审计日志、限流
  - 集成测试：分析器全流程、优化器 → 分析器、因子流水线
- **本次没有重建虚拟环境重新跑测试**，以上是归档前的状态。

---

## 8. 如何恢复运行

### 8.1 重建环境

```bat
cd /d "path\to\quantify"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 8.2 启动界面

```bat
start.bat
```

或手动：

```bat
.venv\Scripts\streamlit run app.py --server.port 57921
```

浏览器打开 `http://localhost:57921`。

### 8.3 跑测试

```bat
.venv\Scripts\python -m pytest tests/ -v
```

### 8.4 生成收盘后数据

```bat
.venv\Scripts\python scheduler.py --force
```

第一次运行会：
- 拉取行情与日线（较慢，781 个缓存文件可复用）
- 生成 `results/optimizer/<日期>/`
- 生成 `data_cache/rankings/<日期>.parquet`

在此之前，界面里的「因子评分」不会出现，会退化为策略投票模式。

### 8.5 生成日报 / 跑小资金扫描

```bat
.venv\Scripts\python daily_report.py
.venv\Scripts\python quick_scan.py
```

### 8.6 数据源注意事项

- 东方财富 `push2.eastmoney.com` 在部分网络/代理环境下不可达，系统会自动降级到新浪，但 PE/PB、换手率、市值字段会缺失。
- 新浪接口有频率限制，触发 HTTP 456 后会暂停；不要在短时间内反复全市场扫描。
- 所有网络请求都有超时与降级逻辑，单只股票失败不会拖垮整条流水线。

### 8.7 解除归档

按 `.claude/project-context.md` 的复苏指南：

1. 把 `pipeline.status` 从 `ARCHIVED` 改为 `ACTIVE`
2. 在 `.claude/fief-config.yaml` 中启用需要的 agent
3. 运行 `/orchestrator`

但更推荐手动恢复开发，先修第 10 节列出的问题。

---

## 9. 设计亮点

1. **真实的 A 股摩擦建模**：T+1、涨跌停、最低佣金、印花税、滑点、整手，回测结果比「裸信号 × 收益」可信得多。
2. **防前视偏差**：信号统一 `shift(1)`，回测和研究口径一致。
3. **离线计算 / 在线查询分离**：重计算不阻塞界面，符合量化系统的常见工程结构。
4. **多级数据源降级**：东方财富 → 新浪 → 本地缓存；每个新数据源失败都只产生 NaN，不阻塞流水线。
5. **多因子 + IC 加权 + 行业中性**：从「单股看均线」升级到「全市场横截面比较」，是项目最有含金量的部分。
6. **AI 沙箱**：AI 只能调用白名单工具，参数有硬边界，写操作有频率限制和审计日志，不允许文件系统写入、真实交易、网络访问。
7. **人机协同**：系统只给建议，不自动下单，保留人工判断环节。
8. **配置集中化**：1200 元资金、成本、阈值、优化权重全部在 `settings.yaml`，没有散落的魔法数字。
9. **可验证性**：Walk-Forward、OOS 检查、103 个测试，说明作者有意识地对抗过拟合和回归。

---

## 10. 已知问题与限制

> 以下问题来自静态代码梳理，归档后未运行时验证。按影响程度排序。

### 10.1 README 与实际结构不一致（文档问题）

`README.md` 仍在描述「五个阶段 + notebooks + data/strategies/backtest/risk/execution/utils 目录」的教学版结构，但仓库里：

- 没有 `notebooks/` 目录
- 没有 `data/`、`strategies/`、`backtest/`、`risk/`、`execution/`、`utils/` 这些顶层目录
- 实际架构是 `core/` 模块化 + Streamlit

README 的「项目结构」和「五阶段」部分已过时，应以本文件和第 4 节为准。

### 10.2 日报的「详细投票」表会全部显示 `-`（功能缺陷）

`daily_report.py` 里取投票明细用的键是：

```python
v('趋势均线(20/50)')
v('MACD交叉(12/26/9)')
v('动量(60日)')
v('量确认(20/50)')
v('反转(BB+RSI)')
```

但 `core/strategies.py` 里实际的策略名是：

```text
趋势-均线 / 趋势-MACD / 动量 / 量确认 / 反转
```

键名不匹配，导致新生成的日报「详细投票」列全部是 `-`。归档里的 `2026-07-05` 报告是旧版本生成的，所以还有值。

### 10.3 集成权重实际上退化为等权（逻辑缺陷）

`scheduler.py` 给 `EnsembleWeight.update()` 传的是每个策略的**一个平均 Sharpe 值**（单元素列表）：

```python
perf[name] = [np.mean(sharpes)] if sharpes else [0.0]
```

而 `EnsembleWeight._rolling_sharpe()` 要求至少 5 个样本，否则返回 0：

```python
if len(returns) < 5:
    return 0.0
```

结果所有策略得分都是 0，softmax 后权重均匀。**「按近期表现动态加权」目前没有真正生效**，需要改成传入每个策略的日收益序列，或放宽单值处理。

### 10.4 IC 不会自动更新，排名快照也未生成（未完成闭环）

- `compute_rank_ic()` 已实现且有测试，但调度器没有调用它。
- `_load_or_init_ic()` 只是读取 `data_cache/rankings/factor_ic.json`，不存在就用一份手工默认 IC 表。
- 归档时 `data_cache/rankings/` 不存在，说明因子流水线在归档前没有成功产出过排名快照。

所以「IC 动态加权」的机制是完整的，但「每日用未来收益回算 IC 并迭代」的闭环没有接上。

### 10.5 行业动量因子是空实现（占位）

`compute_sector_momentum_factors()` 直接 `return {}`，注释说明需要板块指数 OHLCV 数据。因此 11 类因子里实际生效的是 10 类。

### 10.6 AI 模型配置与实际默认值不一致

- `config/settings.yaml` 写的是 `model: "deepseek-chat"`
- `ai.py` 的默认值是 `model="deepseek-v4-flash"`
- `app.py` 构造 `AIClient(api_key=...)` 时没有传 `model`，所以**实际用的是 `deepseek-v4-flash`，配置文件里的 `ai.model` 没有被读取**

### 10.7 部分风控函数已实现但未接入主流程

| 函数 | 现状 |
|---|---|
| `kelly_fraction` | 只有测试调用，主流程用固定比例仓位 |
| `volatility_target_shares` | 只有测试调用 |
| `check_stops` | 只有测试调用，界面只显示固定/ATR 止损价，没有持仓后的多级止损检查 |
| `regime_strategy_weights` | 已定义并导出，但 `analyzer.py` 没有使用，策略权重只来自优化器或等权 |

这些是「写好了但还没接线」的功能，不是坏代码。

### 10.8 小资金扫描的硬编码

`quick_scan.py` 顶部写死 `CAPITAL = 1000`、`MAX_PRICE = 10.0`，与 `settings.yaml` 的 1200 元 / 12 元不一致。两个工具面向的场景略有不同（1000 元纯现金 vs 1200 元账户），但参数没有统一到配置。

### 10.9 归档相关的路径残留

`.claude/agents/sikong.md`、`sikou.md` 里的 `PROJECT_ROOT` 仍写着归档前的本地路径。如果直接按 agent 流程恢复，路径需要先改。

### 10.10 不是实盘系统

- 没有券商 API、没有订单管理、没有真实成交回报
- 没有处理停牌、除权、退市、涨跌停无法成交等极端情况
- 所有结果仅供学习研究，不构成投资建议

---

## 11. 版本演进

Git 历史共 23 个提交，集中在 2026-07-10 ～ 07-11。主要阶段：

| 阶段 | 代表提交 | 内容 |
|---|---|---|
| 基线 | `02d92ab` / initial commit | Quantify Pro v4 baseline，103 tests passing |
| 数据层扩展 | `4c9f007` 之前若干 | 东方财富 PE/PB/市值、北向资金、两融、财报、资金流、行业、涨停池 |
| 因子体系 | `c6675e9` | 因子计算引擎（动量/波动/技术/价值/规模/质量/北向/两融/资金流/情绪/行业） |
| 横截面 | `6d98f0e` | 横截面排名 + IC 加权评分 |
| 股票池 | `9a78e64` | universe 过滤 |
| 因子接入 | `fbfee18` / `4c9f007` | 调度器因子流水线、分析器因子得分 |
| 修复 | `310ca32` | 财报指标误取最旧一期 → 改为最新一期 |
| 修复 | `e83e605` | 补上 7 个被静默忽略的因子 IC 权重 |
| 修复 | `95d7fce` 等 | 新浪降级排序、无效代码过滤、东方财富重试次数 |

设计文档：

- `docs/superpowers/specs/2026-07-09-auto-optimizer-design.md` — 自动优化器 v2
- `docs/superpowers/specs/2026-07-10-modern-factor-upgrade-design.md` — 多因子升级设计
- `docs/superpowers/plans/2026-07-10-modern-factor-upgrade.md` — 1,659 行实施计划（按任务拆分）

---

## 12. 术语表

| 术语 | 解释 |
|---|---|
| OHLCV | 开盘/最高/最低/收盘/成交量 |
| 前复权 qfq | 调整历史价格，使分红拆股不影响回测 |
| T+1 | A 股当天买入次日才能卖出 |
| 整手 | 1 手 = 100 股，买入数量必须是 100 的整数倍 |
| 前视偏差 | 用未来信息交易过去，回测最常见错误 |
| IC | 因子值与未来收益的相关性 |
| Rank IC | 因子排名与未来收益排名的 Spearman 相关 |
| Z-score | 标准化，`(x - mean) / std` |
| 行业中性 | 减去行业均值，消除行业整体偏差 |
| 最大回撤 | 从历史最高点到最低点的最大跌幅 |
| Sharpe | 每承担一单位波动获得的超额收益 |
| Walk-Forward | 滚动训练/测试，检验策略在未知数据上的表现 |
| 凯利公式 | 根据胜率和赔率计算最优下注比例 |
| 涨停/跌停 | A 股单日涨跌幅限制，主板通常 ±10% |

---

## 13. 结论：这个项目适合怎么用

### 作为学习材料

它覆盖了一条完整的量化流水线：

```text
数据获取 → 清洗缓存 → 技术指标 → 策略 → 回测 → 风控
→ 多因子 → 横截面排名 → 参数优化 → AI 代理 → 可视化 → 日报
```

比单看教程更完整，尤其是「A 股摩擦建模 + 多因子 + 防过拟合」这三块。

### 作为工程参考

值得借鉴的设计：

- 数据源降级与缓存
- 离线 / 在线分层
- 配置集中化
- AI 工具沙箱与审计
- 回测真实性约束

### 如果要恢复开发，建议顺序

1. 重建 `.venv`，跑通 `pytest tests/ -v`
2. 修 `daily_report.py` 的策略键名（10.2）
3. 修 `EnsembleWeight` 的输入格式（10.3）
4. 把 `compute_rank_ic()` 接入调度器，形成 IC 更新闭环（10.4）
5. 补行业动量因子（10.5）
6. 统一 AI 模型配置（10.6）
7. 把凯利 / 波动率目标 / 多级止损接入主流程（10.7）
8. 更新 README 与 agent 路径（10.1 / 10.9）

### 最后

这是一个**学习与研究性质**的半自动系统，不是能直接实盘的交易机器人。归档状态是合理的：它已经完成了「把量化交易的主要环节都走一遍」的使命。

---

*本文档由 Claude Code 基于归档源码静态梳理生成，2026-11-11。*
