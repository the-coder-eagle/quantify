# Auto-Optimizer: 策略自动迭代优化系统 v2

## 概述

让系统自动"试参数 → 回测打分 → 选最优 → 动态加权 → 第二天用"，形成一个自我改进的闭环。方案 A（定时参数扫描）+ 方案 C（策略集成动态权重）组合。

**v2 新增**：
- 交易日检测：非交易日不重复计算
- AI 高权限代理：AI 可以调用回测和参数调整工具，但操作范围受限

## 用户背景

- 大学生，资金约 1200 元人民币
- 系统 CAPITAL = 1,200，max_price 默认 12 元
- 目标：学习量化交易 + 找到适合小额资金的稳健策略

---

## 架构

```
┌──────────────────────────────────────────────────────────────┐
│                      quantify 系统                            │
│                                                               │
│  ┌──────────────────┐   ┌──────────────────┐                 │
│  │   optimizer.py    │   │   ai_agent.py    │  ← 新增         │
│  │   (自动优化)       │   │   (AI高权限代理)  │                │
│  │                   │   │                  │                 │
│  │  ParamScanner     │   │  AI + 工具集      │                │
│  │  EnsembleWeight   │   │  - run_backtest   │                │
│  │  TradingCalendar  │   │  - adjust_params  │                │
│  │                   │   │  - adjust_weights │                │
│  │                   │   │  - scan_market    │                │
│  └────────┬──────────┘   └────────┬─────────┘                │
│           │                       │                           │
│           └───────────┬───────────┘                           │
│                       ▼                                       │
│           ┌───────────────────┐                               │
│           │  analyzer.py      │  ← 改用最优参数+动态权重       │
│           └────────┬──────────┘                               │
│                    │                                          │
│                    ▼                                          │
│              app.py（Streamlit 前端）                          │
│                    │                                          │
│                    ▼                                          │
│           scheduler.py（交易日自动运行）                        │
└──────────────────────────────────────────────────────────────┘
```

---

## 模块设计

### 1. TradingCalendar — 交易日历 (新增)

**职责**：判断今天是否A股交易日，防止非交易日重复计算。

```
检测逻辑（按优先级）：
  1. 检查本地缓存：results/optimizer/{date}/ 是否已存在 → 跳过
  2. 检查星期：周六/周日 → 非交易日
  3. 检查时间：15:00 前不运行（盘中数据不完整）
  4. 检查节假日：使用 akshare 交易日历接口验证
  5. 全部通过 → 是交易日，执行优化
```

```python
def is_trading_day(date=None) -> bool:
    """Return True if date is an A-share trading day and we haven't run yet."""
    # 1. 已有结果 → 跳过
    # 2. 周末 → 跳过
    # 3. 盘中 → 跳过
    # 4. akshare 节假日查询 → 跳过
```

### 2. ParamScanner — 参数扫描器（方案 A）

**职责**：对每个策略的参数空间做网格搜索，回测评分，输出最优参数。

```
输入：股票历史数据 DataFrame, 策略类, 参数网格, 回测天数
处理：
  1. 对 param_grid 的所有组合做 product 展开
  2. 每个组合跑 BacktestEngine.run()
  3. 记录 Sharpe, MaxDD, WinRate, ProfitFactor
  4. 综合打分 = Sharpe × 0.4 + (1-|MaxDD|) × 0.3 + WinRate × 0.2 + ProfitFactor × 0.1
  5. 排序，取 top-3
输出：{"strategy_name": [{"params": {...}, "score": 0.85}, ...], ...}
```

**策略参数网格**：

| 策略 | 参数空间 | 组合数 |
|------|---------|--------|
| MATrend | fast: [10,20,30], slow: [40,50,60], ma_type: ["sma","ema"] | 18 |
| MACDTrend | fast: [8,12,16], slow: [21,26,31], signal: [7,9,11] | 27 |
| Momentum | lookback: [30,60,90], threshold: [0.0, 0.02] | 6 |
| VolumeConfirm | fast: [10,20], slow: [40,50], vol_mult: [1.0, 1.2, 1.5] | 12 |
| MeanReversion | bb_window: [15,20,25], oversold: [25,30,35], overbought: [65,70,75] | 27 |

总计 90 组合。单股票约 9 秒，50 只股票约 7-8 分钟。

**防重复机制**：
- 结果写入 `results/optimizer/{YYYY-MM-DD}/` 目录
- 目录存在 → 今日已优化 → 跳过
- 非交易日 → 跳过
- AI 手动触发 → 不受此限制（写入 `results/optimizer/{date}_ai_{timestamp}/`）

### 3. EnsembleWeight — 动态权重（方案 C）

**职责**：根据各策略近期表现，动态调整投票权重。

```
输入：各策略最近N天的信号序列 + 实际收益
处理：
  1. 滑动窗口（默认20天）
  2. 计算每个策略的滚动 Sharpe
  3. 权重 = softmax(滚动Sharpe × temperature)
     - temperature=2: 表现好的权重更大但不极端
  4. 最小权重保护：每个策略至少 5%
输出：{"策略名": weight, ...} 总和=1.0
```

### 4. AIAgent — AI 高权限代理 (新增)

**职责**：给 AI 提供受控的工具集，允许它在沙箱内运行回测、调整参数，但不能触碰系统其他部分。

**设计原则**：
- AI 只能调用我们定义的工具函数，不能执行任意代码
- 所有修改都有边界检查（参数范围、权重限制）
- 所有操作记录审计日志
- 可以回滚任何 AI 做出的修改

**AI 可用工具清单**：

| 工具名 | 功能 | 权限 | 限制 |
|--------|------|------|------|
| `run_backtest` | 对某只股票用指定参数跑回测 | 读 | 最多同时跑3个 |
| `get_market_snapshot` | 获取当前市场快照 | 读 | 受 min/max_price 限制 |
| `get_stock_history` | 获取个股历史K线 | 读 | 无限制 |
| `get_optimization_results` | 读取最近优化结果 | 读 | 只读，不能写 |
| `get_performance` | 查看近期策略表现 | 读 | 无限制 |
| `adjust_strategy_params` | 修改某策略的参数 | **写** | 参数必须在预设范围内 |
| `adjust_ensemble_weights` | 调整策略投票权重 | **写** | 单次变动≤30%，权重和=1 |
| `add_to_watchlist` | 添加关注股票 | **写** | 最多20只 |
| `force_rescan` | 触发立即重新扫描 | **写** | 每天最多3次 |

**安全边界**：

```python
# 参数调整的范围限制（硬编码，AI 无法绕过）
PARAM_BOUNDS = {
    "MATrend":       {"fast": (5, 60), "slow": (20, 120)},
    "MACDTrend":     {"fast": (5, 30), "slow": (15, 50), "signal": (5, 20)},
    "Momentum":      {"lookback": (10, 120), "threshold": (-0.05, 0.05)},
    "VolumeConfirm": {"fast": (5, 40), "slow": (20, 80), "vol_mult": (0.8, 2.0)},
    "MeanReversion": {"bb_window": (10, 40), "oversold": (15, 40), "overbought": (60, 85)},
}

# 权重调整限制
MAX_WEIGHT_CHANGE = 0.30    # 单次最多调整 30%
MIN_STRATEGY_WEIGHT = 0.05  # 每个策略至少 5%
```

**AI 不能做的事**：
- 修改系统代码文件
- 执行真实的交易下单
- 访问网络（除了我们提供的数据接口）
- 修改 PARAM_BOUNDS 等安全限制
- 删除历史记录

**审计日志**：
```
results/audit/
├── 2026-07-09.jsonl   # 每天一个日志文件
│   {"time":"15:32:01","tool":"adjust_weights","args":{"趋势-均线":0.35,...},"reason":"AI: 动量策略近期表现优异，小幅加仓"}
│   {"time":"15:35:22","tool":"adjust_params","args":{"MATrend":{"fast":15,"slow":45}},"reason":"AI: 缩短均线周期以适应波动加大"}
```

### 5. OptimizationScheduler — 定时调度

**职责**：管理优化任务的执行时机。

- **触发时机**：交易日 15:00 后
- **防重复**：TradingCalendar 检查 + 目录存在检查
- **扫描范围**：市场成交额 top 50（受 max_price 限制）+ 用户关注列表
- **持久化**：`results/optimizer/{YYYY-MM-DD}/`
  - `best_params.json`：各策略最优参数
  - `weights.json`：动态权重
  - `scan_report.json`：完整扫描报告
- **清理策略**：保留最近 60 天

### 6. 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `core/optimizer.py` | **新建** | ParamScanner + EnsembleWeight + TradingCalendar |
| `core/ai_agent.py` | **新建** | AI 高权限代理 + 工具定义 + 安全边界 |
| `core/analyzer.py` | 修改 | 支持传入自定义参数和权重 |
| `app.py` | 修改 | Tab3 AI 对话接入 ai_agent 工具 |
| `scheduler.py` | **新建** | 交易日定时任务入口 |
| `config/optimizer.yaml` | **新建** | 优化器配置 |
| `results/audit/` | **新建** | AI 操作审计日志目录 |
| `tests/test_optimizer.py` | **新建** | 优化器测试 |
| `tests/test_ai_agent.py` | **新建** | AI 代理安全测试 |

---

## 数据流

```
交易日 15:00
  │
  ├─→ TradingCalendar.is_trading_day()
  │     ├─ 已有结果 → 跳过 ✗
  │     ├─ 周末/节假日 → 跳过 ✗
  │     └─ 是交易日 → 继续 ↓
  │
  ├─→ market_snapshot(limit=50, max_price=12)
  │     └─→ ParamScanner.scan(each_stock)
  │           └─→ EnsembleWeight.update(each_stock)
  │                 └─→ 写入 results/optimizer/{date}/
  │
  ├─→ [可选] AI Agent 审查结果
  │     └─→ adjust_params / adjust_weights（受安全限制）
  │           └─→ 写入审计日志
  │
  └─→ 第二天开盘
        └─→ app.py 加载最新优化结果 + AI 调整
              └─→ analyze_stock(df, params, weights)
```

---

## AI 交互流程

```
用户: "最近市场波动大，把止损调紧一点"
  │
  ▼
AI Agent 收到消息
  │
  ├─→ 调用 get_performance() 看近期表现
  ├─→ 调用 get_optimization_results() 看当前参数
  ├─→ 分析：当前 ATR 止损倍数是 2.0，建议调到 1.5
  │
  ├─→ 调用 adjust_strategy_params("MeanReversion", {"oversold": 35, "overbought": 65})
  │     ├─ 边界检查: 35 ∈ [15,40] ✓, 65 ∈ [60,85] ✓
  │     ├─ 写入新参数
  │     └─ 记录审计日志
  │
  └─→ 回复: "已将反转策略的布林带阈值放宽到 35/65，
            并建议将 ATR 止损倍数从 2.0 调到 1.5。
            需要我执行止损调整吗？"
```

---

## 错误处理

- **数据不足**（<60天）：跳过该股票，记录日志
- **回测异常**（全为0信号）：标记为无效，不参与排序
- **API 限流**：指数退避重试，最多 3 次
- **非交易日触发**：静默跳过，记录日志
- **AI 参数越界**：拒绝修改，返回错误消息给 AI
- **AI 调用过于频繁**：force_rescan 每天限 3 次，其他写操作每分钟限 10 次
- **磁盘写入失败**：logger.error，不影响主流程
- **无优化结果时**：回退到默认参数

---

## 测试策略

- **单元测试**：TradingCalendar 交易日判断
- **单元测试**：ParamScanner 已知最优参数还原
- **单元测试**：EnsembleWeight 权重和=1，边界处理
- **安全测试**：AI 参数越界被拒绝
- **安全测试**：AI 权重超限被拒绝
- **安全测试**：AI 不能调用未注册的工具
- **集成测试**：scan → save → load → analyze 链路
- **回归测试**：默认参数下输出与现有一致

---

## 部署说明

```bash
# crontab：仅交易日 15:30 触发
# scheduler.py 内部会再次检查是否交易日
30 15 * * 1-5 cd /path/to/quantify && python scheduler.py

# 常驻模式（同时运行 Streamlit + 定时任务）
python scheduler.py --daemon
```

---

## 风险与限制

- **过拟合风险**：参数优化本质是曲线拟合，Walk-Forward 验证可部分缓解
- **计算资源**：90组合×50只股票≈4500次回测，单线程约7-8分钟
- **AI 幻觉**：AI 可能提出不合理的参数调整，边界检查会拦截，但不会阻止AI输出错误建议
- **AI 成本**：每次对话约消耗 DeepSeek API token，建议使用便宜的 deepseek-chat 模型
- **权重滞后**：动态权重基于过去20天，趋势突变时会滞后1-3天
