# Quantify — A股量化分析与模拟回测

基于 Python、Streamlit 和 AkShare 的学习项目，提供行情筛选、技术因子分析、策略回测、参数优化和 AI 辅助分析。**不应将分析信号视为投资建议或直接用于实盘。**

## 启动

要求 Python 3.10 或更高版本。以下命令在仓库根目录执行：

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell 改用：
# .venv\Scripts\Activate.ps1

python -m pip install -r requirements-lock.txt
python -m streamlit run app.py --server.port 57921
```

浏览器访问 http://localhost:57921 。Windows 用户也可以查看并运行仓库中的 `start.bat`。

## 测试

```bash
python -m pytest tests/ -q
```

默认测试使用合成数据与替身，并跳过 7 项在线行情检查，不验证真实 AI 服务。在线行情检查需显式开启，可能受网络、限流和缓存影响：

```bash
python -m pytest tests/ -q --run-live -m live
```

这些在线检查允许 API 返回空结果，测试通过不代表行情源可用或数据新鲜。

## 当前项目结构

- `app.py`：Streamlit 分析界面。
- `core/data.py`：行情获取与缓存。
- `core/factors.py`、`core/cross_section.py`、`core/analyzer.py`：因子与选股分析。
- `core/strategies.py`、`core/backtest.py`：传统策略与向量化回测。
- `core/optimizer.py`：参数搜索、策略权重与优化结果读取。
- `core/risk.py`、`core/regime.py`：风险辅助计算与市场状态。
- `core/ai_agent.py`、`ai.py`：AI 客户端与受限工具调用。
- `config/settings.yaml`：应用配置。
- `scheduler.py`、`daily_report.py`、`quick_scan.py`：调度、报告与扫描入口。
- `tests/`：自动化测试。

## AI 工具的边界

每次对话最多执行 5 个工具调用；写入工具另有频率限制。模型不能指定输出目录。保存的参数提案必须符合已定义的数值范围，窗口长度必须为整数；权重必须覆盖全部五种已知策略、总和为 1，且每项不低于 5%。

AI 参数和权重修改会保存到 `results/optimizer/ai_overrides.json`，**目前运行时加载器不会自动应用此文件**。工具返回 `active: false`，不要把“保存成功”当成分析或回测参数已生效。审计记录位于 `results/audit/`。这些应用层限制不是隔离操作系统的安全沙箱。

## 回测限制

目前回测是简化的向量化模型：

- 策略信号滞后一根 K 线，使用收盘收益率计算。
- 计入比例佣金、滑点与卖出印花税；不模拟逐笔撮合、资金余额、100 股整手、最低佣金或完整 T+1 约束。
- 涨跌幅被统一截断至配置范围，不等同于真实涨跌停无法成交的处理，也不覆盖不同板块和 ST 规则。
- 交易胜率等统计使用毛收益，包含末尾未平仓区间；权益收益扣除比例成本。
- Walk-forward 测试窗口独立从空仓开始，窗口内重新预热，净收益连续复利拼接；不代表跨窗口连续持仓。

因此收益、回撤与交易统计仅用于研究比较，不应视为可兑现的实盘表现。

## 行情故障排查

行情依赖第三方服务，可能遇到限流、网络错误或数据延迟。先查看日志和缓存日期，稍后重试；不要把缺失行情解释为没有风险或没有交易机会。在线服务及 AI 配置请参照代码与 `config/`，不要将密钥提交到仓库。
