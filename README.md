# Quant Strategy（MVP）

模块化量化研究项目：**数据 → 因子面板 → IC → 回测（Top-K + 等权 / 夏普 / 风险平价）→ 基准与超额收益 → 换手与成本 → 绩效与作图 → 实验记录落盘**。

**文档与代码**：以 `main.py` 与 `config.Settings` 为准；更新行为后请同步修改 `docs/ENGINEERING_OVERVIEW.md`、`docs/FLOW_AND_MODULES.md` 及本 README 相关段落（仓库无自动文档校验）。

### MVP 定稿（范围）

**本仓库 MVP 已交付**，指下面闭环可稳定跑通、用于研究与对内演示；不要求实盘下单。

| 在 MVP 内 | 不在 MVP 内（后续扩展） |
|------------|-------------------------|
| 行情接入（CSV / Tushare / 合成兜底）、四因子面板、IC 与可选 CSV/图 | `live/signal_system.generate_signals`、`live/paper_trading.run_paper_trading`（仅占位） |
| 月末再平衡、Top-K、`portfolio_weighting`：`equal` / `max_sharpe` / `risk_parity` | `fuse_models` 除 `mean_zscore` / `mean` 外的 `method`（如 `dynamic`、`xgboost`） |
| 单因子回测 + **IC 驱动或等权** z-score 融合回测、`meta["rebalance_log"]` | `main` 未接 `run_multi_backtest(factors, weights)` 线性加权入口（代码已有，非主流程） |
| 绩效 `summarize`、股票池等权基准、超额收益 / 跟踪误差 / 信息比率、换手率与预估成本、净值/IC/权重/换手图、`performance_summary.csv`、`run_config.json`、调仓/换手日志 CSV、`persist_run_outputs` 落盘 | 真实券商 API、实时风控与订单路由 |

## 文档

- **项目介绍（MVP 工程）**：[docs/MVP_PROJECT_ARTICLE.md](docs/MVP_PROJECT_ARTICLE.md) — Quant Strategy 的定位、模块关系、默认全流程与主流程表、数据/因子/IC/融合/回测与后续扩展方向
- **主流程与各模块**：[docs/FLOW_AND_MODULES.md](docs/FLOW_AND_MODULES.md)（含 Mermaid 流程图）
- **工程总览（技术细节）**：[docs/ENGINEERING_OVERVIEW.md](docs/ENGINEERING_OVERVIEW.md)
- **接口与数据契约**：[docs/INTERFACE_AND_CONTRACTS.md](docs/INTERFACE_AND_CONTRACTS.md)
- **代码结构**：[docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md)

原创长文与小红书草稿默认只在本地保留，并通过 `.gitignore` 排除，避免随公开代码仓库发布。

## 环境

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Token：优先环境变量 `TUSHARE_TOKEN`；未设置时使用 `config.py` 内 `_TUSHARE_TOKEN_LOCAL`（**勿将含真实密钥的 config 提交远程**）。

## 目录结构

```
data/           # 原始/演示数据（如 prices_demo.csv）
output/         # 运行生成：nav_compare.png、excess_nav_compare.png、turnover_compare.png、performance_summary.csv、cache/、rebalance_logs/、turnover_logs/ 等
factors/        # 因子与 panel_builder
backtest/       # backtest_single、backtest_multi、utils
models/         # fusion、optimizer
analysis/       # performance、benchmark、turnover、plotting、ic
live/           # data_feed、cache_io（MVP 用）；signal/paper 非 MVP 占位
config.py
main.py
```

## 运行

```bash
python main.py
```

### 当前 `main.py` 实际顺序（与代码一致）

1. **数据**：`data/prices_demo.csv` 优先；否则 Tushare（`main._DEFAULT_TS_SYMBOLS`）；失败则合成宽表。得到 `prices`（宽表）与 `long_df`。
2. **因子面板**：`factors.panel_builder.build_four_factor_panel`（四列：`MOMENTUM`、`VOLATILITY`、`PE`、`ROE`）。
3. **落盘**：若 `persist_run_outputs`，`live.cache_io.save_run_cache` → `output/cache/`（`prices_long.csv`、`prices_wide_close.csv`、`factor_panel.csv`、`run_meta.txt`）。
4. **IC**：`analysis.ic` 对各因子列及 **与融合同构的** FUSED 得分算日截面 Spearman；若 `persist_run_outputs`，另存 `ic_*.csv`。
5. **单因子回测**：对每列 `run_single_backtest(fname, factor_values=col, ...)`（**预计算因子**，不调注册表重算）。
6. **融合回测**：**IC 滞后滚动列权（默认）或等权** z-score → `run_multi_backtest(fused=..., factor_name="FUSED_ZSCORE")`（内部仍调 `run_single_backtest`）。
7. **基准与超额收益**：`analysis.benchmark.equal_weight_benchmark_nav` 构造股票池等权基准；每条策略补 `benchmark_ann_return`、`excess_ann_return`、`tracking_error`、`information_ratio`。
8. **换手与成本**：`analysis.turnover` 从 `meta["rebalance_log"]` 估算逐期 `turnover`、`estimated_cost`，并汇总 `avg_turnover`、`total_turnover`、`estimated_total_cost`。
9. **实验记录与作图**：若 `persist_run_outputs`，保存 `output/cache/run_config.json`、`output/performance_summary.csv`、`output/rebalance_logs/*.csv`、`output/turnover_logs/*.csv`、`ic_compare.png`、`ic_timeseries_*.png`、`weights_*.png`、`turnover_compare.png`；`plot_nav` → `output/nav_compare.png`，超额净值 → `output/excess_nav_compare.png`。

### 回测与配置要点

- **再平衡**：默认 `config.rebalance_freq = "ME"`（月末）；**Top-K** 默认 `top_k=5`；因子截面**降序**取前 K。
- **持仓权重**：`config.portfolio_weighting`：`"equal"`、**`"max_sharpe"`（当前默认）** 或 **`"risk_parity"`**；后两者在再平衡日对 Top-K 用历史日收益估协方差（夏普另需 μ），分别调用 `models.optimizer.maximize_sharpe` / `risk_parity`，样本不足等失败则等权。
- **调仓记录**：`meta["rebalance_log"]`；`main` 会打印每期标的与权重，并在 `persist_run_outputs=True` 时保存到 `output/rebalance_logs/*.csv`。
- **换手记录**：`analysis.turnover` 以调仓目标权重变化估算成交占比，并在 `persist_run_outputs=True` 时保存到 `output/turnover_logs/*.csv`。

### 依赖

见 `requirements.txt`（含 **pandas、numpy、scipy、matplotlib、scikit-learn、tushare** 等）。

### 非 MVP（占位或扩展）

- `live/signal_system.generate_signals`、`live/paper_trading.run_paper_trading`
- `models.fusion.fuse_models`：仅 `mean_zscore` / `mean` 可用，其它 `method` 会报错

### 测试

```bash
python3 -m unittest tests.test_optimizer tests.test_backtest_multi tests.test_backtest_single tests.test_plotting tests.test_fusion tests.test_cache_io tests.test_benchmark tests.test_turnover -v
```
