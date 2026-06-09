# Quant Strategy（MVP）

模块化量化研究项目：**数据 → 因子面板 → 数据质量 → IC → 因子诊断（Top-K 多头超额 + 分组收益单调性）→ 多因子权重建议 → 融合回测（IC 滚动列权 + 训练段静态综合权重 + 调仓日前滚动综合权重）→ 可交易性 / 流动性过滤 → 回测（Top-K + 等权 / 夏普 / 风险平价）→ 决策审计日志 → 基准与超额收益 → 换手与成本 → 风险暴露与集中度 → 绩效与作图 → 实验记录落盘**。

**文档与代码**：以 `main.py` 与 `config.Settings` 为准；更新行为后请同步修改 `docs/ENGINEERING_OVERVIEW.md`、`docs/FLOW_AND_MODULES.md` 及本 README 相关段落（仓库无自动文档校验）。

### MVP 定稿（范围）

**本仓库 MVP 已交付**，指下面闭环可稳定跑通、用于研究与对内演示；不要求实盘下单。

| 在 MVP 内 | 不在 MVP 内（后续扩展） |
|------------|-------------------------|
| 行情接入（CSV / Tushare / 合成兜底）、多因子面板（量价 + 财务）、数据质量 / 覆盖率报告、IC 与可选 CSV/图、因子 Top-K 多头超额诊断、分组收益与单调性分析、多因子权重建议表 | `live/signal_system.generate_signals`、`live/paper_trading.run_paper_trading`（仅占位） |
| 月末再平衡、Top-K、可交易性 / 流动性过滤、停牌 / 涨跌停交易约束、`portfolio_weighting`：`equal` / `max_sharpe` / `risk_parity`，`max_position_weight` 单票权重上限，`max_rebalance_turnover` 单次换手上限 | `fuse_models` 除 `mean_zscore` / `mean` 外的 `method`（如 `dynamic`、`xgboost`） |
| 单因子回测 + **IC 驱动或等权** z-score 融合回测 + **训练段静态综合权重**验证回测 + **调仓日前滚动综合权重**回测、`meta["rebalance_log"]`、`meta["decision_log"]` | `main` 未接 `run_multi_backtest(factors, weights)` 原始因子线性加权入口（代码已有，非主流程） |
| 绩效 `summarize`、股票池等权基准、超额收益 / 跟踪误差 / 信息比率、换手率与预估成本、HHI / effective_n 持仓集中度、净值/IC/权重/换手/集中度/覆盖率图、`performance_summary.csv`、`run_config.json`、调仓/决策审计/换手/集中度日志 CSV、`persist_run_outputs` 落盘 | 真实券商 API、实时风控与订单路由 |

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
output/         # 运行生成：nav_compare.png、excess_nav_compare.png、turnover_compare.png、performance_summary.csv、cache/、data_quality/、factor_diagnostics/、risk_exposure/ 等
factors/        # 因子与 panel_builder
backtest/       # backtest_single、backtest_multi、utils
models/         # fusion、factor_weighting、optimizer
analysis/       # performance、benchmark、turnover、risk_exposure、data_quality、plotting、ic
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
2. **因子面板**：`factors.panel_builder.build_four_factor_panel`（默认七列：`MOMENTUM`、`MOMENTUM_60D`、`REVERSAL_5D`、`VOLATILITY`、`VOLUME_RATIO_20D`、`PE`、`ROE`）。
3. **数据质量**：`analysis.data_quality` 输出价格覆盖、因子覆盖、调仓日覆盖报告；若 `persist_run_outputs`，保存到 `output/data_quality/`。
4. **落盘**：若 `persist_run_outputs`，`live.cache_io.save_run_cache` → `output/cache/`（`prices_long.csv`、`prices_wide_close.csv`、`factor_panel.csv`、`factor_panel_zscore.csv`、`run_meta.txt`）。
5. **IC 与稳定性诊断**：`analysis.ic` 对各因子列及 **与融合同构的** FUSED 得分算日截面 Spearman；同时输出 IC 分布分位数、正负占比和滚动稳定性；若 `persist_run_outputs`，另存 `ic_*.csv` 与 `output/ic_diagnostics/*.csv`。
6. **因子诊断**：`analysis.factor_diagnostics.batch_factor_long_excess` 对每个因子构造 Top-K 等权多头腿，并相对股票池等权基准输出 `excess_ann_return`、`tracking_error`、`information_ratio`；`batch_factor_group_returns` 按 `Settings.factor_group_count` 分组计算持有期收益、Top-Bottom、胜率与 `monotonicity_score`。
7. **多因子权重建议**：`models.factor_weighting.build_factor_weight_summary` 综合 IC 分布、rolling IC、Top-Bottom 与单调性，输出 `factor_score` 与 `fusion_weight`。全样本表用于诊断审计；训练段表用于 `FUSED_SCORE_WEIGHTED`；滚动权重日志用于 `FUSED_ROLLING_SCORE_WEIGHTED`。
8. **单因子回测**：对每列 `run_single_backtest(fname, factor_values=col, ...)`（**预计算因子**，不调注册表重算）。
9. **融合回测**：第一条是 **IC 滞后滚动列权（默认）或等权** z-score → `FUSED_ZSCORE`；第二条是训练段 `fusion_weight` 固定后应用到验证段 → `FUSED_SCORE_WEIGHTED`；第三条是每个调仓日前只用历史窗口重新计算权重 → `FUSED_ROLLING_SCORE_WEIGHTED`。三条都通过 `run_multi_backtest(fused=...)` 进入同一套 Top-K 回测。
10. **可交易性过滤**：若配置了 `min_avg_volume` 或 `min_avg_amount`，回测会在 Top-K 前按过去 `liquidity_lookback_days` 的平均成交量 / 成交额过滤候选股票；过滤前后候选数写入 `rebalance_log`。
11. **停牌 / 涨跌停约束**：若 `enable_trade_status_filter=True`，回测会读取 `is_suspended`、`is_limit_up`、`is_limit_down`，限制停牌买卖、涨停买入 / 加仓、跌停卖出 / 减仓，并把阻断原因写入 `decision_log`。
12. **决策审计日志**：回测同步生成 `meta["decision_log"]`，逐股票记录因子分数、排序、是否通过流动性过滤、是否入选、交易状态、上期权重、原始目标权重、最终目标权重、动作和原因标签。
13. **基准与超额收益**：`analysis.benchmark.equal_weight_benchmark_nav` 构造股票池等权基准；每条策略补 `benchmark_ann_return`、`excess_ann_return`、`tracking_error`、`information_ratio`。
14. **换手与成本**：`analysis.turnover` 从 `meta["rebalance_log"]` 估算逐期 `turnover`、`estimated_cost`，并汇总 `avg_turnover`、`total_turnover`、`estimated_total_cost`。
15. **风险暴露与集中度**：`analysis.risk_exposure` 从同一份调仓日志计算 `hhi`、`effective_n`、`top1_weight`、`top3_weight` 等，判断策略是否过度集中。
16. **实验记录与作图**：若 `persist_run_outputs`，保存 `output/cache/run_config.json`、`output/performance_summary.csv`、`output/ic_diagnostics/*.csv`、`output/factor_diagnostics/long_excess_summary.csv`、`output/factor_diagnostics/group_return_detail.csv`、`output/factor_diagnostics/group_return_summary.csv`、`output/factor_diagnostics/factor_weight_summary.csv`、`output/factor_diagnostics/factor_weight_train_summary.csv`、`output/factor_diagnostics/rolling_factor_weight_log.csv`、`output/rebalance_logs/*.csv`、`output/decision_logs/*.csv`、`output/turnover_logs/*.csv`、`output/risk_exposure/*.csv`、`output/data_quality/*.csv`、`ic_compare.png`、`ic_timeseries_*.png`、`weights_*.png`、`turnover_compare.png`、`risk_exposure/effective_n_compare.png`；`plot_nav` → `output/nav_compare.png`，超额净值 → `output/excess_nav_compare.png`。

### 回测与配置要点

- **再平衡**：默认 `config.rebalance_freq = "ME"`（月末）；**Top-K** 默认 `top_k=5`；因子截面**降序**取前 K。
- **IC 稳定性**：`config.ic_rolling_windows` 默认 `(20, 60)`；诊断层会统计 IC 分位数、正负占比、滚动均值和滚动正值比例。
- **因子分组**：`config.factor_group_count` 默认 `5`；诊断层按因子从低到高分组，`G1` 为低分组，`G5` 为高分组，观察 Top-Bottom 与单调性。
- **多因子权重建议**：全样本 `factor_weight_summary.csv` 用于观察权重是否合理；训练段 `factor_weight_train_summary.csv` 生成 `FUSED_SCORE_WEIGHTED`；滚动日志 `rolling_factor_weight_log.csv` 记录每个调仓日前实际使用的因子权重，并生成 `FUSED_ROLLING_SCORE_WEIGHTED`。
- **滚动因子权重保护**：`rolling_factor_weight_lookback_days` 控制历史窗口，`rolling_factor_weight_min_days` 控制最少历史样本，`rolling_factor_weight_min_weight` / `rolling_factor_weight_max_weight` 控制因子权重上下限，`rolling_factor_weight_smoothing` 控制新旧权重平滑。
- **可交易性过滤**：`min_avg_volume` / `min_avg_amount` 默认为 `0`，表示关闭；设为正数后，回测会在 Top-K 前用过去 `liquidity_lookback_days` 的平均成交量 / 成交额过滤候选股票。调仓日志会记录 `n_candidates_before_liquidity`、`n_candidates_after_liquidity`、`liquidity_filter_enabled` 等字段。
- **停牌 / 涨跌停约束**：`enable_trade_status_filter` 默认 `False`；开启后，回测读取 `long_prices` / `trade_status_data` 中的 `is_suspended`、`is_limit_up`、`is_limit_down`。停牌不能买卖，涨停不能买入 / 加仓，跌停不能卖出 / 减仓；`decision_logs` 会记录 `trade_blocked` 与 `trade_block_reason`。
- **持仓权重**：`config.portfolio_weighting`：`"equal"`、**`"max_sharpe"`（当前默认）** 或 **`"risk_parity"`**；后两者在再平衡日对 Top-K 用历史日收益估协方差（夏普另需 μ），分别调用 `models.optimizer.maximize_sharpe` / `risk_parity`，样本不足等失败则等权。
- **单票权重上限**：`config.max_position_weight` 默认 `0.4`；当优化权重可行且超过上限时，会裁剪并重新分配剩余权重，`rebalance_log[].weighting` 记录为 `max_sharpe_capped` / `risk_parity_capped` 等。
- **单次换手上限**：`config.max_rebalance_turnover` 默认 `1.0`；首次建仓不节流，之后若目标权重变化超过上限，会按比例向新目标移动，`rebalance_log` 记录 `target_turnover`、`turnover_capped`、`turnover_scale`。
- **调仓记录**：`meta["rebalance_log"]`；`main` 会打印每期标的与权重，并在 `persist_run_outputs=True` 时保存到 `output/rebalance_logs/*.csv`，其中包含流动性过滤前后候选数。
- **决策审计记录**：`meta["decision_log"]`；在 `persist_run_outputs=True` 时保存到 `output/decision_logs/*.csv`，逐股票解释 `buy` / `sell` / `increase` / `decrease` / `hold` / `skip` 及原因。
- **换手记录**：`analysis.turnover` 以调仓目标权重变化估算成交占比，并在 `persist_run_outputs=True` 时保存到 `output/turnover_logs/*.csv`。
- **集中度记录**：`analysis.risk_exposure` 以 HHI 与 effective_n 衡量持仓是否集中，并在 `persist_run_outputs=True` 时保存到 `output/risk_exposure/`。

### 依赖

见 `requirements.txt`（含 **pandas、numpy、scipy、matplotlib、scikit-learn、tushare** 等）。

### 非 MVP（占位或扩展）

- `live/signal_system.generate_signals`、`live/paper_trading.run_paper_trading`
- `models.fusion.fuse_models`：仅 `mean_zscore` / `mean` 可用，其它 `method` 会报错

### 测试

```bash
python3 -m unittest tests.test_optimizer tests.test_backtest_multi tests.test_backtest_single tests.test_plotting tests.test_fusion tests.test_cache_io tests.test_benchmark tests.test_turnover tests.test_data_quality tests.test_risk_exposure tests.test_factors tests.test_factor_preprocess tests.test_factor_diagnostics tests.test_ic tests.test_factor_weighting -v
```
