# Quant Strategy（MVP）

模块化量化研究项目：**数据 → 因子面板 → 因子清洗与行业内标准化 → 数据质量 → IC → 因子诊断（Top-K 多头超额 + 分组收益单调性）→ 多因子权重建议 → 样本外验证与因子失效监控 → 融合回测（IC 滚动列权 + 训练段静态综合权重 + 调仓日前滚动综合权重）→ 可交易性 / 流动性过滤 → 回测（Top-K + 等权 / 夏普 / 风险平价）→ 单票 / 行业 / 波动率 / 最小持仓约束 → 决策审计日志 → 基准与超额收益 → 换手与成本 → 风险暴露与集中度 → 绩效与作图 → 实验记录落盘 → 目标权重转订单计划 → 订单预检查 → 纸面交易 → 账户状态持久化 → 每日纸面交易运行器 → 日终纸面交易脚本 → 纸面交易日报 → 运行失败 / 异常检查 → 交易日日历 / 重复运行保护 → 每日调度入口 → 统一券商接口协议 / 模拟券商适配器 → 日终纸面交易接入统一券商接口 → 真实券商只读 Adapter 骨架**。

**文档与代码**：以 `main.py` 与 `config.Settings` 为准；更新行为后请同步修改 `docs/ENGINEERING_OVERVIEW.md`、`docs/FLOW_AND_MODULES.md` 及本 README 相关段落（仓库无自动文档校验）。

### MVP 定稿（范围）

**本仓库 MVP 已交付**，指下面闭环可稳定跑通、用于研究与对内演示；不要求实盘下单。

| 在 MVP 内 | 不在 MVP 内（后续扩展） |
|------------|-------------------------|
| 行情接入（CSV / Tushare / 合成兜底）、多因子面板（量价 + 估值 + 质量 + 成长 + 现金流）、横截面/行业内标准化、数据质量 / 覆盖率报告、IC 与可选 CSV/图、因子 Top-K 多头超额诊断、分组收益与单调性分析、多因子权重建议表、样本外验证与因子失效监控 | `live/signal_system.generate_signals`、真实券商 API |
| 月末再平衡、Top-K、可交易性 / 流动性过滤、停牌 / 涨跌停交易约束、`portfolio_weighting`：`equal` / `max_sharpe` / `risk_parity`，`max_position_weight` 单票权重上限，`max_industry_weight` 行业权重上限，`target_volatility` 波动率目标与现金仓位，`min_positions` 最小持仓数量，`max_rebalance_turnover` 单次换手上限 | `fuse_models` 除 `mean_zscore` / `mean` 外的 `method`（如 `dynamic`、`xgboost`） |
| 单因子回测 + **IC 驱动或等权** z-score 融合回测 + **训练段静态综合权重**验证回测 + **调仓日前滚动综合权重**回测、`meta["rebalance_log"]`、`meta["decision_log"]` | `main` 未接 `run_multi_backtest(factors, weights)` 原始因子线性加权入口（代码已有，非主流程） |
| 绩效 `summarize`、股票池等权基准、超额收益 / 跟踪误差 / 信息比率、换手率与预估成本、HHI / effective_n 持仓集中度、净值/IC/权重/换手/集中度/覆盖率图、`performance_summary.csv`、`run_config.json`、调仓/决策审计/换手/集中度日志 CSV、`persist_run_outputs` 落盘、`live.order_builder` 目标权重转订单计划、`live.order_precheck` 订单预检查、`live.paper_trading` 虚拟账户模拟成交、`live.account_state` 纸面账户状态持久化、`live.paper_runner` 单日纸面交易运行器、`scripts/run_daily_paper.py` 日终纸面交易脚本、`live.paper_report` 纸面交易日报与因子健康状态、`live.manual_confirmation` 小资金人工确认实盘单、`live.execution_feedback` 真实成交回填与执行偏差分析、`live.paper_guard` 运行失败 / 异常检查、`live.paper_run_control` 交易日日历 / 重复运行保护、`scripts/run_scheduled_daily_paper.py` 每日调度入口、`live.broker` 统一券商接口协议与模拟券商适配器、真实券商只读 Adapter 骨架、日终纸面交易可通过 `--execution-mode simulated_broker` 走统一券商接口 | 真实券商交易 API、实时风控与订单路由 |

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

Token：优先环境变量 `TUSHARE_TOKEN`；未设置时使用 `config.py` 内 `_TUSHARE_TOKEN_LOCAL`（**勿将含真实密钥的 config 提交远程**）。真实股票池默认读取 `data/stock_pool.xlsx`，也可用环境变量 `QUANT_STOCK_POOL_PATH` 指向本机 Excel/CSV；行情拉取后默认缓存到 `data/prices_tushare_cache.csv`，也可用 `QUANT_TUSHARE_PRICE_CACHE` 改路径。

## 目录结构

```
data/           # 原始/演示数据（如 prices_demo.csv、stock_pool.xlsx、本地行情缓存）
output/         # 运行生成：nav_compare.png、excess_nav_compare.png、turnover_compare.png、performance_summary.csv、cache/、data_quality/、factor_diagnostics/、risk_exposure/ 等
factors/        # 因子与 panel_builder
backtest/       # backtest_single、backtest_multi、utils
models/         # fusion、factor_weighting、optimizer
analysis/       # performance、benchmark、turnover、risk_exposure、data_quality、factor_diagnostics、factor_validation、plotting、ic
live/           # data_feed、stock_pool、cache_io、order_builder、order_precheck、paper_trading、broker、account_state、paper_runner、paper_report、manual_confirmation、execution_feedback、paper_guard、paper_run_control、daily_paper_cli；signal 非 MVP 占位
scripts/        # build_live_universe.py、run_daily_paper.py、build_execution_feedback.py、run_scheduled_daily_paper.py 等日常运行入口
config.py
main.py
```

## 运行

```bash
python main.py
```

### 当前 `main.py` 实际顺序（与代码一致）

1. **数据**：`data/prices_demo.csv` 优先；否则读取 Tushare 行情缓存；若缓存不存在，则从 `Settings.stock_pool_path` 指定的 Excel/CSV 股票池读取标的并拉取 Tushare 日线，同时写入 `Settings.tushare_price_cache_path`；若股票池不存在才使用 `main._DEFAULT_TS_SYMBOLS` 示例股票池；失败则合成宽表。得到 `prices`（宽表）与 `long_df`。
2. **基础因子面板**：`factors.panel_builder.build_four_factor_panel`（历史命名保留；当前默认十四列：`MOMENTUM`、`MOMENTUM_60D`、`REVERSAL_5D`、`VOLATILITY`、`VOLUME_RATIO_20D`、`PE`、`ROE`、`GROSS_MARGIN`、`NET_MARGIN`、`LOW_DEBT_TO_ASSETS`、`REVENUE_GROWTH`、`PROFIT_GROWTH`、`FREE_CASH_FLOW_YIELD`、`CASH_PROFIT_QUALITY`）。
3. **机器学习打分因子**：若 `enable_ml_score=True`，`factors.factor_ml.build_ml_score_factor` 用已有因子面板滚动训练梯度提升类模型，预测未来收益并追加 `ML_SCORE`；该列只作为候选因子进入后续 IC、分组收益、样本外验证和回测。
4. **因子清洗与行业内标准化**：`factors.preprocess.preprocess_factor_panel` 默认读取 `industry_col` 行业字段，在同一交易日同一行业内做 winsorize + z-score；缺行业或行业样本少于 `factor_industry_min_count` 时回退全股票池横截面 z-score。原始 `factor_panel.csv` 保留审计，IC/诊断/回测使用标准化后的研究面板。
5. **数据质量**：`analysis.data_quality` 输出价格覆盖、因子覆盖、调仓日覆盖报告；若 `persist_run_outputs`，保存到 `output/data_quality/`。
6. **落盘**：若 `persist_run_outputs`，`live.cache_io.save_run_cache` → `output/cache/`（`prices_long.csv`、`prices_wide_close.csv`、`factor_panel.csv`、`factor_panel_zscore.csv`、`run_meta.txt`）；若生成 `ML_SCORE`，另写 `output/factor_diagnostics/ml_score_training_log.csv`。
7. **IC 与稳定性诊断**：`analysis.ic` 对各因子列及 **与融合同构的** FUSED 得分算日截面 Spearman；同时输出 IC 分布分位数、正负占比和滚动稳定性；若 `persist_run_outputs`，另存 `ic_*.csv` 与 `output/ic_diagnostics/*.csv`。
8. **因子诊断**：`analysis.factor_diagnostics.batch_factor_long_excess` 对每个因子构造 Top-K 等权多头腿，并相对股票池等权基准输出 `excess_ann_return`、`tracking_error`、`information_ratio`；`batch_factor_group_returns` 按 `Settings.factor_group_count` 分组计算持有期收益、Top-Bottom、胜率与 `monotonicity_score`。
9. **多因子权重建议**：`models.factor_weighting.build_factor_weight_summary` 综合 IC 分布、rolling IC、Top-Bottom 与单调性，输出 `factor_score` 与 `fusion_weight`。全样本表用于诊断审计；训练段表用于 `FUSED_SCORE_WEIGHTED`；滚动权重日志用于 `FUSED_ROLLING_SCORE_WEIGHTED`；`analysis.factor_weight_stability` 进一步监控滚动权重的稳定性、漂移事件和组合层主导因子。
10. **样本外验证与因子失效监控**：`analysis.factor_validation` 按 `factor_weight_train_ratio` 切成训练段和验证段，分别计算 IC、多头超额、Top-Bottom 与单调性，并生成 `OK/WATCH/DEGRADED/FAILED` 状态表；同时可按滚动窗口输出 `rolling_out_of_sample_validation.csv` 与 `rolling_out_of_sample_summary.csv`，观察因子跨时间窗口是否稳定。
10A. **多股票池验证**：`analysis.multi_universe_validation` 与 `scripts/build_multi_universe_validation.py` 可读取多个已完成回测的 output 目录，汇总策略绩效和因子多头超额，输出跨股票池稳健性表。
11. **单因子回测**：对每列 `run_single_backtest(fname, factor_values=col, ...)`（**预计算因子**，不调注册表重算）。
12. **融合回测**：第一条是 **IC 滞后滚动列权（默认）或等权** z-score → `FUSED_ZSCORE`；第二条是训练段 `fusion_weight` 固定后应用到验证段 → `FUSED_SCORE_WEIGHTED`；第三条是每个调仓日前只用历史窗口重新计算权重 → `FUSED_ROLLING_SCORE_WEIGHTED`。三条都通过 `run_multi_backtest(fused=...)` 进入同一套 Top-K 回测。
13. **可交易性过滤**：若配置了 `min_avg_volume` 或 `min_avg_amount`，回测会在 Top-K 前按过去 `liquidity_lookback_days` 的平均成交量 / 成交额过滤候选股票；过滤前后候选数写入 `rebalance_log`。
14. **行业权重上限**：若 `max_industry_weight` 在 `(0, 1)`，回测会读取 `industry_col` 指定的行业字段，在目标权重生成后限制单个行业暴露，并把 `industry_cap_applied`、`max_industry_exposure`、`n_industries` 写入调仓日志。
15. **波动率目标**：若 `target_volatility > 0`，回测会用历史协方差估算目标组合年化波动；当估算波动超过目标时，只降低股票仓位、不加杠杆，剩余权重作为现金，并记录 `volatility_target_scale`、`cash_target_weight`。
16. **最小持仓数量**：若 `min_positions > 0` 且有效目标持仓数不足，回测会把股票总仓位缩到 `min_positions_exposure`，剩余作为现金，并把 `min_positions_applied` 写入日志。
17. **停牌 / 涨跌停约束**：若 `enable_trade_status_filter=True`，回测会读取 `is_suspended`、`is_limit_up`、`is_limit_down`，限制停牌买卖、涨停买入 / 加仓、跌停卖出 / 减仓，并把阻断原因写入 `decision_log`。
18. **决策审计日志**：回测同步生成 `meta["decision_log"]`，逐股票记录因子分数、排序、是否通过流动性过滤、是否入选、所属行业、交易状态、上期权重、原始目标权重、最终目标权重、动作和原因标签。
19. **风格层暴露与收益关联**：`analysis.style_exposure` 从融合策略调仓日志、复合风格分数和净值曲线计算逐期风格暴露、暴露汇总与暴露-下一期收益关联，解释策略实际偏向量价、质量等哪类风格。
20. **基准与超额收益**：`analysis.benchmark.equal_weight_benchmark_nav` 构造股票池等权基准；每条策略补 `benchmark_ann_return`、`excess_ann_return`、`tracking_error`、`information_ratio`。
21. **换手与成本**：`analysis.turnover` 从 `meta["rebalance_log"]` 估算逐期 `turnover`、`estimated_cost`，并汇总 `avg_turnover`、`total_turnover`、`estimated_total_cost`。
22. **风险暴露与集中度**：`analysis.risk_exposure` 从同一份调仓日志计算 `hhi`、`effective_n`、`top1_weight`、`top3_weight` 等，判断策略是否过度集中。
23. **实验记录与作图**：若 `persist_run_outputs`，保存 `output/cache/run_config.json`、`output/performance_summary.csv`、`output/ic_diagnostics/*.csv`、`output/factor_diagnostics/long_excess_summary.csv`、`output/factor_diagnostics/group_return_detail.csv`、`output/factor_diagnostics/group_return_summary.csv`、`output/factor_diagnostics/factor_weight_summary.csv`、`output/factor_diagnostics/factor_weight_train_summary.csv`、`output/factor_diagnostics/rolling_factor_weight_log.csv`、`output/factor_diagnostics/factor_weight_stability_summary.csv`、`output/factor_diagnostics/factor_weight_drift_events.csv`、`output/factor_diagnostics/factor_weight_portfolio_drift.csv`、`output/factor_diagnostics/style_exposure*.csv`、`output/factor_diagnostics/ml_score_training_log.csv`、`output/factor_validation/out_of_sample_validation.csv`、`output/factor_validation/factor_decay_monitor.csv`、`output/factor_validation/rolling_out_of_sample_validation.csv`、`output/factor_validation/rolling_out_of_sample_summary.csv`、`output/rebalance_logs/*.csv`、`output/decision_logs/*.csv`、`output/turnover_logs/*.csv`、`output/risk_exposure/*.csv`、`output/data_quality/*.csv`、`ic_compare.png`、`ic_timeseries_*.png`、`weights_*.png`、`turnover_compare.png`、`risk_exposure/effective_n_compare.png`；`plot_nav` → `output/nav_compare.png`，超额净值 → `output/excess_nav_compare.png`。多股票池验证脚本另写 `output/multi_universe_validation/*.csv`。
23. **订单计划**：`live.order_builder` 可把最近一期目标权重、当前持仓、现金 / 总资产和最新价格转换成 `BUY/SELL`、目标股数、调整股数、预估金额与交易原因；`live.cache_io.save_order_plans` 可保存到 `output/order_plans/*.csv`。该层不连接券商、不模拟成交。
24. **订单预检查**：`live.order_precheck` 对订单计划做现金、可卖数量、买入手数、最小订单金额、停牌 / 涨停买入 / 跌停卖出检查；`live.cache_io.save_order_checks` 可保存到 `output/order_checks/*.csv`。该层只输出 `PASS/BLOCK` 和原因，不修改订单。
25. **纸面交易**：`live.paper_trading` 只执行通过预检查的订单，按手续费更新虚拟现金和持仓，记录 `FILLED/SKIPPED`、现金变化、持仓变化与原因；`live.cache_io.save_paper_trades` 可保存到 `output/paper_trades/*.csv`。
26. **纸面账户状态**：`live.account_state` 保存 / 读取纸面账户现金、持仓和每日快照，输出到 `output/paper_account/<strategy>/account.csv`、`positions.csv`、`snapshots.csv`。
27. **每日纸面交易运行器**：`live.paper_runner.run_daily_paper_trade` 读取上一日纸面账户状态，串联订单生成、预检查、成交执行、持仓更新、账户快照和落盘，形成可每天调用一次的纸面交易入口。默认沿用 `paper_trading`，也可用 `execution_mode="simulated_broker"` 通过统一券商接口执行。
28. **日终纸面交易脚本**：`scripts/run_daily_paper.py` 从 `output/rebalance_logs/<strategy>.csv` 读取最近一期目标权重，从 `output/cache/prices_wide_close.csv` 读取最新价格，调用每日纸面交易运行器并打印摘要。
29. **纸面交易日报**：`live.paper_report` 将单日纸面运行结果整理为 Markdown，默认写入 `output/paper_reports/<strategy>/<date>.md`，便于复盘每天买卖、阻断、成交、持仓、账户变化和因子健康状态。
30. **小资金人工确认实盘单**：`live.manual_confirmation` 基于同一份订单计划和预检查结果生成 `output/live_orders/<strategy>/<date>_manual_confirm.csv/.md`，预留人工执行回填字段；该层只给建议，不自动下单。
31. **真实成交回填与执行偏差分析**：`live.execution_feedback` 读取人工确认单中回填的 `executed_qty/executed_price`，比较建议订单和真实执行，输出 `output/execution_feedback/<strategy>/` 下的逐笔偏差、汇总和 Markdown 报告。
32. **运行失败 / 异常检查**：`live.paper_guard` 在日终纸面交易前后检查目标权重、价格日期、价格有效性、账户现金、持仓、订单检查和成交日志；ERROR 级问题直接阻断运行，WARNING 级问题进入命令摘要与日报。
33. **交易日日历 / 重复运行保护**：`live.paper_run_control` 从价格缓存提取交易日日历，默认阻断非交易日运行；若同一策略同一日期已有纸面账户快照，默认阻断重复写入，避免无意覆盖账户状态。
34. **每日调度入口**：`scripts/run_scheduled_daily_paper.py` 包装日终纸面交易命令，适合交给 cron / launchd / 服务器调度器调用，并把 stdout、stderr、参数和退出码写入 `output/scheduler_logs/<date>.log`。
35. **实盘目标池确认**：`scripts/build_live_universe.py` 从人工股票池和价格缓存生成 `stock_pool_filter_report_<date>.csv` 与 `active_universe_<date>.csv`，记录哪些股票通过、哪些被剔除以及剔除原因。后续纸面交易和券商接口应优先读取确认后的 active universe。
36. **统一券商接口协议与通道 Factory**：`live.broker` 定义 `BrokerAdapter`、`BrokerAccount`、`BrokerPosition`、`BrokerOrder`，并提供 `SimulatedBroker`；`live.broker_factory` 根据 `broker_mode/broker_provider` 创建模拟、只读或后续真实 Adapter。策略与订单层只依赖 `get_account/get_positions/get_orders/submit_order/cancel_order` 等统一方法；未来 QMT / PTrade / 掘金只需实现同一协议并注册到 Factory。
37. **日终纸面交易接入统一券商接口**：`scripts/run_daily_paper.py --execution-mode simulated_broker` 可把日终订单计划交给 `SimulatedBroker` 执行，并保留原有 `paper_trades`、账户状态和 Markdown 日报输出，同时额外返回统一券商订单回报 `broker_orders`。
38. **真实券商只读 Adapter 骨架**：`live.broker.RealBrokerReadOnlyAdapter` 和 `RealBrokerConfig` 定义真实券商接入的只读入口，可查询账户、持仓和订单快照；`submit_order/cancel_order` 会抛出 `BrokerReadOnlyError`，防止尚未验证前误下单。
39. **纸面账户 / 真实账户只读对账**：`live.broker_reconcile` 对比纸面账户和只读券商账户的现金、总资产和逐股票持仓差异；`scripts/reconcile_paper_broker.py` 可读取券商导出的账户 / 持仓 CSV，输出 `output/broker_reconciliation/<strategy>/` 下的 CSV 与 Markdown 对账报告。

### 日终纸面交易

先运行 `python main.py` 生成目标权重和价格缓存，再运行：

```bash
python scripts/run_daily_paper.py
```

常用参数：

```bash
python scripts/run_daily_paper.py --strategy FUSED_ROLLING_SCORE_WEIGHTED
python scripts/run_daily_paper.py --trade-date 2024-01-26
python scripts/run_daily_paper.py --trade-status data/trade_status.csv
python scripts/run_daily_paper.py --no-persist
python scripts/run_daily_paper.py --no-report
python scripts/run_daily_paper.py --no-manual-confirm
python scripts/run_daily_paper.py --factor-decay-monitor output/factor_validation/factor_decay_monitor.csv
python scripts/run_daily_paper.py --no-guard
python scripts/run_daily_paper.py --max-price-age-days 3
python scripts/run_daily_paper.py --allow-non-trading-day
python scripts/run_daily_paper.py --allow-rerun
python scripts/run_daily_paper.py --execution-mode simulated_broker
```

脚本默认读取 `output/rebalance_logs/<strategy>.csv` 与 `output/cache/prices_wide_close.csv`，输出订单计划、订单预检查、纸面成交、纸面账户状态、Markdown 日报和小资金人工确认单。`--no-persist` 可用于只检查流程和摘要，不写账户文件；`--no-report` 可只写 CSV 与账户状态，不生成日报；`--no-manual-confirm` 可关闭人工确认单；`--factor-decay-monitor` 可指定因子失效监控 CSV，并写入命令摘要、Markdown 日报和人工确认单；`--style-exposure` 可指定 `style_exposure.csv`，默认读取 `output/factor_diagnostics/style_exposure.csv` 并把最近一期目标组合风格暴露写入命令摘要和 Markdown 日报；`--no-guard` 可临时关闭运行检查；`--max-price-age-days` 控制价格日期超过多少自然日后给出 stale warning；`--allow-non-trading-day` 允许在非交易日强制运行；`--allow-rerun` 允许覆盖同一交易日已有纸面账户快照；`--execution-mode simulated_broker` 可让日终流程通过统一模拟券商执行订单。

### 纸面账户 / 券商只读对账

当真实券商或量化终端能导出账户和持仓 CSV 后，可先做只读对账，不下单：

```bash
python scripts/reconcile_paper_broker.py \
  --strategy FUSED_ROLLING_SCORE_WEIGHTED \
  --trade-date 2026-06-22 \
  --broker-account data/broker_account.csv \
  --broker-positions data/broker_positions.csv
```

账户 CSV 至少包含 `cash/market_value/total_asset`；持仓 CSV 至少包含 `symbol/shares`，可选 `available_shares`。输出包括账户差异、持仓差异和 Markdown 对账报告。

### 真实成交回填

人工在券商终端执行后，把真实成交数量和价格回填到人工确认单，再运行：

```bash
python scripts/build_execution_feedback.py \
  --strategy FUSED_ROLLING_SCORE_WEIGHTED \
  --trade-date 2026-06-22
```

也可显式指定确认单：

```bash
python scripts/build_execution_feedback.py \
  --manual-confirm output/live_orders/FUSED_ROLLING_SCORE_WEIGHTED/2026-06-22_manual_confirm.csv
```

脚本输出到 `output/execution_feedback/<strategy>/`，包括逐笔执行偏差、汇总表和 Markdown 报告。该步骤只分析真实执行结果，不修改纸面账户、不连接券商。

### 每日调度入口

调度入口只负责“运行一次并记录日志”，不在 Python 内部常驻循环。可把它交给 cron、launchd 或服务器调度器：

```bash
python scripts/run_scheduled_daily_paper.py --strategy FUSED_ROLLING_SCORE_WEIGHTED
python scripts/run_scheduled_daily_paper.py --log-date 2024-01-26 --strategy TEST --no-persist
```

未识别参数会透传给 `run_daily_paper.py`，调度日志写到 `output/scheduler_logs/<date>.log`。脚本退出码与日终纸面交易一致，便于系统调度器判断成功或失败。

### 实盘目标池确认

在接券商接口前，先把人工研究池过滤成当日可用的 active universe：

```bash
python scripts/build_live_universe.py \
  --stock-pool data/stock_pool.xlsx \
  --prices output/cache/prices_wide_close.csv \
  --trade-date 2026-06-23
```

输出默认写入 `output/live_universe/`：

- `stock_pool_filter_report_<date>.csv`：完整过滤报告，含 `active` 与 `exclude_reason`。
- `active_universe_<date>.csv`：当天确认后的实盘目标池，只保留通过过滤的股票。

多股票池可用 `--output-subdir live_universe/<pool_name>` 分目录保存，避免同一日期互相覆盖。

### 回测与配置要点

- **再平衡**：默认 `config.rebalance_freq = "ME"`（月末）；**Top-K** 默认 `top_k=5`；因子截面**降序**取前 K。
- **IC 稳定性**：`config.ic_rolling_windows` 默认 `(20, 60)`；诊断层会统计 IC 分位数、正负占比、滚动均值和滚动正值比例。
- **因子分组**：`config.factor_group_count` 默认 `5`；诊断层按因子从低到高分组，`G1` 为低分组，`G5` 为高分组，观察 Top-Bottom 与单调性。
- **机器学习打分因子**：`enable_ml_score=True` 时，`ML_SCORE` 会用已有因子特征滚动训练并预测未来 `ml_score_forward_days` 日收益；`ml_score_model` 可设为 `lightgbm`、`catboost`、`xgboost`、`hist_gradient_boosting` 或 `auto`，缺少可选依赖时会回退到 sklearn 实现。它只是候选因子，仍需经过 IC、分组收益、样本外验证和回测。
- **行业内标准化**：`factor_standardize_by_industry=True` 默认开启；`main.py` 会从股票池 `子行业` / `分类` 或行情长表 `industry_col` 读取行业，在同一交易日同一行业内做 winsorize + z-score。若行业缺失或行业样本少于 `factor_industry_min_count`，该部分回退全股票池横截面 z-score。可用 `QUANT_FACTOR_STANDARDIZE_BY_INDUSTRY=0` 做关闭对照。
- **多因子权重建议**：全样本 `factor_weight_summary.csv` 用于观察权重是否合理；训练段 `factor_weight_train_summary.csv` 生成 `FUSED_SCORE_WEIGHTED`；滚动日志 `rolling_factor_weight_log.csv` 记录每个调仓日前实际使用的因子权重，并生成 `FUSED_ROLLING_SCORE_WEIGHTED`；`factor_weight_stability_summary.csv`、`factor_weight_drift_events.csv`、`factor_weight_portfolio_drift.csv` 用于观察滚动权重是否稳定、是否有跳变、是否被单一因子主导。
- **样本外验证与因子失效监控**：`analysis.factor_validation` 复用 IC、多头超额和分组收益口径，按 `factor_weight_train_ratio` 切分训练段 / 验证段，保存 `output/factor_validation/out_of_sample_validation.csv` 与 `factor_decay_monitor.csv`；滚动样本外验证按 `rolling_oos_train_days` / `rolling_oos_validation_days` / `rolling_oos_step_days` 多窗口复查因子稳定性，保存 `rolling_out_of_sample_validation.csv` 与 `rolling_out_of_sample_summary.csv`。
- **多股票池验证**：`scripts/build_multi_universe_validation.py` 读取多个已完成回测输出目录，生成 `strategy_universe_performance.csv`、`strategy_universe_robustness.csv`、`factor_universe_performance.csv`、`factor_universe_robustness.csv`，用于判断策略和因子是否只在某一个股票池里有效。
- **参数敏感性分析**：`scripts/build_parameter_sensitivity.py` 复用已有 `output/cache` 中的价格和标准化因子面板，对 `top_k`、调仓频率、配权方式、单票上限、换手上限、波动率目标等参数做一维扰动，生成 `parameter_sensitivity_detail.csv` 与 `parameter_sensitivity_summary.csv`，用于判断策略是否过度依赖某个精确参数。
- **滚动因子权重保护**：`rolling_factor_weight_lookback_days` 控制历史窗口，`rolling_factor_weight_min_days` 控制最少历史样本，`rolling_factor_weight_min_weight` / `rolling_factor_weight_max_weight` 控制因子权重上下限，`rolling_factor_weight_smoothing` 控制新旧权重平滑。
- **可交易性过滤**：`min_avg_volume` / `min_avg_amount` 默认为 `0`，表示关闭；设为正数后，回测会在 Top-K 前用过去 `liquidity_lookback_days` 的平均成交量 / 成交额过滤候选股票。调仓日志会记录 `n_candidates_before_liquidity`、`n_candidates_after_liquidity`、`liquidity_filter_enabled` 等字段。
- **停牌 / 涨跌停约束**：`enable_trade_status_filter` 默认 `False`；开启后，回测读取 `long_prices` / `trade_status_data` 中的 `is_suspended`、`is_limit_up`、`is_limit_down`。停牌不能买卖，涨停不能买入 / 加仓，跌停不能卖出 / 减仓；`decision_logs` 会记录 `trade_blocked` 与 `trade_block_reason`。
- **持仓权重**：`config.portfolio_weighting`：`"equal"`、**`"max_sharpe"`（当前默认）** 或 **`"risk_parity"`**；后两者在再平衡日对 Top-K 用历史日收益估协方差（夏普另需 μ），分别调用 `models.optimizer.maximize_sharpe` / `risk_parity`，样本不足等失败则等权。
- **单票权重上限**：`config.max_position_weight` 默认 `0.4`；当优化权重可行且超过上限时，会裁剪并重新分配剩余权重，`rebalance_log[].weighting` 记录为 `max_sharpe_capped` / `risk_parity_capped` 等。
- **行业权重上限**：`config.max_industry_weight` 默认 `0`，表示关闭；设为 `(0, 1)` 后，回测会从 `long_prices` / `industry_data` 中读取 `industry_col`（默认 `industry`），限制单个行业目标权重，避免组合因为 Top-K 或优化器把资金集中到同一行业。
- **波动率目标**：`config.target_volatility` 默认 `0`，表示关闭；设为正数后，回测用 `volatility_target_lookback_days` 的历史收益协方差估算组合年化波动，若超过目标则按比例降低股票仓位，剩余记为现金。该 MVP 只降风险，不主动加杠杆。
- **最小持仓数量**：`config.min_positions` 默认 `0`，表示关闭；开启后若有效目标持仓数少于阈值，会把股票总仓位缩到 `min_positions_exposure`，剩余记为现金，避免可交易标的不足时硬满仓。
- **订单生成、预检查、纸面交易、账户状态、日报、人工确认单、真实成交回填、异常检查、运行控制与调度入口**：`config.order_lot_size` 默认 `100`，用于 A 股一手约束；`config.min_order_amount` 默认 `0`，可过滤金额太小的碎片订单；`config.order_cash_buffer` 默认 `0`，用于买入后现金缓冲检查；`config.paper_initial_cash` 默认 `1_000_000`，用于虚拟账户初始化。`live.paper_runner.run_daily_paper_trade` 可把这些步骤串成单日纸面交易流程，`scripts/run_daily_paper.py` 可从已有回测输出里自动读取输入并运行，`--execution-mode simulated_broker` 可通过统一模拟券商执行订单，`live.paper_report` 会生成 Markdown 日报并展示因子健康与目标组合风格暴露，`live.style_exposure_monitor` 会从 `style_exposure.csv` 读取当前策略最近一期风格暴露，`live.manual_confirmation` 会生成小资金人工确认单，`live.execution_feedback` 会读取人工回填后的确认单并生成执行偏差报告，`live.paper_guard` 会在运行前后拦截 ERROR 级异常并记录 WARNING 级风险提示，`live.paper_run_control` 会默认阻断非交易日运行和重复覆盖同日快照，`scripts/run_scheduled_daily_paper.py` 可作为系统调度器的单次运行入口。`config.broker_mode` 默认 `simulated`，`live.broker_factory.create_broker_adapter` 可按 `broker_mode/broker_provider` 创建模拟或只读 Adapter；真实券商建议先用 `real_readonly` 验证资金、持仓和订单读取。当前不真实下单。
- **单次换手上限**：`config.max_rebalance_turnover` 默认 `1.0`；首次建仓不节流，之后若目标权重变化超过上限，会按比例向新目标移动，`rebalance_log` 记录 `target_turnover`、`turnover_capped`、`turnover_scale`。
- **调仓记录**：`meta["rebalance_log"]`；`main` 会打印每期标的与权重，并在 `persist_run_outputs=True` 时保存到 `output/rebalance_logs/*.csv`，其中包含流动性过滤前后候选数、行业上限是否触发、最大行业暴露、目标波动缩放比例、最小持仓检查和现金目标仓位等。
- **决策审计记录**：`meta["decision_log"]`；在 `persist_run_outputs=True` 时保存到 `output/decision_logs/*.csv`，逐股票解释 `buy` / `sell` / `increase` / `decrease` / `hold` / `skip` 及原因，并记录所属行业与行业上限调整标记。
- **换手记录**：`analysis.turnover` 以调仓目标权重变化估算成交占比，并在 `persist_run_outputs=True` 时保存到 `output/turnover_logs/*.csv`。
- **集中度记录**：`analysis.risk_exposure` 以 HHI 与 effective_n 衡量持仓是否集中，并在 `persist_run_outputs=True` 时保存到 `output/risk_exposure/`。

### 依赖

见 `requirements.txt`（含 **pandas、openpyxl、numpy、scipy、matplotlib、scikit-learn、tushare** 等）。

### 非 MVP（占位或扩展）

- `live/signal_system.generate_signals`
- `models.fusion.fuse_models`：仅 `mean_zscore` / `mean` 可用，其它 `method` 会报错

### 测试

```bash
python3 -m unittest tests.test_optimizer tests.test_backtest_multi tests.test_backtest_single tests.test_plotting tests.test_fusion tests.test_cache_io tests.test_benchmark tests.test_turnover tests.test_data_quality tests.test_risk_exposure tests.test_factors tests.test_factor_ml tests.test_factor_preprocess tests.test_factor_diagnostics tests.test_factor_validation tests.test_multi_universe_validation tests.test_parameter_sensitivity tests.test_factor_weight_stability tests.test_style_exposure tests.test_style_exposure_monitor tests.test_ic tests.test_factor_weighting tests.test_stock_pool tests.test_order_builder tests.test_order_precheck tests.test_paper_trading tests.test_broker tests.test_broker_reconcile tests.test_account_state tests.test_paper_runner tests.test_daily_paper_cli tests.test_paper_report tests.test_manual_confirmation tests.test_execution_feedback tests.test_paper_guard tests.test_paper_run_control tests.test_paper_scheduler -v
```
