# 工程总览（技术说明）

> 本文档描述 **quant_strategy** 仓库的**当前实现**、**数据与计算约定**、**MVP 主流程与明确非 MVP 的占位**，便于隔段时间回到项目时快速对齐。  
> 与 [INTERFACE_AND_CONTRACTS.md](./INTERFACE_AND_CONTRACTS.md)（接口契约）、[CODE_STRUCTURE.md](./CODE_STRUCTURE.md)（目录职责）、[FLOW_AND_MODULES.md](./FLOW_AND_MODULES.md)（主流程与各模块说明）互补；**代码变更后请同步更新本节对应段落**。

---

## 1. 项目定位与当前成熟度

**定位**：A 股日频、研究向的 **「数据 → 因子面板 → 数据质量 → IC 与稳定性诊断 → 因子诊断（Top-K 多头超额 + 分组收益单调性）→ 多因子权重建议与滚动权重 → 样本外验证与因子失效监控 → 单因子/融合回测 → 基准与超额收益 → 换手与成本 → 风险暴露与集中度 → 绩效与净值图 → 可选落盘」** 闭环；目录上预留 **实盘信号 / 模拟盘** 等扩展位。

**MVP 定稿**：上述闭环 **已实现并可作为交付边界**；`live.stock_pool` 与 `scripts/build_live_universe.py` 已可把人工股票池过滤成实盘目标池确认文件；`live.order_builder`、`live.order_precheck`、`live.paper_trading`、`live.broker`、`live.broker_reconcile`、`live.account_state`、`live.paper_runner`、`live.paper_report`、`live.manual_confirmation`、`live.execution_feedback`、`live.paper_guard`、`live.paper_run_control`、`live.paper_scheduler` 与 `scripts/run_daily_paper.py` / `scripts/run_scheduled_daily_paper.py` / `scripts/build_execution_feedback.py` 已补充为准实盘准备层，可把目标权重转换成订单计划、做基础可执行性检查、用虚拟账户或模拟券商验证成交协议、保存纸面账户状态、生成 Markdown 日报和小资金人工确认单，回填真实成交并分析执行偏差，在日终运行前后检查异常，保护交易日运行和重复写入，并提供可交给系统调度器的单次运行入口；日终纸面交易可通过 `--execution-mode simulated_broker` 走统一券商接口；`RealBrokerReadOnlyAdapter` 已提供真实券商只读接入骨架；`broker_reconcile` 可对比纸面账户与只读真实账户差异；**不包含** 真实券商交易 adapter、实时行情订阅、以及 `fuse_models` 中除 `mean_zscore` / `mean` 以外的方法。`main` 未调用 `run_multi_backtest(factors, weights)` 的线性加权路径，属产品取舍而非 MVP 缺口。**融合得分默认**由 **各因子日 IC 的滞后滚动均值** 做 z-score 列权（`fuse_ic_weighted_zscore`，可配置关闭回等权）；IC **不**写入股票层 `maximize_sharpe` / `risk_parity` 的 μ、Σ。

**已跑通**：

- 行情：`data/prices_demo.csv`、Tushare 日线本地缓存、Excel/CSV 股票池驱动的 Tushare 多标的日线，或合成宽表兜底。
- 财务：`fina_indicator` 拉取并与交易日对齐（PE、ROE、毛利率、净利率、资产负债率、营收增长、利润增长等）。
- 因子：`factors/panel_builder.build_four_factor_panel` 一次性产出基础多列（短/长动量、短反转、低波、成交量放大、PE、ROE、毛利率、净利率、低资产负债率、营收增长、利润增长）；若 `enable_ml_score=True`，`factors.factor_ml.build_ml_score_factor` 会用这些基础因子滚动训练梯度提升类模型，追加 `ML_SCORE` 作为机器学习打分因子；`main` 中单因子回测 **传入预计算 `factor_values`**（不重复算子）。`run_single_backtest` 仍支持仅传 `factor_name` 走注册表自动算子。
- 因子预处理：`factors.preprocess` 按交易日横截面对每列因子做 winsorize 与 z-score，生成 `factor_panel_zscore.csv`；多因子融合复用同一套标准化口径。
- 回测：**月末再平衡（`ME`）**、**Top-K 多头**、**收盘价成交**、**单边手续费**；若配置 `min_avg_volume` / `min_avg_amount`，Top-K 前会按过去窗口平均成交量 / 成交额做可交易性过滤。持仓在 Top-K 内为 **`portfolio_weighting`**：`equal`（1/K）、**`max_sharpe`**（历史日收益估 μ、Σ 后 `maximize_sharpe`，失败等权）或 **`risk_parity`**（同窗口估 Σ 后 `risk_parity`，样本不足等失败等权）。目标权重生成后会经过 `max_position_weight` 单票上限、`max_industry_weight` 行业权重上限、`target_volatility` 波动率目标、`min_positions` 最小持仓数量、`max_rebalance_turnover` 单次换手上限，以及可选的 `enable_trade_status_filter` 停牌 / 涨跌停交易约束。每期 `meta["rebalance_log"]` 记录选股、可交易性过滤前后候选数、行业暴露、目标波动缩放、最小持仓检查、现金目标仓位、约束后权重与节流信息；`meta["decision_log"]` 逐股票记录入选、过滤、所属行业、波动率缩放、最小持仓缩放、买卖、交易阻断和节流原因。
- **IC 与稳定性诊断**：`analysis.ic` 日截面 Spearman vs 前瞻收益；不参与调仓；可落盘 `output/cache/ic_*.csv`。同时输出 IC 分布分位数、正负占比、极端值和滚动稳定性到 `output/ic_diagnostics/`。
- **因子诊断**：`analysis.factor_diagnostics` 对每个因子构造 Top-K 等权多头腿，计算相对股票池等权基准的 `excess_ann_return`、`tracking_error`、`information_ratio`；同时按 `Settings.factor_group_count` 做分组收益，输出 Top-Bottom 与 `monotonicity_score`，用于回答“高分组有没有主动收益”和“全排序是否有收益层次”。`ML_SCORE` 与其他因子使用同一套诊断口径，不能因为来自机器学习模型就跳过验证。
- **多因子权重建议**：`models.factor_weighting` 综合 IC 分布、rolling IC、Top-Bottom 与单调性，输出 `factor_score` 和 `fusion_weight`；全样本表用于诊断审计，训练段表会作为 `FUSED_SCORE_WEIGHTED` 的静态权重来源，调仓日前历史窗口会生成 `rolling_factor_weight_log.csv`。
- **样本外验证与因子失效监控**：`analysis.factor_validation` 复用 IC、多头超额和分组收益口径，把历史日期按 `factor_weight_train_ratio` 切成训练段 / 验证段，比较 `ic_mean`、正 IC 占比、多头超额、Top-Bottom 和单调性，并输出 `OK/WATCH/DEGRADED/FAILED` 状态。
- **多因子融合**：默认 **`fuse_ic_weighted_zscore`**（各因子日 IC 经 `shift(1)+rolling` 得非负列权，再对横截面 z-score 加权；失败或配置关闭时用 **`fuse_equal_weight_zscore`**）→ `FUSED_ZSCORE`。另有 **`fuse_static_weight_zscore`**，将训练段 `fusion_weight` 固定后应用到验证段 → `FUSED_SCORE_WEIGHTED`。第三条为调仓日前滚动综合权重 → `FUSED_ROLLING_SCORE_WEIGHTED`，每期只用历史窗口、带权重上下限和平滑。三者都通过 `run_multi_backtest(fused=...)` 进入同一套 Top-K 回测；另支持 `run_multi_backtest(factors, weights)` 原始因子线性加权入口（`main` 当前未用）。
- **优化**：`models.optimizer.maximize_sharpe` / `risk_parity`；回测在 `portfolio_weighting` 为 `max_sharpe` 或 `risk_parity` 时于再平衡日调用对应函数。流程见 [FLOW_AND_MODULES.md](./FLOW_AND_MODULES.md)。
- **数据质量**：`analysis.data_quality` 在因子面板构建后统计价格覆盖率、因子覆盖率、每日覆盖率与调仓日有效截面规模。
- 绩效：`analysis.performance.summarize`（年化收益、波动、夏普、最大回撤等）。
- **基准与超额收益**：`analysis.benchmark.equal_weight_benchmark_nav` 用当前股票池生成每日等权基准；`summarize_excess` 为每条策略补充 `excess_ann_return`、`tracking_error`、`information_ratio`。
- **换手与成本**：`analysis.turnover` 由 `meta["rebalance_log"]` 计算逐期目标权重变化，补充 `avg_turnover`、`total_turnover`、`estimated_total_cost` 等指标。
- **风险暴露与集中度**：`analysis.risk_exposure` 由同一份 `rebalance_log` 计算 HHI、`effective_n`、Top 权重和持仓数，补充组合是否过度集中的风控视角。
- **订单生成、预检查、统一券商接口、纸面交易、账户状态、每日运行器、日终脚本、日报、人工确认单、真实成交回填、异常检查、运行控制与调度入口**：`live.order_builder.build_order_plan` 将目标权重、当前持仓、最新价格和现金 / 总资产转换为订单计划，字段包括 `BUY/SELL`、目标股数、调整股数、预估金额、当前/目标权重与交易原因；`build_order_plan_from_rebalance_meta` 可从最近一期 `meta["rebalance_log"]` 直接提取目标权重。`live.order_precheck.precheck_order_plan` 进一步检查现金、可卖数量、买入手数、最小金额、停牌 / 涨停买入 / 跌停卖出约束，并输出 `PASS/BLOCK` 与原因。`live.broker` 定义统一交易适配器协议：查资金、查持仓、查订单、下单、撤单；`SimulatedBroker` 可用同一协议立即成交订单计划；`RealBrokerReadOnlyAdapter` 是真实券商只读骨架，允许查询账户、持仓、订单，禁止下单和撤单。`live.paper_trading.run_paper_trading` 只执行 `PASS` 订单，按手续费更新虚拟现金和持仓，并记录 `FILLED/SKIPPED`、现金变化与持仓变化。`live.account_state` 保存 / 读取虚拟现金、持仓和每日账户快照，使纸面账户可以连续运行。`live.paper_runner.run_daily_paper_trade` 将这些步骤串成单日纸面交易入口，默认走旧纸面成交，也可通过 `execution_mode="simulated_broker"` 走统一模拟券商，并把 `broker_orders` 转成兼容的 `paper_trades`。`live.paper_report` 将运行结果写成 Markdown 日报。`live.manual_confirmation` 基于订单计划、预检查和可选因子失效监控生成小资金人工确认单，预留真实执行回填字段；`live.execution_feedback` 再读取这些回填字段，对比建议订单与实际成交，输出成交状态、数量差异、滑点和金额差异。`live.paper_guard` 检查目标权重、价格日期、价格有效性、现金、持仓、订单检查和成交日志，ERROR 阻断，WARNING 写入摘要与日报。`live.paper_run_control` 从价格缓存提取交易日日历，默认阻断非交易日运行和同日重复覆盖快照。`live.paper_scheduler` 记录一次调度运行的参数、stdout、stderr 与退出码。`scripts/run_daily_paper.py` 从已有回测输出读取最近目标权重和最新价格并打印摘要；`scripts/run_scheduled_daily_paper.py` 适合交给 cron / launchd / 服务器调度器调用；`scripts/build_execution_feedback.py` 用于人工成交回填后的执行偏差复盘。该层已有统一券商接口协议、模拟实现和只读骨架，但不连接真实交易 API。

**作图**：

- `analysis/plotting.plot_nav`：支持 `Series` 或多列 `DataFrame`；`normalize=True` 时按各列首行有效值归一便于对比；`save_path` 存在时使用 `Agg` 后端写 PNG。
- `analysis/plotting.plot_ic`：日 IC 序列（或多因子对比）；单列时可选滚动均线；`persist_run_outputs` 时 `main` 写 `output/ic_compare.png`、`ic_timeseries_<因子>.png`。
- `analysis/plotting.plot_weights` + `rebalance_log_to_weights_frame`：由 `meta["rebalance_log"]` 得到宽表后堆叠面积图（或热力图）；`persist_run_outputs` 时 `main` 写 `output/weights_<因子>.png`。
- `main.py` 收集各因子、`FUSED_ZSCORE`、`FUSED_SCORE_WEIGHTED`、`FUSED_ROLLING_SCORE_WEIGHTED` 与 `BENCH_EQUAL_WEIGHT` 净值后保存 **`output/nav_compare.png`**；另保存 **`output/excess_nav_compare.png`** 比较各策略相对基准的超额净值。
- `analysis/plotting.plot_turnover`：绘制各策略逐期换手率；`persist_run_outputs` 时 `main` 写 **`output/turnover_compare.png`**。
- `analysis/plotting.plot_effective_n`：绘制各策略逐期有效持仓数；`persist_run_outputs` 时 `main` 写 **`output/risk_exposure/effective_n_compare.png`**。
- `analysis/plotting.plot_factor_coverage`：绘制因子有效覆盖率；`persist_run_outputs` 时 `main` 写 **`output/data_quality/factor_coverage.png`**。

**数据落盘（缓存）**：

- `live/cache_io.save_run_cache`：在因子面板构建成功后，将 **`prices_long.csv`**、**`prices_wide_close.csv`**、**`factor_panel.csv`**、**`factor_panel_zscore.csv`**、**`run_meta.txt`** 写入 **`output/cache/`**；若生成 `ML_SCORE`，`main` 另写 **`output/factor_diagnostics/ml_score_training_log.csv`**。
- `live/cache_io.save_data_quality_reports`：写 **`output/data_quality/*.csv`**，保存价格 / 因子 / 调仓日覆盖率报告。
- `analysis.ic.save_ic_series`：在 IC 计算完成且 `persist_run_outputs` 时写 **`ic_<因子名>.csv`**。
- `analysis.ic.save_ic_diagnostics`：写 **`output/ic_diagnostics/ic_distribution_summary.csv`** 与 **`ic_rolling_stability.csv`**，保存 IC 分布和滚动稳定性。
- `live/cache_io.save_factor_diagnostics`：写 **`output/factor_diagnostics/long_excess_summary.csv`**、**`group_return_detail.csv`**、**`group_return_summary.csv`**、**`factor_weight_summary.csv`**、**`factor_weight_train_summary.csv`**、**`rolling_factor_weight_log.csv`**，保存每个因子的 Top-K 多头超额、分组收益、单调性、全样本权重诊断、训练段静态融合权重与调仓日前滚动权重。
- `analysis.factor_validation.save_factor_validation_outputs`：写 **`output/factor_validation/out_of_sample_validation.csv`** 与 **`factor_decay_monitor.csv`**，保存训练段 / 验证段因子评价对照和失效监控状态。
- `live/cache_io.save_run_config`：写 **`output/cache/run_config.json`**，保存本次 `Settings` 配置快照。
- `live/cache_io.save_performance_summary`：写 **`output/performance_summary.csv`**，汇总每条策略的年化收益、波动、夏普、最大回撤，并包含相对基准、换手率、预估成本与集中度指标。
- `live/cache_io.save_rebalance_logs`：写 **`output/rebalance_logs/<策略名>.csv`**，记录每次调仓的日期、标的、权重、配权方式、排序、流动性过滤前后候选数、行业上限是否触发、最大行业暴露、目标波动、缩放比例、最小持仓检查、现金目标仓位与阈值。
- `live/cache_io.save_decision_logs`：写 **`output/decision_logs/<策略名>.csv`**，记录每个调仓日每只候选/上期持仓股票的因子分数、排序、流动性过滤、入选状态、行业、波动率缩放标记、上期权重、原始目标权重、最终目标权重、动作与原因。
- `live/cache_io.save_turnover_logs`：写 **`output/turnover_logs/<策略名>.csv`**，记录每期换手率、预估成本、持仓数与配权方式。
- `live/cache_io.save_order_plans`：写 **`output/order_plans/<策略名>.csv`**，保存由目标权重、当前持仓和最新价格生成的订单计划。
- `live/cache_io.save_order_checks`：写 **`output/order_checks/<策略名>.csv`**，保存订单预检查结果和阻断原因。
- `live/cache_io.save_paper_trades`：写 **`output/paper_trades/<策略名>.csv`**，保存纸面交易成交 / 跳过日志。
- `live/cache_io.save_risk_exposure_logs` / `save_risk_exposure_summary`：写 **`output/risk_exposure/concentration_logs/<策略名>.csv`** 与 **`output/risk_exposure/concentration_summary.csv`**，记录逐期与汇总集中度。
- `config.Settings.persist_run_outputs`（默认 `True`）为关时跳过上述写入。

**明确非 MVP（占位 / 后续）**：

- `live/signal_system.py`：`NotImplementedError`（实盘信号生成占位）；定时任务编排、券商接口仍未实现。

---

## 2. 目录与模块职责（与代码一致）

| 路径 | 职责 |
|------|------|
| `config.py` | `Settings`：`data_dir`、`output_dir`、`stock_pool_path`、`stock_pool_code_col`、`tushare_price_cache_path`、`backtest_start`/`end`、`rebalance_freq`（默认 `ME`）、`top_k`、`commission_rate`、`portfolio_weighting`（`equal`/`max_sharpe`/`risk_parity`）、`max_position_weight`、`max_industry_weight`、`industry_col`、`target_volatility`、`volatility_target_lookback_days`、`volatility_target_min_obs`、`min_positions`、`min_positions_exposure`、`order_lot_size`、`min_order_amount`、`order_cash_buffer`、`paper_initial_cash`、`max_rebalance_turnover`、`liquidity_lookback_days`、`min_avg_volume`、`min_avg_amount`、`enable_trade_status_filter`、`optimizer_return_window`、`optimizer_min_obs`、`ic_forward_days`、`ic_rolling_windows`、`factor_group_count`、`fusion_use_ic_weights`、`fusion_ic_rolling_window`、`fusion_ic_min_periods`、`factor_weight_train_ratio`、`rolling_factor_weight_*`、`persist_run_outputs`、动量/波动/财务窗口等；`get_tushare_token()`（环境变量优先，本地回退**勿提交密钥**）。 |
| `main.py` | 入口：加载本地 demo / Tushare 缓存 / 股票池 Tushare 数据 → `build_four_factor_panel` → 数据质量报告 → 可选 `save_run_cache` → IC 与稳定性诊断 → 因子诊断（Top-K 多头超额 + 分组收益单调性）→ 多因子权重建议、训练段权重与滚动权重日志 → 可选 `save_ic_series` 与 **IC/权重 PNG** → 多列因子各 `run_single_backtest(..., factor_values=列, long_prices=long_df)` → 可交易性过滤与决策审计 → **`_build_fused_zscore_panel`（IC 列权或等权）** + **`fuse_static_weight_zscore`（训练段静态权重）** + **调仓日前滚动综合权重** → `run_multi_backtest(fused=..., long_prices=long_df)` → 股票池等权基准与超额指标 → 换手与成本 → 风险暴露与集中度 → 绩效汇总 / 调仓日志 / 决策审计日志 / 换手日志 / 集中度日志 / 配置快照落盘 → `plot_nav`。 |
| `data/` | 原始/演示数据；存在 `prices_demo.csv` 时优先读本地；真实股票池可放 `stock_pool.xlsx`，Tushare 日线缓存默认 `prices_tushare_cache.csv`。 |
| `live/data_feed.py` | `fetch_daily_panel`、`fetch_fina_indicator_panel`、`load_prices_from_csv` 等。 |
| `live/stock_pool.py` | `load_stock_pool_frame`、`build_stock_pool_filter_report`、`active_universe_from_report`、`save_universe_files`：读取人工股票池，保留主题/启用状态，按价格覆盖、流动性、停牌 / 涨跌停生成过滤报告与 active universe。 |
| `live/cache_io.py` | `save_run_cache` → `output/cache/` 行情、原始因子面板与标准化因子面板；`save_data_quality_reports`、`save_factor_diagnostics`、`save_run_config`、`save_performance_summary`、`save_rebalance_logs`、`save_decision_logs`、`save_turnover_logs`、`save_order_plans`、`save_order_checks`、`save_paper_trades`、`save_risk_exposure_logs`、`save_risk_exposure_summary` → 实验运行记录。 |
| `live/account_state.py` | `save_account_state`、`load_account_state`、`positions_from_trades`：保存 / 读取纸面账户现金、持仓和每日快照。 |
| `live/order_builder.py` | `build_order_plan`、`build_order_plan_from_rebalance_meta`：把目标权重转换成买卖股数和预估金额，是订单预检查、纸面交易和券商接口之前的准实盘准备层。 |
| `live/order_precheck.py` | `precheck_order_plan`：检查订单计划的现金、可用股数、买入手数、最小金额、停牌和涨跌停约束，是纸面交易和真实下单前的基础风控闸门。 |
| `live/broker.py` | `BrokerAdapter`、`BrokerAccount`、`BrokerPosition`、`BrokerOrder`、`SimulatedBroker`、`RealBrokerConfig`、`RealBrokerReadOnlyAdapter`：统一券商接口协议、模拟券商适配器和真实券商只读骨架。 |
| `live/broker_reconcile.py` | `reconcile_paper_with_broker`、`save_reconciliation_outputs`：比较纸面账户和只读券商账户，输出账户 / 持仓差异与 Markdown 对账报告。 |
| `live/paper_trading.py` | `run_paper_trading`、`paper_account_snapshot`：用虚拟账户执行通过预检查的订单，更新现金、持仓并生成账户快照，是券商接口前的纸面验证层。 |
| `live/paper_runner.py` | `run_daily_paper_trade`：读取上一日纸面账户状态，串联订单计划、预检查、执行模式选择、纸面成交或模拟券商成交、持仓更新、账户快照与 CSV 落盘，是自动调度前的一日运行入口。 |
| `live/paper_report.py` | `build_daily_paper_report`、`save_daily_paper_report`：生成并保存纸面交易 Markdown 日报，展示执行模式和券商订单回报。 |
| `live/manual_confirmation.py` | `build_manual_confirmation_sheet`、`save_manual_confirmation`：生成小资金人工确认实盘单，包含预检查状态、因子健康状态和真实执行回填字段。 |
| `live/execution_feedback.py` | `build_execution_feedback`、`save_execution_feedback`：读取人工确认单的真实成交回填字段，生成逐笔执行偏差、汇总表和 Markdown 报告。 |
| `live/paper_guard.py` | `validate_daily_inputs`、`validate_daily_result`、`raise_on_guard_errors`：检查日终纸面交易输入与结果异常，区分 ERROR 和 WARNING。 |
| `live/paper_run_control.py` | `load_trading_calendar_from_prices`、`validate_daily_run_control`：从价格缓存提取交易日日历，阻断非交易日运行和重复覆盖快照。 |
| `live/paper_scheduler.py` | `run_scheduled_daily_paper`：运行一次日终纸面交易，写调度日志并返回退出码。 |
| `live/daily_paper_cli.py` | `run_daily_paper_from_outputs`、`format_daily_paper_summary`：读取已有调仓日志与价格缓存，运行交易日控制、异常检查、日终纸面交易、日报和人工确认单生成，并生成命令行摘要；支持 `execution_mode`。 |
| `scripts/run_daily_paper.py` | 日终纸面交易命令行入口：调用 `live.daily_paper_cli.main`，支持策略名、交易日、交易状态文件、执行模式、只读检查模式、关闭日报、关闭人工确认单、关闭 guard、允许非交易日和允许重复运行。 |
| `scripts/build_live_universe.py` | 实盘目标池确认入口：从股票池、价格缓存和可选交易状态生成 `stock_pool_filter_report_<date>.csv` 与 `active_universe_<date>.csv`。 |
| `scripts/run_scheduled_daily_paper.py` | 调度命令行入口：调用 `live.paper_scheduler.run_scheduled_daily_paper`，透传日终纸面交易参数并写 `output/scheduler_logs/<date>.log`。 |
| `scripts/build_execution_feedback.py` | 真实成交回填入口：读取人工确认单 CSV，生成 `output/execution_feedback/<strategy>/` 下的逐笔偏差、汇总和 Markdown 报告。 |
| `backtest/backtest_utils.py` | `to_returns`、`long_to_wide`、`wide_to_long`、`prices_to_wide_close`、`align_panel`。 |
| `backtest/backtest_single.py` | `run_single_backtest`：再平衡日可交易性过滤、Top-K、**等权 / 夏普 / 风险平价**、单票权重上限、行业权重上限、波动率目标与现金仓位、最小持仓数量、单次换手上限、停牌 / 涨跌停交易约束、撮合与净值；`meta` 含 `rebalance_log`、`decision_log`、`portfolio_weighting`、`max_position_weight`、`max_industry_weight`、`target_volatility`、`min_positions`、`max_rebalance_turnover`。 |
| `backtest/backtest_multi.py` | `run_multi_backtest`：`fused=` 或 `factors`+`weights` 合成一列后转调 `run_single_backtest`。 |
| `analysis/ic.py` | `daily_ic_spearman`、`summarize_ic`、`ic_distribution_summary`、`ic_rolling_stability`、`save_ic_series`、`save_ic_diagnostics`。 |
| `factors/` | 各 `calc_*` + `FACTOR_REGISTRY` + `panel_builder`；`factor_ml.py` 生成二阶机器学习打分因子 `ML_SCORE`。 |
| `analysis/data_quality.py` | `price_coverage`、`factor_coverage`、`factor_daily_coverage`、`rebalance_coverage`。 |
| `analysis/performance.py` | `summarize(nav)`。 |
| `analysis/benchmark.py` | `equal_weight_benchmark_nav`、`summarize_excess`、`excess_nav_frame`。 |
| `analysis/factor_diagnostics.py` | `factor_long_only_nav`、`factor_long_excess_summary`、`batch_factor_long_excess`、`factor_group_return_detail`、`summarize_group_returns`、`batch_factor_group_returns`：因子 Top-K 多头腿、分组收益与单调性诊断。 |
| `analysis/factor_validation.py` | `build_out_of_sample_validation`、`build_factor_decay_monitor`、`save_factor_validation_outputs`：训练段 / 验证段因子评价对照、失效状态判断和 CSV 落盘。 |
| `analysis/turnover.py` | `turnover_frame`、`summarize_turnover`、`turnover_wide`。 |
| `analysis/risk_exposure.py` | `concentration_frame`、`summarize_concentration`、`effective_n_wide`。 |
| `analysis/plotting.py` | `plot_nav`、`plot_ic`、`plot_weights`、`plot_turnover`、`plot_effective_n`、`plot_factor_coverage`、`rebalance_log_to_weights_frame`。 |
| `models/fusion.py` | `fuse_equal_weight_zscore`、`fuse_ic_weighted_zscore`、`fuse_static_weight_zscore`、`fuse_models`（仅部分 `method`）。 |
| `models/factor_weighting.py` | `build_factor_weight_summary`：将因子评价指标合成 `factor_score` / `fusion_weight`；全样本用于诊断，训练段用于静态融合验证。 |
| `factors/factor_ml.py` | `build_ml_score_factor`、`forward_return_label`：用基础因子特征滚动训练机器学习模型，输出 `ML_SCORE` 和训练日志。 |
| `factors/preprocess.py` | `winsorize_series`、`cross_sectional_zscore`、`preprocess_factor_panel`；统一因子清洗和标准化口径。 |
| `models/optimizer.py` | `maximize_sharpe`、`risk_parity`；由 `backtest_single` 在对应 `portfolio_weighting` 时再平衡日调用。 |

---

## 3. 数据形态（技术细节）

### 3.1 长表（long）

多行，每行一条记录，至少包含：

- `trade_date`（`datetime64`）
- `ts_code`（如 `600519.SH`）
- `open, high, low, close, volume`（CSV 可为 `vol`，加载时统一为 `volume`）

**用途**：Tushare 合并结果、CSV 读入、PE/ROE/质量/成长财务因子与财报 `merge_asof` 的左表。

### 3.2 宽表（wide）

- **行**：交易日（`DatetimeIndex`）
- **列**：`ts_code`
- **值**：通常为 `close`

**用途**：`calc_momentum`、回测主循环按日取 `prices_wide.loc[dt]` 向量；`to_returns` 对整表 `pct_change()`。

### 3.3 因子 PanelLong

`pandas.Series`，`MultiIndex` 两级名：`date`、`symbol`；与 `prices_wide` 的日期、列名对齐后，调仓日用 `factor_values.xs(dt, level=0)` 取**当日横截面**。

---

## 4. 从 `main.py` 到绩效的逐步执行

下列顺序与当前 `main()` **一致**（因子名循环顺序见 §7 `DEFAULT_FACTOR_ORDER`）。

| 步骤 | 行为 | 关键代码 |
|------|------|-----------|
| 1 | 读配置并打印 `portfolio_weighting` | `get_settings()` |
| 2a | 若存在 `data/prices_demo.csv` | `load_prices_from_csv` → `long_to_wide` → `prices` |
| 2b | 否则若存在 Tushare 行情缓存 | `load_prices_from_csv(settings.tushare_price_cache_path)` → `long_to_wide` |
| 2c | 否则读取股票池并拉取 Tushare | `load_stock_pool(settings.stock_pool_path)` → `fetch_daily_panel(...)` → 写行情缓存 → `long_to_wide` |
| 2d | 股票池不存在则使用默认示例股票；失败则合成宽表 | `_DEFAULT_TS_SYMBOLS` / `_demo_price_wide()`，`wide_to_long` → `long_df` |
| 3 | 构建基础多因子面板 | `build_four_factor_panel(prices, long_df, settings)` → `panel` |
| 4 | 追加机器学习打分因子 | `build_ml_score_factor(panel, prices, settings)` → `ML_SCORE`（可配置关闭） |
| 5 | 数据质量报告 | `price_coverage`、`factor_coverage`、`rebalance_coverage` |
| 6 | 因子预处理与可选落盘 | `preprocess_factor_panel`、`save_run_cache`（`persist_run_outputs`） |
| 7 | IC：各因子列 + **与融合同构的** FUSED 得分 | `daily_ic_spearman`、`summarize_ic`；可选 `save_ic_series` |
| 8 | IC 分布与稳定性诊断 | `ic_distribution_summary`、`ic_rolling_stability` → `ic_distribution_summary.csv`、`ic_rolling_stability.csv` |
| 9 | 因子诊断 | `batch_factor_long_excess`、`batch_factor_group_returns` → `long_excess_summary.csv`、`group_return_detail.csv`、`group_return_summary.csv` |
| 10 | 多因子权重建议 | `build_factor_weight_summary` → `factor_weight_summary.csv` / `factor_weight_train_summary.csv` / `rolling_factor_weight_log.csv` |
| 11 | 样本外验证与因子失效监控 | `build_out_of_sample_validation`、`build_factor_decay_monitor` → `output/factor_validation/*.csv` |
| 12 | 单因子回测 ×N | `run_single_backtest(fname, factor_values=panel[fname], ...)` |
| 13 | 融合回测 ×3 | **`_build_fused_zscore_panel`**（`fuse_ic_weighted_zscore` 或等权）→ `FUSED_ZSCORE`；**`fuse_static_weight_zscore`**（训练段权重、验证段回测）→ `FUSED_SCORE_WEIGHTED`；调仓日前滚动综合权重 → `FUSED_ROLLING_SCORE_WEIGHTED` |
| 14 | 绩效与打印 | `summarize`；`_print_backtest_block` 打印 `rebalance_log`、绩效 |
| 15 | 基准与超额收益 | `equal_weight_benchmark_nav`、`summarize_excess` |
| 16 | 换手与成本 | `turnover_frame`、`summarize_turnover` |
| 17 | 风险暴露与集中度 | `concentration_frame`、`summarize_concentration` |
| 18 | 实验记录落盘 | `run_config.json`、`performance_summary.csv`、`ic_diagnostics/*.csv`、`factor_diagnostics/*.csv`、`factor_validation/*.csv`、`data_quality/*.csv`、`rebalance_logs/*.csv`、`turnover_logs/*.csv`、`risk_exposure/*.csv` |
| 19 | 净值、超额净值、覆盖率、换手与集中度图 | `plot_nav` / `plot_factor_coverage` / `plot_turnover` / `plot_effective_n` |

**多因子关系**：多条单因子回测为 **同一 `panel` 的不同列** 的独立策略；融合回测为 **IC 列权、训练段静态权重或调仓日前滚动权重** 得到的综合得分经 `run_multi_backtest` 的独立策略。调仓日 **不会**临时拼接字段再跑一条，融合得分会在进入回测前预先生成。

---

## 5. `run_single_backtest` 内部逻辑（技术细节）

### 5.1 输入

- `prices`：宽表收盘价，或含 `trade_date, ts_code, close` 的长表（经 `prices_to_wide_close` 统一）。
- `factor_name`：`FACTOR_REGISTRY` 键。
- 可选 `factor_values`：若传入则跳过自动算子。
- 可选 `long_prices`：PE/ROE/质量/成长财务因子用；缺省则由 `wide_to_long(prices_wide)` 生成。
- 可选 `finance_df`：PE/ROE/质量/成长财务因子用；缺省则 `fetch_fina_indicator_panel(...)`。
- `kwargs`：`vol_window`、`token` 等。

### 5.2 因子自动分支

| `factor_name` | 数据依赖 | 计算要点 |
|----------------|----------|----------|
| `MOMENTUM` | `prices_wide` | \(P_t/P_{t-L}-1\)，`L = momentum_lookback`（默认 20），`stack` → PanelLong。 |
| `MOMENTUM_60D` | `prices_wide` | 同动量口径，默认 `momentum_long_lookback=60`，用于补充更长周期趋势。 |
| `REVERSAL_5D` | `prices_wide` | 过去 `reversal_lookback` 日收益取负，越大表示短期跌得越多。 |
| `VOLATILITY` | `to_returns(prices_wide)` | 滚动标准差 × √252 年化，再取 **负号**（因子越大 = 历史波动越低）。 |
| `VOLUME_RATIO_20D` | `long_df.volume` | 成交量 / 过去 `volume_ratio_window` 日均量 - 1；无 volume 时该列为空并由主流程跳过。 |
| `PE` | `fina` + `long_px` | `merge_asof(..., direction="backward")` 按 `ann_date` 对齐到 `trade_date`；`PE=close/eps`（eps>0），输出 **-PE**（因子越大 = 市盈率越低）。 |
| `ROE` | 同上 | 对齐后取 **`roe`** 列（越大越好）。 |
| `GROSS_MARGIN` | 同上 | 对齐后取毛利率候选列，越高越好。 |
| `NET_MARGIN` | 同上 | 对齐后取净利率候选列，越高越好。 |
| `LOW_DEBT_TO_ASSETS` | 同上 | 对齐后取资产负债率候选列并取负号，越高表示负债率越低。 |
| `REVENUE_GROWTH` | 同上 | 对齐后取营收同比增长候选列，越高越好。 |
| `PROFIT_GROWTH` | 同上 | 对齐后取利润同比增长候选列，越高越好。 |

财务拉取区间：`backtest_start` 前再回溯 `fina_history_years`（默认 2 年），避免样本初期无可用财报。

### 5.3 再平衡日历

- `settings.rebalance_freq` 默认 **`ME`**（月末，兼容 Pandas 3；`M` 在 `backtest_single._resample_freq_alias` 中映射为 `ME`）。
- `rebalance_dates = prices_wide.resample(rf).last().index ∩ prices_wide.index`。

### 5.4 调仓日算法（横截面）

1. `sc = factor_values.xs(dt, level=0).dropna()`。
2. `sc.sort_values(ascending=False)`：**因子数值越大越优先**（各因子已通过定义统一「大=好」）。
3. 按顺序形成有效候选池；若开启流动性阈值，先过滤候选池，再取前 `k` 只得到 `picks`。
4. **配权**：若 `portfolio_weighting == "max_sharpe"` 且样本足够，用过去 `optimizer_return_window` 日日收益估 μ、Σ，调用 `maximize_sharpe` 得权重；若为 **`risk_parity`** 且样本足够，估 Σ 后调用 **`risk_parity`**；否则等权（1/K）。目标权重之后统一经过 `max_position_weight` 单票上限处理、`max_industry_weight` 行业上限处理、`target_volatility` 波动率目标处理、`min_positions` 最小持仓数量处理，再经过 `max_rebalance_turnover` 单次换手上限处理；失败标签见 `rebalance_log[].weighting`（`max_sharpe_fallback` / `risk_parity_fallback` / `equal`），触发单票上限时追加 `_capped`，触发换手节流时追加 `_turnover_capped`。
5. **交易状态约束**：若 `enable_trade_status_filter=True`，读取 `is_suspended` / `is_limit_up` / `is_limit_down`。停牌不能买卖，涨停不能买入 / 加仓，跌停不能卖出 / 减仓；被阻断原因写入 `decision_log.trade_block_reason`。
6. **撮合**：目标市值按权重分配；**先卖后买**；手续费 `commission_rate` × 成交额；现金不足时对买入批量 **缩放**。
7. 将本日 `picks`、权重、`weighting`、行业暴露、目标波动缩放、最小持仓检查与现金仓位记入 **`meta["rebalance_log"]`**；将候选/上期持仓的排序、过滤、行业、入选、权重变化、动作和原因记入 **`meta["decision_log"]`**。

### 5.5 净值

- 初始现金 `1.0`，持仓股数 `Series(0, index=symbols)`。
- 每日：`nav = cash + Σ(shares_i × price_i)`（无效价跳过该项）。
- 输出：`nav` 为 `Series`（`date` 索引），`meta` 含 `n_rebalances`、`portfolio_weighting`、`max_position_weight`、`max_rebalance_turnover`、`rebalance_log`、`decision_log` 等。

---

## 6. 绩效指标（`analysis/performance.summarize`）

输入为**日频净值序列** `nav`（已 `dropna`）。

| 字段 | 计算方式（简述） |
|------|------------------|
| `ann_return` | \((NAV_T/NAV_0)^{252/(n-1)} - 1\)，\(n\) 为样本交易日数。 |
| `ann_vol` | 日收益率 `pct_change` 的样本标准差 × √252。 |
| `sharpe` | `(ann_return - risk_free) / ann_vol`，默认 `risk_free=0`。 |
| `max_drawdown` | 对日收益累积 `(1+r).cumprod()` 相对历史峰值的 **最小** \((cum/peak - 1)\)。 |

**注意**：此为研究常用简化口径；与券商报表或考虑无风险曲线日频的夏普可能不一致。

---

## 7. 默认股票池与因子列表（`main.py`）

- **股票池**：优先 `Settings.stock_pool_path`（默认 `data/stock_pool.xlsx`，可用 `QUANT_STOCK_POOL_PATH` 指向本机文件）；缺失时才使用 `_DEFAULT_TS_SYMBOLS` 示例股票池。
- **因子循环顺序**：基础顺序来自 `factors.panel_builder.DEFAULT_FACTOR_ORDER`：`MOMENTUM` → `MOMENTUM_60D` → `REVERSAL_5D` → `VOLATILITY` → `VOLUME_RATIO_20D` → `PE` → `ROE` → `GROSS_MARGIN` → `NET_MARGIN` → `LOW_DEBT_TO_ASSETS` → `REVENUE_GROWTH` → `PROFIT_GROWTH`；若 `enable_ml_score=True` 且模型生成有效预测，`main` 会在末尾追加 `ML_SCORE`，后续 IC/诊断/回测均使用动态 `factor_order`。

---

## 8. 依赖与运行环境

- **Python 包**：见根目录 `requirements.txt`（含 `pandas`、`numpy`、`scipy`、`matplotlib`、`scikit-learn`、`tushare` 等）。
- **解释器**：建议使用项目内 `.venv`；`.vscode/settings.json` 已指向 `${workspaceFolder}/.venv/bin/python`。
- **Tushare**：日线与 `fina_indicator` 均需 **积分/权限**；失败时 `main` 会回退合成数据（财务因子在无财务时为空列，后续诊断/回测按有效覆盖处理）。

---

## 9. 后续可补充方向（与代码占位对应）

| 方向 | 说明 |
|------|------|
| 作图 | `plot_nav` / `plot_ic` / `plot_weights`；`persist_run_outputs` 时写 IC 与权重 PNG（见 `main`）。 |
| 实盘链路 | `scripts/build_live_universe.py` 已提供实盘目标池确认，`scripts/run_daily_paper.py` 已提供日终纸面命令，`live.paper_report` 已提供日报，`live.manual_confirmation` 已提供人工确认单，`live.execution_feedback` 已提供真实成交回填与执行偏差分析，`live.paper_guard` 已提供运行异常检查，`live.paper_run_control` 已提供交易日日历与重复运行保护，`scripts/run_scheduled_daily_paper.py` 已提供系统调度入口，`live.broker` 已提供统一券商接口、模拟券商和真实券商只读骨架，`live.broker_reconcile` 已提供纸面 / 真实账户只读对账，日终流程已可通过 `--execution-mode simulated_broker` 使用该接口；后续补具体券商 API 同步与真实交易 adapter。 |
| 融合扩展 | `fuse_models` 更多 `method`（如 `dynamic`、`xgboost`）。 |
| 数据 | 启动时读 `output/cache` 命中则跳过 Tushare；或规范落盘至 `data/`。 |
| 检验 | 扩展 `unittest`/`pytest`（含 `tests/test_optimizer.py`、`test_backtest_*`、`test_plotting.py`、`test_fusion.py`）。 |
| 更真实成交 | 次日开盘成交、盘口滑点、部分成交、成交回报与券商对账。 |
| 绩效 | 基准超额收益等。 |

---

## 10. 文档索引

| 文档 | 内容 |
|------|------|
| [MVP_PROJECT_ARTICLE.md](./MVP_PROJECT_ARTICLE.md) | 项目正文：MVP 定位、流程图、主流程表、数据/因子/IC/融合/回测与扩展方向。 |
| [xiaohongshu/README.md](./xiaohongshu/README.md) | 小红书用分篇稿：按模块拆分、配图与标签建议。 |
| [INTERFACE_AND_CONTRACTS.md](./INTERFACE_AND_CONTRACTS.md) | 字段契约、Token 与路径约定、再平衡频率说明。 |
| [CODE_STRUCTURE.md](./CODE_STRUCTURE.md) | 模块分工与推荐阅读顺序。 |
| [FLOW_AND_MODULES.md](./FLOW_AND_MODULES.md) | 主流程 Mermaid 与各模块职责表。 |
| **本文 ENGINEERING_OVERVIEW.md** | 端到端技术总览与公式级行为说明。 |

---

**文档维护**：与代码**非自动同步**；合并功能后请同时更新本文、`FLOW_AND_MODULES.md`、`README.md` 及 `INTERFACE` 中相关契约。建议在 PR 说明中写「已更新文档：…」。
