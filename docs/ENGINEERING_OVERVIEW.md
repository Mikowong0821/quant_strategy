# 工程总览（技术说明）

> 本文档描述 **quant_strategy** 仓库的**当前实现**、**数据与计算约定**、**MVP 主流程与明确非 MVP 的占位**，便于隔段时间回到项目时快速对齐。  
> 与 [INTERFACE_AND_CONTRACTS.md](./INTERFACE_AND_CONTRACTS.md)（接口契约）、[CODE_STRUCTURE.md](./CODE_STRUCTURE.md)（目录职责）、[FLOW_AND_MODULES.md](./FLOW_AND_MODULES.md)（主流程与各模块说明）互补；**代码变更后请同步更新本节对应段落**。

---

## 1. 项目定位与当前成熟度

**定位**：A 股日频、研究向的 **「数据 → 因子面板 → 数据质量 → IC → 单因子/融合回测 → 基准与超额收益 → 换手与成本 → 绩效与净值图 → 可选落盘」** 闭环；目录上预留 **实盘信号 / 模拟盘** 等扩展位。

**MVP 定稿**：上述闭环 **已实现并可作为交付边界**；**不包含** 实盘信号生成、模拟盘撮合、券商接口、以及 `fuse_models` 中除 `mean_zscore` / `mean` 以外的方法。`main` 未调用 `run_multi_backtest(factors, weights)` 的线性加权路径，属产品取舍而非 MVP 缺口。**融合得分默认**由 **各因子日 IC 的滞后滚动均值** 做 z-score 列权（`fuse_ic_weighted_zscore`，可配置关闭回等权）；IC **不**写入股票层 `maximize_sharpe` / `risk_parity` 的 μ、Σ。

**已跑通**：

- 行情：Tushare 多标的日线、或 `data/prices_demo.csv`、或合成宽表兜底。
- 财务：`fina_indicator` 拉取并与交易日对齐（PE/ROE）。
- 因子：`factors/panel_builder.build_four_factor_panel` 一次性产出四列；`main` 中单因子回测 **传入预计算 `factor_values`**（不重复算子）。`run_single_backtest` 仍支持仅传 `factor_name` 走注册表自动算子。
- 回测：**月末再平衡（`ME`）**、**Top-K 多头**、**收盘价成交**、**单边手续费**；持仓在 Top-K 内为 **`portfolio_weighting`**：`equal`（1/K）、**`max_sharpe`**（历史日收益估 μ、Σ 后 `maximize_sharpe`，失败等权）或 **`risk_parity`**（同窗口估 Σ 后 `risk_parity`，样本不足等失败等权）。每期 `meta["rebalance_log"]` 记录选股与权重。
- **IC**：`analysis.ic` 日截面 Spearman vs 前瞻收益；不参与调仓；可落盘 `output/cache/ic_*.csv`。
- **多因子融合**：默认 **`fuse_ic_weighted_zscore`**（各因子日 IC 经 `shift(1)+rolling` 得非负列权，再对横截面 z-score 加权；失败或配置关闭时用 **`fuse_equal_weight_zscore`**）→ `run_multi_backtest(fused=...)` → 内部 `run_single_backtest`。另支持 `run_multi_backtest(factors, weights)` **线性加权** 合成得分（与上述融合不同，`main` 当前未用）。
- **优化**：`models.optimizer.maximize_sharpe` / `risk_parity`；回测在 `portfolio_weighting` 为 `max_sharpe` 或 `risk_parity` 时于再平衡日调用对应函数。流程见 [FLOW_AND_MODULES.md](./FLOW_AND_MODULES.md)。
- **数据质量**：`analysis.data_quality` 在因子面板构建后统计价格覆盖率、因子覆盖率、每日覆盖率与调仓日有效截面规模。
- 绩效：`analysis.performance.summarize`（年化收益、波动、夏普、最大回撤等）。
- **基准与超额收益**：`analysis.benchmark.equal_weight_benchmark_nav` 用当前股票池生成每日等权基准；`summarize_excess` 为每条策略补充 `excess_ann_return`、`tracking_error`、`information_ratio`。
- **换手与成本**：`analysis.turnover` 由 `meta["rebalance_log"]` 计算逐期目标权重变化，补充 `avg_turnover`、`total_turnover`、`estimated_total_cost` 等指标。

**作图**：

- `analysis/plotting.plot_nav`：支持 `Series` 或多列 `DataFrame`；`normalize=True` 时按各列首行有效值归一便于对比；`save_path` 存在时使用 `Agg` 后端写 PNG。
- `analysis/plotting.plot_ic`：日 IC 序列（或多因子对比）；单列时可选滚动均线；`persist_run_outputs` 时 `main` 写 `output/ic_compare.png`、`ic_timeseries_<因子>.png`。
- `analysis/plotting.plot_weights` + `rebalance_log_to_weights_frame`：由 `meta["rebalance_log"]` 得到宽表后堆叠面积图（或热力图）；`persist_run_outputs` 时 `main` 写 `output/weights_<因子>.png`。
- `main.py` 收集各因子、`FUSED_ZSCORE` 与 `BENCH_EQUAL_WEIGHT` 净值后保存 **`output/nav_compare.png`**；另保存 **`output/excess_nav_compare.png`** 比较各策略相对基准的超额净值。
- `analysis/plotting.plot_turnover`：绘制各策略逐期换手率；`persist_run_outputs` 时 `main` 写 **`output/turnover_compare.png`**。
- `analysis/plotting.plot_factor_coverage`：绘制因子有效覆盖率；`persist_run_outputs` 时 `main` 写 **`output/data_quality/factor_coverage.png`**。

**数据落盘（缓存）**：

- `live/cache_io.save_run_cache`：在因子面板构建成功后，将 **`prices_long.csv`**、**`prices_wide_close.csv`**、**`factor_panel.csv`**、**`run_meta.txt`** 写入 **`output/cache/`**。
- `live/cache_io.save_data_quality_reports`：写 **`output/data_quality/*.csv`**，保存价格 / 因子 / 调仓日覆盖率报告。
- `analysis.ic.save_ic_series`：在 IC 计算完成且 `persist_run_outputs` 时写 **`ic_<因子名>.csv`**。
- `live/cache_io.save_run_config`：写 **`output/cache/run_config.json`**，保存本次 `Settings` 配置快照。
- `live/cache_io.save_performance_summary`：写 **`output/performance_summary.csv`**，汇总每条策略的年化收益、波动、夏普、最大回撤，并包含相对基准、换手率与预估成本指标。
- `live/cache_io.save_rebalance_logs`：写 **`output/rebalance_logs/<策略名>.csv`**，记录每次调仓的日期、标的、权重、配权方式与排序。
- `live/cache_io.save_turnover_logs`：写 **`output/turnover_logs/<策略名>.csv`**，记录每期换手率、预估成本、持仓数与配权方式。
- `config.Settings.persist_run_outputs`（默认 `True`）为关时跳过上述写入。

**明确非 MVP（占位 / 后续）**：

- `live/signal_system.py`、`live/paper_trading.py`：`NotImplementedError`（实盘链路基座）。

---

## 2. 目录与模块职责（与代码一致）

| 路径 | 职责 |
|------|------|
| `config.py` | `Settings`：`data_dir`、`output_dir`、`backtest_start`/`end`、`rebalance_freq`（默认 `ME`）、`top_k`、`commission_rate`、`portfolio_weighting`（`equal`/`max_sharpe`/`risk_parity`）、`optimizer_return_window`、`optimizer_min_obs`、`ic_forward_days`、`fusion_use_ic_weights`、`fusion_ic_rolling_window`、`fusion_ic_min_periods`、`persist_run_outputs`、动量/波动/财务窗口等；`get_tushare_token()`（环境变量优先，本地回退**勿提交密钥**）。 |
| `main.py` | 入口：拉数 → `build_four_factor_panel` → 数据质量报告 → 可选 `save_run_cache` → IC → 可选 `save_ic_series` 与 **IC/权重 PNG** → 四因子各 `run_single_backtest(..., factor_values=列)` → **`_build_fused_zscore_panel`（IC 列权或等权）** + `run_multi_backtest(fused=...)` → 股票池等权基准与超额指标 → 换手与成本 → 绩效汇总 / 调仓日志 / 换手日志 / 配置快照落盘 → `plot_nav`。 |
| `data/` | 原始/演示数据；存在 `prices_demo.csv` 时优先读本地。 |
| `live/data_feed.py` | `fetch_daily_panel`、`fetch_fina_indicator_panel`、`load_prices_from_csv` 等。 |
| `live/cache_io.py` | `save_run_cache` → `output/cache/` 行情与因子面板；`save_data_quality_reports`、`save_run_config`、`save_performance_summary`、`save_rebalance_logs`、`save_turnover_logs` → 实验运行记录。 |
| `backtest/backtest_utils.py` | `to_returns`、`long_to_wide`、`wide_to_long`、`prices_to_wide_close`、`align_panel`。 |
| `backtest/backtest_single.py` | `run_single_backtest`：再平衡日 Top-K、**等权 / 夏普 / 风险平价**、撮合与净值；`meta` 含 `rebalance_log`、`portfolio_weighting`。 |
| `backtest/backtest_multi.py` | `run_multi_backtest`：`fused=` 或 `factors`+`weights` 合成一列后转调 `run_single_backtest`。 |
| `analysis/ic.py` | `daily_ic_spearman`、`summarize_ic`、`save_ic_series`。 |
| `factors/` | 各 `calc_*` + `FACTOR_REGISTRY` + `panel_builder`。 |
| `analysis/data_quality.py` | `price_coverage`、`factor_coverage`、`factor_daily_coverage`、`rebalance_coverage`。 |
| `analysis/performance.py` | `summarize(nav)`。 |
| `analysis/benchmark.py` | `equal_weight_benchmark_nav`、`summarize_excess`、`excess_nav_frame`。 |
| `analysis/turnover.py` | `turnover_frame`、`summarize_turnover`、`turnover_wide`。 |
| `analysis/plotting.py` | `plot_nav`、`plot_ic`、`plot_weights`、`plot_turnover`、`plot_factor_coverage`、`rebalance_log_to_weights_frame`。 |
| `models/fusion.py` | `fuse_equal_weight_zscore`、`fuse_ic_weighted_zscore`、`fuse_models`（仅部分 `method`）。 |
| `models/optimizer.py` | `maximize_sharpe`、`risk_parity`；由 `backtest_single` 在对应 `portfolio_weighting` 时再平衡日调用。 |

---

## 3. 数据形态（技术细节）

### 3.1 长表（long）

多行，每行一条记录，至少包含：

- `trade_date`（`datetime64`）
- `ts_code`（如 `600519.SH`）
- `open, high, low, close, volume`（CSV 可为 `vol`，加载时统一为 `volume`）

**用途**：Tushare 合并结果、CSV 读入、PE/ROE 与财报 `merge_asof` 的左表。

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
| 2b | 否则 Tushare | `fetch_daily_panel(main._DEFAULT_TS_SYMBOLS, ...)` → `long_to_wide` |
| 2c | 失败则合成宽表 | `_demo_price_wide()`，`wide_to_long` → `long_df` |
| 3 | 构建四因子面板 | `build_four_factor_panel(prices, long_df, settings)` → `panel` |
| 4 | 数据质量报告 | `price_coverage`、`factor_coverage`、`rebalance_coverage` |
| 5 | 可选落盘行情与面板 | `save_run_cache`（`persist_run_outputs`） |
| 6 | IC：各因子列 + **与融合同构的** FUSED 得分 | `daily_ic_spearman`、`summarize_ic`；可选 `save_ic_series` |
| 7 | 单因子回测 ×4 | `run_single_backtest(fname, factor_values=panel[fname], ...)` |
| 8 | 融合回测 ×1 | **`_build_fused_zscore_panel`**（`fuse_ic_weighted_zscore` 或等权）→ `run_multi_backtest(fused=..., factor_name="FUSED_ZSCORE", ...)` |
| 9 | 绩效与打印 | `summarize`；`_print_backtest_block` 打印 `rebalance_log`、绩效 |
| 10 | 基准与超额收益 | `equal_weight_benchmark_nav`、`summarize_excess` |
| 11 | 换手与成本 | `turnover_frame`、`summarize_turnover` |
| 12 | 实验记录落盘 | `run_config.json`、`performance_summary.csv`、`data_quality/*.csv`、`rebalance_logs/*.csv`、`turnover_logs/*.csv` |
| 13 | 净值、超额净值、覆盖率与换手图 | `plot_nav` / `plot_factor_coverage` / `plot_turnover` |

**多因子关系**：四条回测为 **同一 `panel` 的不同列** 的独立策略；第五条为 **IC 列权或等权 z-score 融合得分** 经 `run_multi_backtest` 的独立策略。调仓日 **不会**把四列现场合成后再单跑一条（合成仅在融合分支预先完成）。

---

## 5. `run_single_backtest` 内部逻辑（技术细节）

### 5.1 输入

- `prices`：宽表收盘价，或含 `trade_date, ts_code, close` 的长表（经 `prices_to_wide_close` 统一）。
- `factor_name`：`FACTOR_REGISTRY` 键。
- 可选 `factor_values`：若传入则跳过自动算子。
- 可选 `long_prices`：PE/ROE 用；缺省则由 `wide_to_long(prices_wide)` 生成。
- 可选 `finance_df`：PE/ROE 用；缺省则 `fetch_fina_indicator_panel(...)`。
- `kwargs`：`vol_window`、`token` 等。

### 5.2 因子自动分支

| `factor_name` | 数据依赖 | 计算要点 |
|----------------|----------|----------|
| `MOMENTUM` | `prices_wide` | \(P_t/P_{t-L}-1\)，`L = momentum_lookback`（默认 20），`stack` → PanelLong。 |
| `VOLATILITY` | `to_returns(prices_wide)` | 滚动标准差 × √252 年化，再取 **负号**（因子越大 = 历史波动越低）。 |
| `PE` | `fina` + `long_px` | `merge_asof(..., direction="backward")` 按 `ann_date` 对齐到 `trade_date`；`PE=close/eps`（eps>0），输出 **-PE**（因子越大 = 市盈率越低）。 |
| `ROE` | 同上 | 对齐后取 **`roe`** 列（越大越好）。 |

财务拉取区间：`backtest_start` 前再回溯 `fina_history_years`（默认 2 年），避免样本初期无可用财报。

### 5.3 再平衡日历

- `settings.rebalance_freq` 默认 **`ME`**（月末，兼容 Pandas 3；`M` 在 `backtest_single._resample_freq_alias` 中映射为 `ME`）。
- `rebalance_dates = prices_wide.resample(rf).last().index ∩ prices_wide.index`。

### 5.4 调仓日算法（横截面）

1. `sc = factor_values.xs(dt, level=0).dropna()`。
2. `sc.sort_values(ascending=False)`：**因子数值越大越优先**（各因子已通过定义统一「大=好」）。
3. 按顺序取前 `k` 只，且当日 `close` 有限：得到 `picks`。
4. **配权**：若 `portfolio_weighting == "max_sharpe"` 且样本足够，用过去 `optimizer_return_window` 日日收益估 μ、Σ，调用 `maximize_sharpe` 得权重，**`_rebalance_to_target_weights`**；若为 **`risk_parity`** 且样本足够，估 Σ 后调用 **`risk_parity`**，同样 **`_rebalance_to_target_weights`**；否则 **`_rebalance_topk_equal_weight`**（1/K）。失败标签见 `rebalance_log[].weighting`（`max_sharpe_fallback` / `risk_parity_fallback` / `equal`）。
5. **撮合**：目标市值按权重分配；**先卖后买**；手续费 `commission_rate` × 成交额；现金不足时对买入批量 **缩放**。
6. 将本日 `picks`、权重、`weighting` 记入 **`meta["rebalance_log"]`**。

### 5.5 净值

- 初始现金 `1.0`，持仓股数 `Series(0, index=symbols)`。
- 每日：`nav = cash + Σ(shares_i × price_i)`（无效价跳过该项）。
- 输出：`nav` 为 `Series`（`date` 索引），`meta` 含 `n_rebalances`、`portfolio_weighting`、`rebalance_log` 等。

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

- **股票池**：`_DEFAULT_TS_SYMBOLS`（8 只，可按权限与标的修改）。
- **因子循环顺序**：`factors.panel_builder.DEFAULT_FACTOR_ORDER`：`MOMENTUM` → `VOLATILITY` → `PE` → `ROE`（与 `main` 中 IC/回测循环一致）。

---

## 8. 依赖与运行环境

- **Python 包**：见根目录 `requirements.txt`（含 `pandas`、`numpy`、`scipy`、`matplotlib`、`scikit-learn`、`tushare` 等）。
- **解释器**：建议使用项目内 `.venv`；`.vscode/settings.json` 已指向 `${workspaceFolder}/.venv/bin/python`。
- **Tushare**：日线与 `fina_indicator` 均需 **积分/权限**；失败时 `main` 会回退合成数据（PE/ROE 在无财务时可能跳过或报错，取决于分支）。

---

## 9. 后续可补充方向（与代码占位对应）

| 方向 | 说明 |
|------|------|
| 作图 | `plot_nav` / `plot_ic` / `plot_weights`；`persist_run_outputs` 时写 IC 与权重 PNG（见 `main`）。 |
| 实盘链路 | `signal_system.generate_signals`、`paper_trading.run_paper_trading`。 |
| 融合扩展 | `fuse_models` 更多 `method`（如 `dynamic`、`xgboost`）。 |
| 数据 | 启动时读 `output/cache` 命中则跳过 Tushare；或规范落盘至 `data/`。 |
| 检验 | 扩展 `unittest`/`pytest`（含 `tests/test_optimizer.py`、`test_backtest_*`、`test_plotting.py`、`test_fusion.py`）。 |
| 更真实成交 | 次日开盘、停牌、涨跌停、最小交易单位等。 |
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
