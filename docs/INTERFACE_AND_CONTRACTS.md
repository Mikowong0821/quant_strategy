# 接口与数据契约（Interface & Data Contracts）

本文档约定各模块的**输入/输出形态**与**对齐规则**，不规定具体算法。实现时可替换内部逻辑，但**对外契约**应尽量稳定，便于回测与实盘共用同一套数据结构。

---

## 1. 时间与标识

| 字段 | 类型 | 说明 |
|------|------|------|
| `trade_date` / `date` | `datetime64[ns]` 或 `YYYY-MM-DD` 字符串（读入后统一为日频 Timestamp，**交易日**，非自然日需与行情对齐） | 截面排序、再平衡、标签对齐的基准轴 |
| `symbol` / `ts_code` | `str` | 如 `600519.SH`、`000001.SZ`；**全项目统一一种命名**（推荐 Tushare 风格 `ts_code`） |

**约定**：长表（long）面板使用二级索引 `(date, symbol)`，且 `date` 升序、`symbol` 字典序，无重复键。

---

## 2. 磁盘数据（`data/`）

### 2.1 行情：`data/stock_<ts_code中的数字部分>.csv` 或聚合多标的 `prices.csv`（二选一需在 `config` 中声明）

**最低必需列**：

| 列名 | dtype | 说明 |
|------|--------|------|
| `trade_date` | date | 交易日 |
| `ts_code` | str | 标的代码 |
| `open` | float64 | 开盘价 |
| `high` | float64 | 最高价 |
| `low` | float64 | 最低价 |
| `close` | float64 | 收盘价（因子与回测默认用收盘价，除非接口显式传入 `price_col`） |
| `vol` 或 `volume` | float64 | 成交量；读入后可在 `data_feed` 层统一重命名为 `volume` |

可选：`amount`（成交额）、`pct_chg` 等；**缺列时由加载层报错或按 config 填充策略处理**。

### 2.2 财务：`data/finance_data.csv`

**最低必需列**（用于 PE/ROE 等；具体财报发布日对齐在因子层实现，契约只要求列存在）：

| 列名 | dtype | 说明 |
|------|--------|------|
| `ts_code` | str | 标的 |
| `end_date` 或 `ann_date` | date | 报告期或公告日（**必须在因子模块 docstring 中写明使用哪一种**） |
| `eps` / `netprofit` / `total_holders_equity` 等 | float64 | 由具体因子声明子集；缺失则该 (date, symbol) 因子为 NaN |

财务与行情对齐：**不做全局强制**；由 `factors/*` 在输出前将结果 reindex 到目标 `(date, symbol)` 网格。

---

## 3. 内存中的核心对象

### 3.1 `PanelLong`（长表面板）

- **类型**：`pd.Series` 或 `pd.DataFrame`，**索引**为 `pd.MultiIndex`，名称为 `["date", "symbol"]`。
- **语义**：每个 `(date, symbol)` 一条记录；用于因子值、信号、权重（单列用 Series，多列用 DataFrame）。

### 3.2 `PricePanel`（宽表，可选）

- **类型**：`pd.DataFrame`，索引 `date`，列 `symbol`，值为 `close`（或 OHLC 用 `pd.MultiIndex` columns，**若采用需在 config 固定**）。
- **用途**：部分回测/优化内部计算；**从磁盘加载后可在 `backtest_utils` 中转换为宽表**。

### 3.3 `NavSeries`（净值曲线）

- **类型**：`pd.Series`，索引 `date`，值 `float`，名称建议 `nav`。
- **约束**：严格递增索引、无重复日期；首日可为 1.0 或实际资金，**全项目统一在 `config`**。

### 3.4 `ICSeries`（optional）

- **类型**：`pd.Series`，索引 `date`，值为当日截面 IC（因子 vs 未来收益）。
- **默认实现**（`analysis.ic`）：截面 **Spearman** 相关；因子在日期 `t` 与前瞻收益 **close(t+h)/close(t)−1** 对齐，`h` 由 `config.ic_forward_days` 指定（默认 `1`）。因子在 `(t, symbol)` 处须仅依赖 **≤t 已公开数据**（与 §5.3 一致）；收益窗口起点为 `t` 收盘、不含更早未公开信息。

---

## 4. 模块级接口（函数签名契约）

命名与参数以代码中 docstring 为准；此处为**逻辑契约摘要**。

| 模块 | 函数 | 输入契约 | 输出契约 |
|------|------|-----------|-----------|
| `config` | `get_settings()` | 无 | 只读配置对象（路径、费率、回测区间、价格列名等） |
| `live.data_feed` | `get_data_tushare(symbol, start, end, ...)` | 合法 `ts_code`、ISO 日期 | 满足 §2.1 列规范的 `pd.DataFrame`（可含额外列） |
| `live.data_feed` | `load_prices_from_csv(path_or_glob)` | 磁盘路径 | 长表或宽表 + 元数据说明（推荐返回 long 并标准化列名） |
| `live.cache_io` | `save_run_cache(settings, long_df, prices_wide, panel)` | `Settings`、行情与面板 | 写 `output/cache/` 下 `prices_long.csv` 等 |
| `live.cache_io` | `save_run_config`、`save_performance_summary`、`save_rebalance_logs`、`save_turnover_logs`、`save_risk_exposure_logs`、`save_risk_exposure_summary`、`save_data_quality_reports` | `Settings`、绩效 dict、回测 meta、换手表、集中度表、数据质量表 | 写 `run_config.json`、`performance_summary.csv`、`rebalance_logs/*.csv`、`turnover_logs/*.csv`、`risk_exposure/*.csv`、`data_quality/*.csv` |
| `factors.factor_*` | `calc_*(..., **kwargs)` | 行情/财务 DataFrame 或 PanelLong | `PanelLong`（Series 或单列表 DataFrame） |
| `backtest.backtest_utils` | `to_returns(prices, price_col="close", ...)` | 宽表或长表（需约定） | 宽表 `pct_change` 或与输入同型的收益 |
| `backtest.backtest_utils` | `align_panel(factor, prices, ...)` | 因子与价格时间轴 | 对齐后的联合索引，缺失为 NaN |
| `backtest.backtest_single` | `run_single_backtest(factor_name, ...)` | `factor_name` 或预计算因子、`Settings.portfolio_weighting`（`equal` / `max_sharpe` / `risk_parity`）、`Settings.max_position_weight`、`Settings.max_rebalance_turnover` | `NavSeries` + `meta`（含 `rebalance_log`：每期 `date/picks/selected_picks/weights/weighting/target_turnover/turnover_capped/turnover_scale`、`portfolio_weighting`、`max_position_weight`、`max_rebalance_turnover` 等） |
| `backtest.backtest_multi` | `run_multi_backtest(fused=..., prices=...)` 或 `run_multi_backtest(factors, weights=..., prices=...)` | 已融合得分 **或** 多列因子 + 线性权重 | `NavSeries` + `meta`（含 `multi_mode`：`pre_fused` / `linear_weight`） |
| `models.optimizer` | `maximize_sharpe` / `risk_parity` | `mu`、`cov` 与标的顺序一致（`risk_parity` 仅需 `cov`） | 权重向量；`maximize_sharpe` / `risk_parity` 在对应 `portfolio_weighting` 时由回测于再平衡日调用 |
| `models.fusion` | `fuse_equal_weight_zscore`、`fuse_ic_weighted_zscore`、`fuse_models(...)` | 多列因子 Panel；`fuse_ic` 另需各列日 IC `Series` | 单列综合得分 `PanelLong` |
| `analysis.ic` | `daily_ic_spearman`、`summarize_ic`、`save_ic_series` | `PanelLong` 单列、价格宽表、`forward_days` / `Settings.ic_forward_days` | `ICSeries`、汇总 dict、可选 `output/cache/ic_*.csv` |
| `analysis.performance` | `summarize(nav, risk_free=0.0, periods=252)` | `NavSeries` | `dict`：`ann_return`, `ann_vol`, `sharpe`, `max_drawdown`, … |
| `analysis.data_quality` | `price_coverage`、`factor_coverage`、`factor_daily_coverage`、`rebalance_coverage` | 价格宽表、因子面板、调仓日序列 | 价格/因子/调仓日覆盖率报告 |
| `analysis.benchmark` | `equal_weight_benchmark_nav`、`summarize_excess`、`excess_nav_frame` | 价格宽表 / 策略净值 / 基准净值 | 股票池等权基准、超额收益指标、超额净值宽表 |
| `analysis.turnover` | `turnover_frame`、`summarize_turnover`、`turnover_wide` | `meta["rebalance_log"]`、手续费率 | 逐期换手表、换手/成本汇总、换手宽表 |
| `analysis.risk_exposure` | `concentration_frame`、`summarize_concentration`、`effective_n_wide` | `meta["rebalance_log"]` | 逐期集中度表、集中度汇总、effective_n 宽表 |
| `analysis.plotting` | `plot_nav`、`plot_ic`、`plot_weights`、`plot_turnover`、`plot_effective_n`、`plot_factor_coverage`、`rebalance_log_to_weights_frame` | 净值；日 IC；权重宽表；换手宽表；effective_n 宽表；覆盖率表 | `save_path` 有值则 Agg 写 PNG，否则 `show` |
| `live.signal_system` | `generate_signals(fused_score, rules, ...)` | `PanelLong` | `PanelLong` 取值 ∈ {-1, 0, 1} 或连续仓位 |
| `live.paper_trading` | `run_paper_trading(symbols, ...)` | 标的列表 + config | 日志 / 成交记录 DataFrame（契约：列含 `date`, `symbol`, `side`, `qty`, `price`） |

---

## 5. 对齐与缺失值

1. **再平衡日**：由 `config.rebalance_freq`（pandas offset 字符串，**月末建议 `ME`**；pandas 3 起 `M` 已弃用，`backtest_single` 内会将 `M` 映射为 `ME`）或显式 `rebalance_dates` 提供；回测模块仅在再平衡日更新目标权重。
2. **停牌/缺失价**：该日该标的不参与交易；若因子为 NaN，**默认剔除该标的于该截面**（或在单因子回测中记为「无效」，由 `backtest_utils` 统一策略）。
3. **未来函数**：因子 `calc_*` 的输出在日期 `t` 必须**仅依赖 ≤ t 的公开数据**；标签（供 fusion 中 ML 使用）在单独函数中计算，**不得**与因子同文件混写而不标注。

---

## 6. 配置与安全

### 6.1 `Settings` 中与主流程强相关的字段（默认值以 `config.py` 为准）

| 字段 | 说明 |
|------|------|
| `project_root` / `data_dir` / `output_dir` | 路径；缓存默认 `output_dir/cache/` |
| `backtest_start` / `backtest_end` | 回测区间（字符串 ISO 日期） |
| `rebalance_freq` | 再平衡频率，默认 `ME`（月末） |
| `top_k` | 每期多头只数 |
| `commission_rate` | 单边手续费率 |
| `momentum_lookback` / `momentum_long_lookback` / `reversal_lookback` / `volume_ratio_window` / `vol_window` | 量价因子的默认窗口：短动量、长动量、短反转、成交量放大、低波 |
| `portfolio_weighting` | `equal` / `max_sharpe` / `risk_parity`（Top-K 内等权、夏普最大化或风险平价 ERC；后两者样本不足时回退等权） |
| `max_position_weight` | 单票目标权重上限；默认 `0.4`，目标权重超过上限时裁剪并重新分配，若因持仓数过少不可行则保留原归一权重 |
| `max_rebalance_turnover` | 单次再平衡目标权重变化上限；默认 `1.0`，首次建仓不节流，`0` 表示关闭 |
| `optimizer_return_window` / `optimizer_min_obs` | 夏普配权用历史收益窗口与最少样本数 |
| `ic_forward_days` | IC 前瞻收益 horizon（交易日） |
| `fusion_use_ic_weights` | `True`（默认）时融合用 `fuse_ic_weighted_zscore`；`False` 时用等权 `fuse_equal_weight_zscore` |
| `fusion_ic_rolling_window` / `fusion_ic_min_periods` | IC 列权：对 `ic.shift(1)` 做 rolling 均值时的窗口与最少样本数 |
| `persist_run_outputs` | 是否写 `output/cache/` 下行情、面板、IC CSV、运行配置，以及 `output/` 下绩效汇总、数据质量报告、调仓日志、换手日志、集中度日志、净值/超额净值/换手/集中度图表等 |

### 6.2 Token 与路径

- **API Token**：优先环境变量 `TUSHARE_TOKEN`；当前工程在 `config.py` 中允许**本地回退**（便于本机跑通）。**含密钥的 `config.py` 勿推送到远程仓库**。
- **路径**：通过 `get_settings()` 的 `data_dir`、`output_dir` 访问 `data/`、`output/`，避免硬编码散落。

---

## 7. 版本与演进

- 若从宽表改为长表为主，**优先在 `backtest_utils` 增加转换函数**，而非修改所有因子文件。
- 新增因子时：在 `factors/__init__.py` 或注册表中登记 `FACTOR_REGISTRY[name] = callable`，供 `run_single_backtest("NAME")` 解析（`main` 当前对手传 `factor_values` 路径可不调注册表）。
