"""
全局配置：路径、回测区间、费率、再平衡等。

Tushare Token：优先读环境变量 TUSHARE_TOKEN；未设置时使用下方本地回退。
本地回退仅用于你本机跑通流程；若将仓库推送到远程，请先清空 _TUSHARE_TOKEN_LOCAL 或改用环境变量。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# 本地开发回退（勿提交含真实 Token 的版本到 Git）
_TUSHARE_TOKEN_LOCAL = ""


@dataclass(frozen=True)
class Settings:
    """项目级只读配置（契约见 docs/INTERFACE_AND_CONTRACTS.md）。"""

    project_root: Path
    data_dir: Path
    output_dir: Path
    stock_pool_path: Path
    stock_pool_code_col: str = "股票代码"
    tushare_price_cache_path: Path | None = None
    price_col: str = "close"
    commission_rate: float = 0.0003
    rebalance_freq: str = "ME"
    force_final_rebalance: bool = False
    trading_days_per_year: int = 252
    backtest_start: str = "2024-01-01"
    backtest_end: str = "2025-01-01"
    top_k: int = 5
    momentum_lookback: int = 20
    momentum_long_lookback: int = 60
    reversal_lookback: int = 5
    volume_ratio_window: int = 20
    vol_window: int = 20
    fina_history_years: int = 2
    persist_run_outputs: bool = True
    # IC：因子 @ 日 t 与前瞻收盘收益 close(t+h)/close(t)-1 的截面 Spearman；h=1 为最常见日频口径
    ic_forward_days: int = 1
    # IC 稳定性诊断：对日 IC 做滚动均值/波动/正值占比统计的窗口。
    ic_rolling_windows: tuple[int, ...] = (20, 60)
    # 因子分组收益：按调仓日横截面分成 N 组，观察低分组到高分组的收益单调性。
    factor_group_count: int = 5
    # 融合：True 时用各因子日 IC 的 shift(1)+rolling 均值做 z-score 后列权（见 models.fusion.fuse_ic_weighted_zscore）
    fusion_use_ic_weights: bool = True
    fusion_ic_rolling_window: int = 60
    fusion_ic_min_periods: int = 20
    # 综合评分静态融合：用前半段样本计算 factor_score/fusion_weight，后半段验证 FUSED_SCORE_WEIGHTED。
    factor_weight_train_ratio: float = 0.5
    # 综合评分滚动融合：每个调仓日前只用历史窗口计算因子权重，再用于当期 FUSED_ROLLING_SCORE_WEIGHTED。
    rolling_factor_weight_lookback_days: int = 120
    rolling_factor_weight_min_days: int = 60
    rolling_factor_weight_min_weight: float = 0.05
    rolling_factor_weight_max_weight: float = 0.60
    rolling_factor_weight_smoothing: float = 0.5
    # 回测内持仓权重：equal=Top-K 等权；max_sharpe=历史收益估 mu/cov 后夏普最大化；risk_parity=同窗口估 cov 后 ERC（失败等权）
    portfolio_weighting: str = "max_sharpe"
    # 单票目标权重上限；0 或 >=1 表示不启用。若 top_k 太小导致上限不可行，则回测保留可行的等权/原权重。
    max_position_weight: float = 0.4
    # 单次再平衡目标权重变化上限；0 表示不启用。首次建仓不节流，避免长期停留在现金状态。
    max_rebalance_turnover: float = 1.0
    # 可交易性过滤：0 表示关闭；开启后在 Top-K 选股前按过去窗口平均成交量/成交额过滤候选股票。
    liquidity_lookback_days: int = 20
    min_avg_volume: float = 0.0
    min_avg_amount: float = 0.0
    # 交易状态约束：默认关闭；开启后读取 is_suspended/is_limit_up/is_limit_down，限制停牌与涨跌停下的买卖。
    enable_trade_status_filter: bool = False
    # 行业权重上限：0 表示关闭；开启后读取 industry 列，限制单个行业目标权重占比。
    max_industry_weight: float = 0.0
    industry_col: str = "industry"
    # 波动率目标：0 表示关闭；开启后按历史协方差估算组合年化波动，超目标时降低股票仓位，剩余保留现金。
    target_volatility: float = 0.0
    volatility_target_lookback_days: int = 60
    volatility_target_min_obs: int = 20
    # 最小持仓数量：0 表示关闭；若有效目标持仓数不足，则把股票总仓位缩到 min_positions_exposure，剩余保留现金。
    min_positions: int = 0
    min_positions_exposure: float = 1.0
    # 订单生成：A 股默认 100 股一手；低于最小订单金额的调仓会被过滤，减少碎片订单。
    order_lot_size: int = 100
    min_order_amount: float = 0.0
    # 订单预检查：买入后至少保留的现金缓冲。
    order_cash_buffer: float = 0.0
    # 纸面交易：虚拟账户默认初始资金。
    paper_initial_cash: float = 1_000_000.0
    optimizer_return_window: int = 60
    optimizer_min_obs: int = 15


def get_settings() -> Settings:
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    output_env = os.environ.get("QUANT_OUTPUT_DIR", "").strip()
    output_dir = Path(output_env).expanduser() if output_env else root / "output"
    stock_pool_env = os.environ.get("QUANT_STOCK_POOL_PATH", "").strip()
    stock_pool_path = Path(stock_pool_env).expanduser() if stock_pool_env else data_dir / "stock_pool.xlsx"
    cache_env = os.environ.get("QUANT_TUSHARE_PRICE_CACHE", "").strip()
    tushare_cache = Path(cache_env).expanduser() if cache_env else data_dir / "prices_tushare_cache.csv"
    start = os.environ.get("QUANT_BACKTEST_START", "").strip() or Settings.backtest_start
    end = os.environ.get("QUANT_BACKTEST_END", "").strip() or Settings.backtest_end
    force_final_rebalance = os.environ.get("QUANT_FORCE_FINAL_REBALANCE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    return Settings(
        project_root=root,
        data_dir=data_dir,
        output_dir=output_dir,
        stock_pool_path=stock_pool_path,
        tushare_price_cache_path=tushare_cache,
        backtest_start=start,
        backtest_end=end,
        force_final_rebalance=force_final_rebalance,
    )


def get_tushare_token() -> str:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if token:
        return token
    local = _TUSHARE_TOKEN_LOCAL.strip()
    if local:
        return local
    raise ValueError("请设置环境变量 TUSHARE_TOKEN，或在 config._TUSHARE_TOKEN_LOCAL 填写本地回退（勿提交）")
