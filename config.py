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
    price_col: str = "close"
    commission_rate: float = 0.0003
    rebalance_freq: str = "ME"
    trading_days_per_year: int = 252
    backtest_start: str = "2024-01-01"
    backtest_end: str = "2025-01-01"
    top_k: int = 5
    momentum_lookback: int = 20
    vol_window: int = 20
    fina_history_years: int = 2
    persist_run_outputs: bool = True
    # IC：因子 @ 日 t 与前瞻收盘收益 close(t+h)/close(t)-1 的截面 Spearman；h=1 为最常见日频口径
    ic_forward_days: int = 1
    # 融合：True 时用各因子日 IC 的 shift(1)+rolling 均值做 z-score 后列权（见 models.fusion.fuse_ic_weighted_zscore）
    fusion_use_ic_weights: bool = True
    fusion_ic_rolling_window: int = 60
    fusion_ic_min_periods: int = 20
    # 回测内持仓权重：equal=Top-K 等权；max_sharpe=历史收益估 mu/cov 后夏普最大化；risk_parity=同窗口估 cov 后 ERC（失败等权）
    portfolio_weighting: str = "max_sharpe"
    optimizer_return_window: int = 60
    optimizer_min_obs: int = 15


def get_settings() -> Settings:
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    output_dir = root / "output"
    return Settings(project_root=root, data_dir=data_dir, output_dir=output_dir)


def get_tushare_token() -> str:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if token:
        return token
    local = _TUSHARE_TOKEN_LOCAL.strip()
    if local:
        return local
    raise ValueError("请设置环境变量 TUSHARE_TOKEN，或在 config._TUSHARE_TOKEN_LOCAL 填写本地回退（勿提交）")
