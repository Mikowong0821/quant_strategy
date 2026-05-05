"""
实盘选股信号：均线、因子阈值等。输出为 PanelLong，取值约定见契约。
"""
from __future__ import annotations

import pandas as pd


def generate_signals(
    fused_score: pd.Series,
    *,
    method: str = "quantile",
    **kwargs: object,
) -> pd.Series:
    """
    :param fused_score: MultiIndex(date, symbol)
    :return: 信号 Series，建议取值 -1 / 0 / 1 或连续目标仓位
    """
    raise NotImplementedError("与 paper_trading 约定的离散/连续信号")
