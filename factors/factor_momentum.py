"""
动量因子。输出契约：PanelLong，索引 (date, symbol)。
"""
from __future__ import annotations

import pandas as pd


def calc_momentum(
    prices_wide: pd.DataFrame,
    *,
    lookback: int = 20,
    price_col: str = "close",
) -> pd.Series:
    """
    :param prices_wide: 索引为 date，列为 symbol 的收盘价宽表
    :param lookback: 动量窗口（交易日）
    :param price_col: 宽表已是价格矩阵时忽略；保留以兼容契约
    :return: MultiIndex(date, symbol) 的动量（过去 lookback 日总收益）
    """
    _ = price_col
    if lookback < 1:
        raise ValueError("lookback 须 >= 1")
    px = prices_wide.sort_index().sort_index(axis=1).astype(float)
    mom = px / px.shift(lookback) - 1.0
    s = mom.stack()
    s.index.names = ["date", "symbol"]
    return s
