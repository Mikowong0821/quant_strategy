"""
短期反转因子。输出契约：PanelLong，索引 (date, symbol)。

返回值为「负的过去 N 日收益」：数值越大表示近期跌得越多，便于与回测层
「因子越大越优先」保持一致。
"""
from __future__ import annotations

import pandas as pd


def calc_reversal(
    prices_wide: pd.DataFrame,
    *,
    lookback: int = 5,
) -> pd.Series:
    """
    :param prices_wide: 索引为 date，列为 symbol 的收盘价宽表
    :param lookback: 反转窗口（交易日）
    :return: MultiIndex(date, symbol) 的短期反转得分
    """
    if lookback < 1:
        raise ValueError("lookback 须 >= 1")
    px = prices_wide.sort_index().sort_index(axis=1).astype(float)
    ret = px / px.shift(lookback) - 1.0
    score = -ret
    s = score.stack()
    s.index.names = ["date", "symbol"]
    return s
