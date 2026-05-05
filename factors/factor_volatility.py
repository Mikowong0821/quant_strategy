"""
波动率因子。输出契约：PanelLong，索引 (date, symbol)。

返回值为「负的年化波动率」：数值越大表示历史波动越低，便于与回测层「因子越大越优先」一致（低波偏好）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calc_volatility(
    returns_wide: pd.DataFrame,
    *,
    window: int = 20,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """
    :param returns_wide: 索引为 date，列为 symbol 的日简单收益率宽表
    :return: MultiIndex(date, symbol)，值为 -sigma（年化），越大越好 = 波动越低越好
    """
    if window < 2:
        raise ValueError("window 须 >= 2")
    r = returns_wide.sort_index().sort_index(axis=1).astype(float)
    min_p = max(5, min(window // 2, window))
    vol = r.rolling(window, min_periods=min_p).std()
    if annualize:
        vol = vol * float(np.sqrt(trading_days))
    score = -vol
    s = score.stack()
    s.index.names = ["date", "symbol"]
    return s
