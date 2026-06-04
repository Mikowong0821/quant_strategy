"""
成交量类因子。输出契约：PanelLong，索引 (date, symbol)。
"""
from __future__ import annotations

import pandas as pd


def calc_volume_ratio(
    volume_wide: pd.DataFrame,
    *,
    window: int = 20,
) -> pd.Series:
    """
    成交量放大因子：volume / rolling_mean(volume) - 1。

    数值越大表示当日成交量相对过去窗口均量放大越明显。
    """
    if window < 2:
        raise ValueError("window 须 >= 2")
    vol = volume_wide.sort_index().sort_index(axis=1).astype(float)
    min_p = min(window, max(2, window // 2))
    base = vol.rolling(window, min_periods=min_p).mean()
    ratio = vol / base - 1.0
    s = ratio.stack()
    s.index.names = ["date", "symbol"]
    return s
