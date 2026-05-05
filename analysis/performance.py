"""
净值序列 -> 年化收益、波动、夏普、最大回撤（契约见 docs/INTERFACE_AND_CONTRACTS.md）。
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def summarize(
    nav: pd.Series,
    *,
    risk_free: float = 0.0,
    periods: int = 252,
) -> Dict[str, Any]:
    """
    :param nav: 日频净值，索引递增 date
    :param risk_free: 年化无风险利率（简单处理时可仅用于夏普分子调整）
    :param periods: 年化用交易日数
    """
    nav = nav.astype(float).dropna()
    if nav.empty:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }
    ret = nav.pct_change().fillna(0.0)
    n = len(nav)
    ann_return = float((nav.iloc[-1] / nav.iloc[0]) ** (periods / max(n - 1, 1)) - 1)
    ann_vol = float(ret.std() * np.sqrt(periods))
    excess = ann_return - risk_free
    sharpe = float(excess / ann_vol) if ann_vol > 1e-12 else float("nan")
    cum = (1 + ret).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    mdd = float(dd.min())
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
    }
