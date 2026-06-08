"""
因子清洗与横截面标准化。

输入为 MultiIndex(date, symbol) × 因子列的原始面板；输出保持同样索引和列。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def winsorize_series(
    s: pd.Series,
    *,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.Series:
    """按分位数对单条 Series 去极值，NaN 保持 NaN。"""
    x = s.astype(float)
    valid = x.dropna()
    if valid.empty:
        return x
    lo = float(valid.quantile(float(lower_q)))
    hi = float(valid.quantile(float(upper_q)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
        return x
    return x.clip(lower=lo, upper=hi)


def cross_sectional_zscore(
    panel: pd.DataFrame,
    *,
    winsorize: bool = True,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    min_count: int = 2,
    constant_fill: Optional[float] = 0.0,
) -> pd.DataFrame:
    """
    对每个交易日、每一列因子，在当日股票池上做去极值与横截面 z-score。

    当日有效样本少于 min_count，或标准差过小时，若 constant_fill 非 None 则填该值，否则保留 NaN。
    """
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise TypeError("panel 须为 MultiIndex(date, symbol) × 因子列")

    out = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    dates = panel.index.get_level_values(0).unique()
    for dt in dates:
        idx = panel.index.get_level_values(0) == dt
        sub = panel.loc[idx].astype(float)
        for col in sub.columns:
            s = sub[col]
            x = winsorize_series(s, lower_q=lower_q, upper_q=upper_q) if winsorize else s
            valid = x.dropna()
            if len(valid) < int(min_count):
                if constant_fill is not None:
                    out.loc[idx, col] = float(constant_fill)
                continue
            mu = float(valid.mean())
            sig = float(valid.std(ddof=0))
            if not np.isfinite(sig) or sig < 1e-12:
                if constant_fill is not None:
                    out.loc[idx, col] = float(constant_fill)
                continue
            out.loc[idx, col] = (x - mu) / sig
    out.index = out.index.set_names(["date", "symbol"])
    return out


def preprocess_factor_panel(
    panel: pd.DataFrame,
    *,
    winsorize: bool = True,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    min_count: int = 2,
) -> pd.DataFrame:
    """生成清洗后的横截面 z-score 因子面板。"""
    return cross_sectional_zscore(
        panel,
        winsorize=winsorize,
        lower_q=lower_q,
        upper_q=upper_q,
        min_count=min_count,
        constant_fill=0.0,
    )
