"""
多因子融合：横截面 z-score 后等权平均，或 **按滞后滚动 IC 均值的非负归一化列权**（IC 驱动融合，最小切片）。

输入为「多列因子」宽面板：行索引为 MultiIndex(date, symbol)，每列为原始因子值。
"""
from __future__ import annotations

from typing import Literal, Mapping

import numpy as np
import pandas as pd

FusionMethod = Literal["mean_zscore", "mean", "dynamic", "xgboost"]


def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """
    对每个交易日、每一列因子，在当日股票池上做横截面 z-score：(x - mean) / std。
    当日 std 过小或无效时该列当日记 0；单点有效样本过少时该列当日记 0。
    """
    out = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    dates = panel.index.get_level_values(0).unique()
    for dt in dates:
        idx = panel.index.get_level_values(0) == dt
        sub = panel.loc[idx].astype(float)
        for col in sub.columns:
            s = sub[col]
            valid = s.dropna()
            if len(valid) < 2:
                out.loc[idx, col] = 0.0
                continue
            mu = float(valid.mean())
            sig = float(valid.std(ddof=0))
            if not np.isfinite(sig) or sig < 1e-12:
                out.loc[idx, col] = 0.0
            else:
                out.loc[idx, col] = (s - mu) / sig
    return out


def fuse_equal_weight_zscore(panel: pd.DataFrame) -> pd.Series:
    """
    先 `cross_sectional_zscore`，再对列等权平均，得到单一综合得分（越大越好）。
    """
    z = cross_sectional_zscore(panel)
    fused = z.mean(axis=1)
    fused = fused.astype(float)
    fused.name = "fused_zscore_mean"
    fused.index = fused.index.set_names(["date", "symbol"])
    return fused


def fuse_ic_weighted_zscore(
    panel: pd.DataFrame,
    ic_by_factor: Mapping[str, pd.Series],
    *,
    rolling_window: int = 60,
    min_periods: int = 20,
) -> pd.Series:
    """
    横截面 z-score 后，按 **各因子日频 IC 的滞后滚动均值** 做列权（非负、按日归一），再加权求和得到融合得分。

    **含义（最小切片）**：近期 IC 越高的因子在综合分里权越大；IC 长期为负的因子权趋近 0（仍保留列，
    极端情况若当日全非正则回退当日等权）。这样 IC 从纯分析进入 **融合得分** 这一条决策链，再进入
    Top-K 回测；**不**改变单因子回测、也不把 IC 直接写入股票层面的 mean–variance 优化。

    **信息集**：日 t 使用的权重仅依赖 **IC 在 t-1 及更早**（先 `ic.shift(1)` 再 rolling），避免用「同日 IC」
    调当日 z-score 的朴素前视；滚动窗口内样本不足时用等权（与 `min_periods` 一致）。

    :param panel: MultiIndex(date, symbol) × 因子列（与 `fuse_equal_weight_zscore` 相同）。
    :param ic_by_factor: 各因子名 → 日频 IC 序列（索引为日历日，须覆盖 panel 涉及日期）。
    :param rolling_window: 滚动均值的窗口长度（交易日）。
    :param min_periods: rolling 最少有效点数，不足则该日列权回退等权。
    """
    cols = list(panel.columns)
    if not cols:
        raise ValueError("panel 至少须有一列因子")
    for c in cols:
        if c not in ic_by_factor:
            raise KeyError("ic_by_factor 缺少列 %r" % c)

    dates = pd.Index(panel.index.get_level_values(0).unique()).sort_values()
    z = cross_sectional_zscore(panel)

    # 权重矩阵：index=date, columns=因子
    wdf = pd.DataFrame(index=dates, columns=cols, dtype=float)
    for col in cols:
        ic_s = ic_by_factor[col].astype(float).sort_index()
        roll = ic_s.shift(1).rolling(int(rolling_window), min_periods=int(min_periods)).mean()
        wdf[col] = roll.reindex(dates)

    n = len(cols)
    eq = np.ones(n, dtype=float) / max(n, 1)
    pieces: list[pd.Series] = []
    for dt in dates:
        idx = panel.index.get_level_values(0) == dt
        row = wdf.loc[dt]
        raw = np.maximum(row.to_numpy(dtype=float), 0.0)
        ssum = float(np.nansum(raw))
        if not np.isfinite(ssum) or ssum <= 1e-18:
            wvec = eq
        else:
            wvec = raw / ssum
        wser = pd.Series(wvec, index=cols)
        zsub = z.loc[idx]
        fused_dt = (zsub * wser).sum(axis=1)
        fused_dt.name = "fused"
        pieces.append(fused_dt)

    fused = pd.concat(pieces).sort_index()
    fused = fused.astype(float)
    fused.name = "fused_zscore_ic_weighted"
    fused.index = fused.index.set_names(["date", "symbol"])
    return fused


def fuse_models(
    scores: pd.DataFrame,
    *,
    method: FusionMethod = "mean_zscore",
    window: int = 3,
    **kwargs: object,
) -> pd.Series:
    """
    :param scores: MultiIndex(date, symbol) × 多列因子（原始值，未 z-score）
    :param method: mean_zscore = 横截面 z-score 后列平均；mean = 不做 z-score 直接行平均
    :param window: 预留（dynamic / xgboost）
    """
    _ = window, kwargs
    if method == "mean_zscore":
        return fuse_equal_weight_zscore(scores)
    if method == "mean":
        s = scores.mean(axis=1)
        s.index = s.index.set_names(["date", "symbol"])
        s.name = "fused_raw_mean"
        return s
    raise NotImplementedError("method=%s 未实现（仅支持 mean_zscore、mean）" % method)
