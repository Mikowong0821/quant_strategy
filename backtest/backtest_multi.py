"""
多因子回测：在已有「综合得分」或「多列因子 + 线性权重」上，复用单因子回测引擎。

典型用法：
- 已由 ``models.fusion`` 得到一列 ``fused`` → 传入 ``fused=``，等价于 ``run_single_backtest(..., factor_values=fused)``，
  但元信息会标明多因子来源。
- 多列原始因子 + ``weights`` → 本模块先做按行线性加权合成一列得分，再走 Top-K 与调仓逻辑。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from backtest.backtest_single import run_single_backtest
from config import Settings, get_settings


def _factors_to_dataframe(factors: Union[List[pd.Series], pd.DataFrame]) -> pd.DataFrame:
    if isinstance(factors, pd.DataFrame):
        return factors.copy()
    if not isinstance(factors, list) or len(factors) == 0:
        raise ValueError("factors 须为非空 list[Series] 或 DataFrame")
    parts: List[pd.Series] = []
    for i, s in enumerate(factors):
        if not isinstance(s, pd.Series):
            raise TypeError("factors 列表元素须为 pandas.Series")
        nm = s.name if s.name is not None else "f%d" % i
        parts.append(s.rename(nm))
    return pd.concat(parts, axis=1)


def _linear_fuse(df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != df.shape[1]:
        raise ValueError("weights 长度须等于因子列数 (%d)" % df.shape[1])
    if not np.all(np.isfinite(w)):
        raise ValueError("weights 须为有限数值")
    s = float(np.sum(w))
    if s <= 0.0:
        raise ValueError("weights 元素和须为正")
    w = w / s
    return (df.mul(w, axis=1)).sum(axis=1)


def run_multi_backtest(
    factors: Optional[Union[List[pd.Series], pd.DataFrame]] = None,
    *,
    weights: Optional[List[float]] = None,
    fused: Optional[pd.Series] = None,
    prices: Optional[pd.DataFrame] = None,
    settings: Optional[Settings] = None,
    factor_name: str = "MULTI_FUSED",
    top_k: Optional[int] = None,
    **kwargs: Any,
) -> tuple[pd.Series, Dict[str, Any]]:
    """
    多因子回测入口：合成单列得分后调用 ``run_single_backtest``。

    :param factors: 多列因子，索引均为 (date, symbol)；与 ``weights`` 联用；与 ``fused`` 二选一
    :param weights: 与 ``factors`` 列顺序一致的线性权重，默认各列等权
    :param fused: 已合成的一列综合得分（如 ``fuse_equal_weight_zscore`` 输出）；若提供则忽略 ``factors``/``weights``
    :param prices: 行情宽表或契约长表
    :param settings: 回测与优化参数
    :param factor_name: 写入 meta 的策略名（如 ``FUSED_ZSCORE``）
    :param top_k: 覆盖 ``settings.top_k``
    :return: ``(nav, meta)``；meta 在单因子基础上增加 ``multi_mode``、``multi_weights`` 等
    """
    if prices is None:
        raise ValueError("run_multi_backtest 需要 prices")

    settings = settings or get_settings()

    if fused is not None:
        if not isinstance(fused, pd.Series):
            raise TypeError("fused 须为 pandas.Series（MultiIndex date, symbol）")
        fused_scores = fused.copy()
        extra: Dict[str, Any] = {
            "multi_mode": "pre_fused",
            "multi_weights": None,
            "multi_columns": None,
        }
    elif factors is not None:
        df = _factors_to_dataframe(factors)
        n = df.shape[1]
        if n == 0:
            raise ValueError("factors 无有效列")
        if weights is None:
            w = np.ones(n, dtype=float) / n
        else:
            w = np.asarray(weights, dtype=float).ravel()
            if w.size != n:
                raise ValueError("weights 长度须为 %d（与因子列数一致）" % n)
        fused_scores = _linear_fuse(df, w)
        fused_scores.name = factor_name
        extra = {
            "multi_mode": "linear_weight",
            "multi_weights": [float(x) for x in (w / np.sum(w)).tolist()],
            "multi_columns": list(df.columns),
        }
    else:
        raise ValueError("须提供 ``fused`` 或 ``factors``")

    fused_scores.index = fused_scores.index.set_names(["date", "symbol"])

    nav, meta = run_single_backtest(
        factor_name,
        factor_values=fused_scores,
        prices=prices,
        settings=settings,
        top_k=top_k,
        **kwargs,
    )
    out_meta = dict(meta)
    out_meta.update(extra)
    return nav, out_meta
