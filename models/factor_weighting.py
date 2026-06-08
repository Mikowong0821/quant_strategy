"""
多因子权重评分。

本模块把因子评价层的指标转成一张“因子权重建议表”。它不直接改变当前
FUSED 回测口径，而是先给出可审计的 factor_score / fusion_weight，便于观察
哪些因子应该被提高或降低权重。
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


_COMPONENT_COLUMNS = [
    "score_mean_ic",
    "score_ic_ir",
    "score_positive_rate",
    "score_rolling_mean",
    "score_rolling_positive_rate",
    "score_top_minus_bottom",
    "score_monotonicity",
]


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _minmax_positive(s: pd.Series) -> pd.Series:
    """非负指标做横向 min-max；全 0 或全空则返回 0。"""
    v = s.astype(float).clip(lower=0.0).replace([np.inf, -np.inf], np.nan)
    if v.notna().sum() == 0:
        return pd.Series(0.0, index=s.index)
    mn = float(v.min(skipna=True))
    mx = float(v.max(skipna=True))
    if not np.isfinite(mx) or mx <= 0:
        return pd.Series(0.0, index=s.index)
    if np.isclose(mx, mn):
        return v.fillna(0.0).apply(lambda x: 1.0 if x > 0 else 0.0)
    return ((v - mn) / (mx - mn)).fillna(0.0).clip(lower=0.0, upper=1.0)


def _factor_level_group_metrics(group_return_summary: pd.DataFrame) -> pd.DataFrame:
    if group_return_summary.empty:
        return pd.DataFrame(columns=["factor", "top_minus_bottom_ann", "monotonicity_score"])
    need = {"factor", "group"}
    missing = need - set(group_return_summary.columns)
    if missing:
        raise ValueError("group_return_summary 缺少列: %s" % sorted(missing))
    out = (
        group_return_summary.sort_values(["factor", "group"])
        .groupby("factor", as_index=False)
        .tail(1)
        .copy()
    )
    keep = ["factor", "top_minus_bottom_ann", "monotonicity_score"]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep].reset_index(drop=True)


def _factor_level_rolling_metrics(
    ic_rolling_stability: pd.DataFrame,
    *,
    preferred_window: int | None = None,
) -> pd.DataFrame:
    if ic_rolling_stability.empty:
        return pd.DataFrame(
            columns=[
                "factor",
                "rolling_window_used",
                "rolling_mean_last",
                "rolling_mean_positive_rate",
            ]
        )
    need = {"factor", "window"}
    missing = need - set(ic_rolling_stability.columns)
    if missing:
        raise ValueError("ic_rolling_stability 缺少列: %s" % sorted(missing))
    frame = ic_rolling_stability.copy()
    if preferred_window is not None:
        pref = frame[frame["window"].astype(int) == int(preferred_window)]
        if not pref.empty:
            frame = pref
    out = (
        frame.sort_values(["factor", "window"])
        .groupby("factor", as_index=False)
        .tail(1)
        .rename(columns={"window": "rolling_window_used"})
        .copy()
    )
    keep = ["factor", "rolling_window_used", "rolling_mean_last", "rolling_mean_positive_rate"]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep].reset_index(drop=True)


def build_factor_weight_summary(
    ic_distribution: pd.DataFrame,
    ic_rolling_stability: pd.DataFrame,
    group_return_summary: pd.DataFrame,
    *,
    factors: Iterable[str] | None = None,
    preferred_rolling_window: int | None = None,
) -> pd.DataFrame:
    """
    构造综合因子权重建议表。

    使用指标：
    - mean_ic / ic_ir / positive_rate
    - rolling_mean_last / rolling_mean_positive_rate
    - top_minus_bottom_ann / monotonicity_score

    输出的 `factor_score` 是各归一化组件均值；`fusion_weight` 为 factor_score 归一化。
    若全部得分为 0，则在有效因子间等权。
    """
    if ic_distribution.empty:
        return pd.DataFrame()
    if "factor" not in ic_distribution.columns:
        raise ValueError("ic_distribution 缺少 factor 列")

    base_cols = ["factor", "mean_ic", "ic_ir", "positive_rate"]
    base = ic_distribution.copy()
    for col in base_cols:
        if col not in base.columns:
            base[col] = np.nan
    base = base[base_cols]

    if factors is not None:
        factor_order = [str(x) for x in factors]
        base = base[base["factor"].astype(str).isin(factor_order)].copy()
    else:
        factor_order = list(base["factor"].astype(str))

    roll = _factor_level_rolling_metrics(
        ic_rolling_stability,
        preferred_window=preferred_rolling_window,
    )
    grp = _factor_level_group_metrics(group_return_summary)
    out = base.merge(roll, on="factor", how="left").merge(grp, on="factor", how="left")

    if out.empty:
        return out

    out["raw_mean_ic"] = out["mean_ic"].map(_safe_float).clip(lower=0.0)
    out["raw_ic_ir"] = out["ic_ir"].map(_safe_float).clip(lower=0.0)
    out["raw_positive_rate"] = (out["positive_rate"].map(_safe_float) - 0.5).clip(lower=0.0) * 2.0
    out["raw_rolling_mean"] = out["rolling_mean_last"].map(_safe_float).clip(lower=0.0)
    out["raw_rolling_positive_rate"] = (
        out["rolling_mean_positive_rate"].map(_safe_float) - 0.5
    ).clip(lower=0.0) * 2.0
    out["raw_top_minus_bottom"] = out["top_minus_bottom_ann"].map(_safe_float).clip(lower=0.0)
    out["raw_monotonicity"] = out["monotonicity_score"].map(_safe_float).clip(lower=0.0, upper=1.0)

    raw_to_score = {
        "raw_mean_ic": "score_mean_ic",
        "raw_ic_ir": "score_ic_ir",
        "raw_positive_rate": "score_positive_rate",
        "raw_rolling_mean": "score_rolling_mean",
        "raw_rolling_positive_rate": "score_rolling_positive_rate",
        "raw_top_minus_bottom": "score_top_minus_bottom",
        "raw_monotonicity": "score_monotonicity",
    }
    for raw_col, score_col in raw_to_score.items():
        out[score_col] = _minmax_positive(out[raw_col])

    out["factor_score"] = out[_COMPONENT_COLUMNS].mean(axis=1)
    score_sum = float(out["factor_score"].sum())
    if np.isfinite(score_sum) and score_sum > 1e-12:
        out["fusion_weight"] = out["factor_score"] / score_sum
        out["weighting_fallback"] = False
    else:
        out["fusion_weight"] = 1.0 / len(out)
        out["weighting_fallback"] = True

    if factor_order:
        order = {name: i for i, name in enumerate(factor_order)}
        out["_order"] = out["factor"].map(order).fillna(len(order)).astype(int)
        out = out.sort_values(["_order", "factor"]).drop(columns=["_order"]).reset_index(drop=True)
    else:
        out = out.sort_values("factor").reset_index(drop=True)

    first = [
        "factor",
        "factor_score",
        "fusion_weight",
        "weighting_fallback",
        "mean_ic",
        "ic_ir",
        "positive_rate",
        "rolling_window_used",
        "rolling_mean_last",
        "rolling_mean_positive_rate",
        "top_minus_bottom_ann",
        "monotonicity_score",
    ]
    rest = [c for c in out.columns if c not in first]
    return out[first + rest]
