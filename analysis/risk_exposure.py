"""
风险暴露与持仓集中度分析。

本模块先从 MVP 最容易稳定落地的维度开始：用回测产生的
``meta["rebalance_log"]`` 计算每期目标持仓的集中度。它不改变策略收益，
只回答一个工程风控问题：这条净值背后，到底押在多少只股票、权重是否过度集中。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def _target_weight_map(rec: Mapping[str, Any]) -> dict[str, float]:
    picks = [str(x) for x in list(rec.get("picks") or [])]
    weights = [float(x) for x in list(rec.get("weights") or [])]
    if not picks:
        return {}

    if len(weights) != len(picks):
        weights = [1.0 / len(picks)] * len(picks)

    out: dict[str, float] = {}
    for sym, w in zip(picks, weights):
        if not np.isfinite(w) or w < 0:
            continue
        out[sym] = out.get(sym, 0.0) + float(w)

    total = float(sum(out.values()))
    if total <= 1e-12:
        return {}
    return {k: v / total for k, v in out.items()}


def concentration_frame(rebalance_log: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """
    将调仓日志转为逐期集中度表。

    输出列：
    - hhi：Herfindahl-Hirschman Index，权重平方和；越高越集中。
    - effective_n：1 / hhi，可理解为“等效持仓只数”；越低越集中。
    - top1_weight / top3_weight：最大单票、前三大持仓权重。
    - max_weight：最大持仓权重，与 top1_weight 同义，保留便于读表。
    - n_positions：有效目标持仓数。
    - weighting：当期配权方式标签。
    """
    rows: list[dict[str, Any]] = []
    for rec in sorted(rebalance_log, key=lambda x: pd.Timestamp(x.get("date"))):
        dt = pd.Timestamp(rec.get("date"))
        weights = sorted(_target_weight_map(rec).values(), reverse=True)
        n_pos = len(weights)
        if n_pos == 0:
            hhi = float("nan")
            effective_n = float("nan")
            top1 = float("nan")
            top3 = float("nan")
            max_w = float("nan")
        else:
            arr = np.asarray(weights, dtype=float)
            hhi = float(np.square(arr).sum())
            effective_n = float(1.0 / hhi) if hhi > 1e-12 else float("nan")
            top1 = float(arr[0])
            top3 = float(arr[:3].sum())
            max_w = top1
        rows.append(
            {
                "date": dt,
                "hhi": hhi,
                "effective_n": effective_n,
                "top1_weight": top1,
                "top3_weight": top3,
                "max_weight": max_w,
                "n_positions": n_pos,
                "weighting": rec.get("weighting", ""),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "hhi",
                "effective_n",
                "top1_weight",
                "top3_weight",
                "max_weight",
                "n_positions",
                "weighting",
            ]
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def summarize_concentration(rebalance_log: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """汇总策略持仓集中度指标，便于并入 performance_summary.csv。"""
    frame = concentration_frame(rebalance_log)
    if frame.empty:
        return {
            "avg_effective_n": np.nan,
            "min_effective_n": np.nan,
            "avg_hhi": np.nan,
            "max_hhi": np.nan,
            "avg_top1_weight": np.nan,
            "max_top1_weight": np.nan,
            "avg_top3_weight": np.nan,
            "max_top3_weight": np.nan,
            "avg_n_positions": np.nan,
            "min_n_positions": np.nan,
            "n_concentration_periods": 0,
        }
    return {
        "avg_effective_n": float(frame["effective_n"].mean()),
        "min_effective_n": float(frame["effective_n"].min()),
        "avg_hhi": float(frame["hhi"].mean()),
        "max_hhi": float(frame["hhi"].max()),
        "avg_top1_weight": float(frame["top1_weight"].mean()),
        "max_top1_weight": float(frame["top1_weight"].max()),
        "avg_top3_weight": float(frame["top3_weight"].mean()),
        "max_top3_weight": float(frame["top3_weight"].max()),
        "avg_n_positions": float(frame["n_positions"].mean()),
        "min_n_positions": int(frame["n_positions"].min()),
        "n_concentration_periods": int(len(frame)),
    }


def effective_n_wide(concentration_by_name: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """将多条策略的 effective_n 合并为宽表，便于画图。"""
    series: dict[str, pd.Series] = {}
    for name, frame in concentration_by_name.items():
        if frame.empty or "date" not in frame or "effective_n" not in frame:
            continue
        s = frame.set_index("date")["effective_n"].astype(float)
        s.name = str(name)
        series[str(name)] = s
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()
