"""
换手率与交易成本分析。

输入来自回测返回的 ``meta["rebalance_log"]``。这里的 turnover 定义为：
本期目标权重相对上期目标权重的绝对变化和，即近似「成交金额 / 组合净值」。
初次建仓从现金到满仓，turnover 通常约为 1.0。
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


def turnover_frame(
    rebalance_log: Sequence[Mapping[str, Any]],
    *,
    commission_rate: float = 0.0,
) -> pd.DataFrame:
    """
    将调仓日志转为逐期换手表。

    输出列：
    - turnover：目标权重绝对变化和，近似成交金额 / 组合净值。
    - estimated_cost：按 ``commission_rate`` 估算的单边交易成本占净值比例。
    - n_positions：本期目标持仓数。
    - weighting：本期配权方式标签。
    """
    rows: list[dict[str, Any]] = []
    prev: dict[str, float] = {}

    for rec in sorted(rebalance_log, key=lambda x: pd.Timestamp(x.get("date"))):
        dt = pd.Timestamp(rec.get("date"))
        cur = _target_weight_map(rec)
        symbols = set(prev) | set(cur)
        turnover = float(sum(abs(cur.get(sym, 0.0) - prev.get(sym, 0.0)) for sym in symbols))
        rows.append(
            {
                "date": dt,
                "turnover": turnover,
                "estimated_cost": turnover * float(commission_rate),
                "n_positions": len(cur),
                "weighting": rec.get("weighting", ""),
            }
        )
        prev = cur

    if not rows:
        return pd.DataFrame(
            columns=["date", "turnover", "estimated_cost", "n_positions", "weighting"]
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def summarize_turnover(
    rebalance_log: Sequence[Mapping[str, Any]],
    *,
    commission_rate: float = 0.0,
) -> dict[str, Any]:
    """汇总换手与预估交易成本指标。"""
    frame = turnover_frame(rebalance_log, commission_rate=commission_rate)
    if frame.empty:
        return {
            "avg_turnover": np.nan,
            "max_turnover": np.nan,
            "total_turnover": np.nan,
            "estimated_total_cost": np.nan,
            "n_turnover_periods": 0,
        }
    return {
        "avg_turnover": float(frame["turnover"].mean()),
        "max_turnover": float(frame["turnover"].max()),
        "total_turnover": float(frame["turnover"].sum()),
        "estimated_total_cost": float(frame["estimated_cost"].sum()),
        "n_turnover_periods": int(len(frame)),
    }


def turnover_wide(turnover_by_name: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """将多条策略的逐期换手表合并为宽表，便于画图。"""
    series: dict[str, pd.Series] = {}
    for name, frame in turnover_by_name.items():
        if frame.empty or "date" not in frame or "turnover" not in frame:
            continue
        s = frame.set_index("date")["turnover"].astype(float)
        s.name = str(name)
        series[str(name)] = s
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()
