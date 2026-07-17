"""
因子权重稳定性与漂移监控。

滚动综合权重会在每个调仓日前重新评估因子，权重变化本身也需要被审计：
某个因子是否突然被大幅提高，组合是否被单一因子主导，权重变化是否过快。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


FACTOR_WEIGHT_STABILITY_COLUMNS = [
    "factor",
    "n_periods",
    "avg_weight",
    "latest_weight",
    "min_weight",
    "max_weight",
    "weight_range",
    "weight_std",
    "avg_abs_change",
    "max_abs_change",
    "total_abs_change",
    "active_rate",
    "zero_weight_rate",
    "stability_score",
    "status",
]

FACTOR_WEIGHT_DRIFT_COLUMNS = [
    "date",
    "factor",
    "event_type",
    "severity",
    "previous_weight",
    "current_weight",
    "weight_change",
    "abs_weight_change",
    "reason",
]

PORTFOLIO_WEIGHT_DRIFT_COLUMNS = [
    "date",
    "n_factors",
    "n_active_factors",
    "dominant_factor",
    "dominant_weight",
    "effective_factor_n",
    "sum_abs_weight_change",
    "weight_turnover",
    "reason",
]


def _prepare_weight_log(weight_log: pd.DataFrame, *, weight_col: str = "final_weight") -> pd.DataFrame:
    if weight_log.empty:
        return pd.DataFrame(columns=["date", "factor", weight_col, "reason"])
    required = {"date", "factor", weight_col}
    missing = required - set(weight_log.columns)
    if missing:
        raise ValueError("rolling_factor_weight_log 缺少列: %s" % sorted(missing))
    df = weight_log.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["factor"] = df["factor"].astype(str)
    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
    if "reason" not in df.columns:
        df["reason"] = ""
    return df.sort_values(["factor", "date"]).reset_index(drop=True)


def factor_weight_stability_summary(
    weight_log: pd.DataFrame,
    *,
    weight_col: str = "final_weight",
    active_threshold: float = 1e-6,
    watch_avg_change: float = 0.08,
    watch_max_change: float = 0.18,
) -> pd.DataFrame:
    """
    汇总每个因子的滚动权重稳定性。

    stability_score 越接近 1，说明权重越平稳；状态只用于提示，不直接剔除因子。
    """
    df = _prepare_weight_log(weight_log, weight_col=weight_col)
    if df.empty:
        return pd.DataFrame(columns=FACTOR_WEIGHT_STABILITY_COLUMNS)

    rows: list[dict[str, object]] = []
    for factor, g in df.groupby("factor", sort=True):
        weights = g[weight_col].astype(float).replace([np.inf, -np.inf], np.nan)
        valid = weights.dropna()
        if valid.empty:
            rows.append(
                {
                    "factor": factor,
                    "n_periods": 0,
                    "avg_weight": np.nan,
                    "latest_weight": np.nan,
                    "min_weight": np.nan,
                    "max_weight": np.nan,
                    "weight_range": np.nan,
                    "weight_std": np.nan,
                    "avg_abs_change": np.nan,
                    "max_abs_change": np.nan,
                    "total_abs_change": np.nan,
                    "active_rate": np.nan,
                    "zero_weight_rate": np.nan,
                    "stability_score": np.nan,
                    "status": "NO_DATA",
                }
            )
            continue

        changes = valid.diff().abs().dropna()
        avg_abs_change = float(changes.mean()) if not changes.empty else 0.0
        max_abs_change = float(changes.max()) if not changes.empty else 0.0
        weight_std = float(valid.std(ddof=0)) if len(valid) > 1 else 0.0
        stability_score = 1.0 - min(1.0, (avg_abs_change + weight_std) / 0.5)
        status = "PASS"
        if avg_abs_change >= watch_avg_change or max_abs_change >= watch_max_change:
            status = "WATCH"

        rows.append(
            {
                "factor": factor,
                "n_periods": int(len(valid)),
                "avg_weight": float(valid.mean()),
                "latest_weight": float(valid.iloc[-1]),
                "min_weight": float(valid.min()),
                "max_weight": float(valid.max()),
                "weight_range": float(valid.max() - valid.min()),
                "weight_std": weight_std,
                "avg_abs_change": avg_abs_change,
                "max_abs_change": max_abs_change,
                "total_abs_change": float(changes.sum()) if not changes.empty else 0.0,
                "active_rate": float((valid > active_threshold).mean()),
                "zero_weight_rate": float((valid <= active_threshold).mean()),
                "stability_score": float(stability_score),
                "status": status,
            }
        )
    return pd.DataFrame(rows)[FACTOR_WEIGHT_STABILITY_COLUMNS]


def factor_weight_drift_events(
    weight_log: pd.DataFrame,
    *,
    weight_col: str = "final_weight",
    active_threshold: float = 1e-6,
    change_threshold: float = 0.10,
    high_change_threshold: float = 0.20,
) -> pd.DataFrame:
    """提取因子权重跳变、进入活跃、退出活跃等漂移事件。"""
    df = _prepare_weight_log(weight_log, weight_col=weight_col)
    if df.empty:
        return pd.DataFrame(columns=FACTOR_WEIGHT_DRIFT_COLUMNS)

    rows: list[dict[str, object]] = []
    for factor, g in df.groupby("factor", sort=True):
        ordered = g.sort_values("date").reset_index(drop=True)
        prev = ordered[weight_col].shift(1)
        for i, rec in ordered.iloc[1:].iterrows():
            prev_w = float(prev.iloc[i])
            curr_w = float(rec[weight_col])
            if not np.isfinite(prev_w) or not np.isfinite(curr_w):
                continue
            delta = curr_w - prev_w
            abs_delta = abs(delta)
            prev_active = prev_w > active_threshold
            curr_active = curr_w > active_threshold
            event_type = ""
            if not prev_active and curr_active:
                event_type = "entered_active"
            elif prev_active and not curr_active:
                event_type = "exited_active"
            elif abs_delta >= change_threshold:
                event_type = "weight_jump" if delta > 0 else "weight_drop"
            if not event_type:
                continue
            severity = "HIGH" if abs_delta >= high_change_threshold else "WATCH"
            rows.append(
                {
                    "date": pd.Timestamp(rec["date"]),
                    "factor": factor,
                    "event_type": event_type,
                    "severity": severity,
                    "previous_weight": prev_w,
                    "current_weight": curr_w,
                    "weight_change": float(delta),
                    "abs_weight_change": float(abs_delta),
                    "reason": str(rec.get("reason", "")),
                }
            )

    if not rows:
        return pd.DataFrame(columns=FACTOR_WEIGHT_DRIFT_COLUMNS)
    return (
        pd.DataFrame(rows)[FACTOR_WEIGHT_DRIFT_COLUMNS]
        .sort_values(["date", "severity", "abs_weight_change"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def factor_weight_portfolio_drift(
    weight_log: pd.DataFrame,
    *,
    weight_col: str = "final_weight",
    active_threshold: float = 1e-6,
) -> pd.DataFrame:
    """从组合层面观察每期因子权重是否过度集中或变化过快。"""
    df = _prepare_weight_log(weight_log, weight_col=weight_col)
    if df.empty:
        return pd.DataFrame(columns=PORTFOLIO_WEIGHT_DRIFT_COLUMNS)

    pivot = (
        df.pivot_table(index="date", columns="factor", values=weight_col, aggfunc="last")
        .sort_index()
        .fillna(0.0)
    )
    reason_by_date = df.groupby("date")["reason"].agg(lambda x: "|".join(sorted(set(map(str, x)))))
    diff = pivot.diff().abs().fillna(0.0)
    rows: list[dict[str, object]] = []
    for dt, weights in pivot.iterrows():
        arr = weights.astype(float).clip(lower=0.0)
        total = float(arr.sum())
        if total > 1e-12:
            normalized = arr / total
        else:
            normalized = pd.Series(0.0, index=arr.index)
        dominant_factor = str(normalized.idxmax()) if len(normalized) else ""
        dominant_weight = float(normalized.max()) if len(normalized) else np.nan
        effective_n = float(1.0 / np.square(normalized.to_numpy(dtype=float)).sum()) if total > 1e-12 else np.nan
        sum_abs_change = float(diff.loc[dt].sum())
        rows.append(
            {
                "date": pd.Timestamp(dt),
                "n_factors": int(len(normalized)),
                "n_active_factors": int((normalized > active_threshold).sum()),
                "dominant_factor": dominant_factor,
                "dominant_weight": dominant_weight,
                "effective_factor_n": effective_n,
                "sum_abs_weight_change": sum_abs_change,
                "weight_turnover": 0.5 * sum_abs_change,
                "reason": str(reason_by_date.get(dt, "")),
            }
        )
    return pd.DataFrame(rows)[PORTFOLIO_WEIGHT_DRIFT_COLUMNS]
