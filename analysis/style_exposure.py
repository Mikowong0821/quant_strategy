"""
风格层暴露与贡献分析。

复合因子给出的是“风格分数”，本模块从每期持仓与权重反推组合暴露，
再把暴露与下一持有期收益做轻量关联，帮助解释收益来自哪类风格倾斜。
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


STYLE_EXPOSURE_COLUMNS = [
    "strategy",
    "date",
    "style",
    "weighted_exposure",
    "abs_weighted_exposure",
    "score_coverage",
    "gross_weight",
    "n_positions",
    "n_scored_positions",
]

STYLE_EXPOSURE_SUMMARY_COLUMNS = [
    "strategy",
    "style",
    "avg_exposure",
    "latest_exposure",
    "avg_abs_exposure",
    "positive_rate",
    "avg_score_coverage",
    "n_periods",
]

STYLE_RETURN_LINK_COLUMNS = [
    "strategy",
    "style",
    "exposure_next_return_corr",
    "avg_next_return_when_positive",
    "avg_next_return_when_nonpositive",
    "positive_exposure_hit_rate",
    "n_periods",
]


def _ensure_style_scores(style_scores: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(style_scores.index, pd.MultiIndex) or style_scores.index.nlevels != 2:
        raise TypeError("style_scores 须为二级 MultiIndex(date, symbol)")
    out = style_scores.copy()
    out.index = out.index.set_names(["date", "symbol"])
    out = out.sort_index()
    out.columns = out.columns.astype(str)
    return out


def _normalize_weights(weights: list[float], n: int) -> np.ndarray:
    arr = np.asarray([float(x) for x in weights[:n]], dtype=float)
    if len(arr) < n:
        arr = np.pad(arr, (0, n - len(arr)), constant_values=np.nan)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    gross = float(np.abs(arr).sum())
    if gross <= 1e-12 and n > 0:
        arr = np.full(n, 1.0 / n, dtype=float)
    return arr


def style_exposure_frame(
    rebalance_log: list[dict],
    style_scores: pd.DataFrame,
    *,
    strategy: str = "",
) -> pd.DataFrame:
    """
    计算单个策略逐调仓期的风格暴露。

    weighted_exposure 是持仓权重与风格 z-score 的加权平均。若风格分数为横截面
    标准化后的分数，正值代表组合相对股票池偏向该风格高分股票。
    """
    scores = _ensure_style_scores(style_scores)
    rows: list[dict[str, object]] = []

    for rec in rebalance_log:
        dt = pd.Timestamp(rec.get("date"))
        picks = [str(x) for x in list(rec.get("picks") or [])]
        if not picks:
            continue
        weights = _normalize_weights(list(rec.get("weights") or []), len(picks))
        gross_weight = float(np.abs(weights).sum())
        if gross_weight <= 1e-12:
            continue

        for style in scores.columns:
            values: list[float] = []
            used_weights: list[float] = []
            for sym, weight in zip(picks, weights):
                key = (dt, sym)
                if key not in scores.index:
                    continue
                score = scores.at[key, style]
                if pd.isna(score) or not np.isfinite(float(score)):
                    continue
                values.append(float(score))
                used_weights.append(float(weight))

            if not values:
                rows.append(
                    {
                        "strategy": str(strategy),
                        "date": dt,
                        "style": str(style),
                        "weighted_exposure": np.nan,
                        "abs_weighted_exposure": np.nan,
                        "score_coverage": 0.0,
                        "gross_weight": gross_weight,
                        "n_positions": len(picks),
                        "n_scored_positions": 0,
                    }
                )
                continue

            w = np.asarray(used_weights, dtype=float)
            v = np.asarray(values, dtype=float)
            coverage = float(np.abs(w).sum() / gross_weight) if gross_weight > 0 else 0.0
            exposure = float(np.nansum(w * v) / gross_weight)
            rows.append(
                {
                    "strategy": str(strategy),
                    "date": dt,
                    "style": str(style),
                    "weighted_exposure": exposure,
                    "abs_weighted_exposure": abs(exposure),
                    "score_coverage": coverage,
                    "gross_weight": gross_weight,
                    "n_positions": len(picks),
                    "n_scored_positions": len(values),
                }
            )

    if not rows:
        return pd.DataFrame(columns=STYLE_EXPOSURE_COLUMNS)
    out = pd.DataFrame(rows)
    return out[STYLE_EXPOSURE_COLUMNS].sort_values(["strategy", "date", "style"]).reset_index(drop=True)


def batch_style_exposure(
    meta_by_name: Mapping[str, Mapping],
    style_scores: pd.DataFrame,
    *,
    strategies: list[str] | None = None,
) -> pd.DataFrame:
    selected = set(str(x) for x in strategies) if strategies is not None else None
    frames: list[pd.DataFrame] = []
    for name, meta in meta_by_name.items():
        if selected is not None and str(name) not in selected:
            continue
        log = list(meta.get("rebalance_log") or [])
        frame = style_exposure_frame(log, style_scores, strategy=str(name))
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=STYLE_EXPOSURE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def summarize_style_exposure(exposure: pd.DataFrame) -> pd.DataFrame:
    if exposure.empty:
        return pd.DataFrame(columns=STYLE_EXPOSURE_SUMMARY_COLUMNS)
    df = exposure.copy()
    df["date"] = pd.to_datetime(df["date"])
    rows: list[dict[str, object]] = []
    for (strategy, style), g in df.groupby(["strategy", "style"], sort=True):
        valid = g.dropna(subset=["weighted_exposure"]).sort_values("date")
        if valid.empty:
            rows.append(
                {
                    "strategy": strategy,
                    "style": style,
                    "avg_exposure": np.nan,
                    "latest_exposure": np.nan,
                    "avg_abs_exposure": np.nan,
                    "positive_rate": np.nan,
                    "avg_score_coverage": float(g["score_coverage"].mean()) if "score_coverage" in g else np.nan,
                    "n_periods": 0,
                }
            )
            continue
        rows.append(
            {
                "strategy": strategy,
                "style": style,
                "avg_exposure": float(valid["weighted_exposure"].mean()),
                "latest_exposure": float(valid.iloc[-1]["weighted_exposure"]),
                "avg_abs_exposure": float(valid["abs_weighted_exposure"].mean()),
                "positive_rate": float((valid["weighted_exposure"] > 0).mean()),
                "avg_score_coverage": float(valid["score_coverage"].mean()),
                "n_periods": int(len(valid)),
            }
        )
    return pd.DataFrame(rows)[STYLE_EXPOSURE_SUMMARY_COLUMNS]


def _period_return(nav: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    ser = nav.sort_index().dropna()
    if ser.empty:
        return np.nan
    start_pos = ser.index.searchsorted(start, side="left")
    end_pos = ser.index.searchsorted(end, side="right") - 1
    if start_pos < 0 or end_pos < 0 or start_pos >= len(ser) or end_pos >= len(ser):
        return np.nan
    v0 = float(ser.iloc[start_pos])
    v1 = float(ser.iloc[end_pos])
    if not np.isfinite(v0) or not np.isfinite(v1) or abs(v0) <= 1e-12:
        return np.nan
    return v1 / v0 - 1.0


def style_exposure_return_link(
    exposure: pd.DataFrame,
    nav_by_strategy: Mapping[str, pd.Series],
) -> pd.DataFrame:
    """
    把每期风格暴露与下一持有期收益关联。

    这不是严格业绩归因，只是回答一个工程问题：当组合更偏某个风格时，
    下一期策略收益是否更容易为正。
    """
    if exposure.empty:
        return pd.DataFrame(columns=STYLE_RETURN_LINK_COLUMNS)
    df = exposure.dropna(subset=["weighted_exposure"]).copy()
    if df.empty:
        return pd.DataFrame(columns=STYLE_RETURN_LINK_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])

    linked_rows: list[dict[str, object]] = []
    for strategy, sg in df.groupby("strategy", sort=True):
        nav = nav_by_strategy.get(str(strategy))
        if nav is None or len(nav) < 2:
            continue
        dates = sorted(pd.Timestamp(x) for x in sg["date"].dropna().unique())
        if not dates:
            continue
        nav_last = pd.Timestamp(nav.dropna().index.max())
        next_dates = {d: (dates[i + 1] if i + 1 < len(dates) else nav_last) for i, d in enumerate(dates)}
        returns = {d: _period_return(nav, d, next_dates[d]) for d in dates}
        tmp = sg.copy()
        tmp["next_period_return"] = tmp["date"].map(returns)
        linked_rows.append(tmp)

    if not linked_rows:
        return pd.DataFrame(columns=STYLE_RETURN_LINK_COLUMNS)
    linked = pd.concat(linked_rows, ignore_index=True).dropna(
        subset=["weighted_exposure", "next_period_return"]
    )
    rows: list[dict[str, object]] = []
    for (strategy, style), g in linked.groupby(["strategy", "style"], sort=True):
        if len(g) < 2:
            corr = np.nan
        else:
            corr = float(g["weighted_exposure"].corr(g["next_period_return"]))
        pos = g[g["weighted_exposure"] > 0]
        nonpos = g[g["weighted_exposure"] <= 0]
        rows.append(
            {
                "strategy": strategy,
                "style": style,
                "exposure_next_return_corr": corr,
                "avg_next_return_when_positive": float(pos["next_period_return"].mean()) if not pos.empty else np.nan,
                "avg_next_return_when_nonpositive": (
                    float(nonpos["next_period_return"].mean()) if not nonpos.empty else np.nan
                ),
                "positive_exposure_hit_rate": (
                    float((pos["next_period_return"] > 0).mean()) if not pos.empty else np.nan
                ),
                "n_periods": int(len(g)),
            }
        )
    return pd.DataFrame(rows)[STYLE_RETURN_LINK_COLUMNS]
