"""
牛熊市 / 市场状态分段表现。

市场状态用股票池等权基准定义，而不是用策略自身净值定义，避免“用结果解释结果”。
默认规则：
- BULL：基准过去 lookback 日收益高于 bull_return_threshold；
- BEAR：基准过去 lookback 日收益低于 bear_return_threshold，或当前回撤低于 bear_drawdown_threshold；
- SIDEWAYS：其余交易日。
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.benchmark import summarize_excess
from analysis.performance import summarize


REGIME_DETAIL_COLUMNS = [
    "strategy",
    "regime",
    "n_days",
    "start_date",
    "end_date",
    "final_nav",
    "total_return",
    "ann_return",
    "ann_vol",
    "sharpe",
    "max_drawdown",
    "benchmark_ann_return",
    "excess_ann_return",
    "tracking_error",
    "information_ratio",
]

REGIME_SUMMARY_COLUMNS = [
    "strategy",
    "n_regimes",
    "bull_ann_return",
    "bear_ann_return",
    "sideways_ann_return",
    "bull_excess_ann_return",
    "bear_excess_ann_return",
    "sideways_excess_ann_return",
    "worst_regime_drawdown",
    "positive_excess_regime_rate",
    "status",
]


def build_market_regime_frame(
    benchmark_nav: pd.Series,
    *,
    lookback_days: int = 60,
    bull_return_threshold: float = 0.10,
    bear_return_threshold: float = -0.10,
    bear_drawdown_threshold: float = -0.15,
) -> pd.DataFrame:
    """基于基准净值生成每日市场状态标签。"""
    nav = benchmark_nav.astype(float).dropna().sort_index()
    nav.index = pd.DatetimeIndex(nav.index, name="date")
    if nav.empty:
        return pd.DataFrame(
            columns=["date", "benchmark_nav", "rolling_return", "drawdown", "regime"]
        )

    lookback = max(int(lookback_days), 1)
    rolling_return = nav / nav.shift(lookback) - 1.0
    drawdown = nav / nav.cummax() - 1.0

    regime = pd.Series("SIDEWAYS", index=nav.index, dtype=object)
    bear_mask = (rolling_return <= float(bear_return_threshold)) | (
        drawdown <= float(bear_drawdown_threshold)
    )
    bull_mask = (rolling_return >= float(bull_return_threshold)) & (~bear_mask)
    regime.loc[bear_mask.fillna(False)] = "BEAR"
    regime.loc[bull_mask.fillna(False)] = "BULL"

    frame = pd.DataFrame(
        {
            "date": nav.index,
            "benchmark_nav": nav.values,
            "rolling_return": rolling_return.values,
            "drawdown": drawdown.values,
            "regime": regime.values,
        }
    )
    return frame


def summarize_regime_days(regime_frame: pd.DataFrame) -> pd.DataFrame:
    """统计每类市场状态覆盖了多少交易日，以及起止日期。"""
    if regime_frame.empty:
        return pd.DataFrame(columns=["regime", "n_days", "start_date", "end_date", "day_ratio"])
    rows: list[dict[str, Any]] = []
    total = len(regime_frame)
    for regime, group in regime_frame.groupby("regime", sort=False):
        rows.append(
            {
                "regime": regime,
                "n_days": int(len(group)),
                "start_date": pd.Timestamp(group["date"].min()),
                "end_date": pd.Timestamp(group["date"].max()),
                "day_ratio": float(len(group) / total) if total else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)


def _normalize_segment(series: pd.Series) -> pd.Series:
    s = series.astype(float).dropna()
    if s.empty:
        return s
    first = float(s.iloc[0])
    if not np.isfinite(first) or abs(first) <= 1e-12:
        return pd.Series(dtype=float, name=s.name)
    return s / first


def strategy_regime_performance(
    nav_by_name: Mapping[str, pd.Series],
    benchmark_nav: pd.Series,
    regime_frame: pd.DataFrame,
    *,
    periods: int = 252,
) -> pd.DataFrame:
    """计算每条策略在 BULL / BEAR / SIDEWAYS 下的绩效和超额。"""
    if not nav_by_name or regime_frame.empty:
        return pd.DataFrame(columns=REGIME_DETAIL_COLUMNS)

    regimes = regime_frame.copy()
    regimes["date"] = pd.to_datetime(regimes["date"])
    rows: list[dict[str, Any]] = []
    bench = benchmark_nav.astype(float).dropna().sort_index()
    bench.index = pd.DatetimeIndex(bench.index, name="date")

    for strategy, nav in nav_by_name.items():
        s = nav.astype(float).dropna().sort_index()
        s.index = pd.DatetimeIndex(s.index, name="date")
        for regime, group in regimes.groupby("regime", sort=False):
            idx = pd.DatetimeIndex(group["date"])
            seg_nav = _normalize_segment(s.reindex(idx).ffill().dropna())
            seg_bench = _normalize_segment(bench.reindex(idx).ffill().dropna())
            common = seg_nav.index.intersection(seg_bench.index)
            seg_nav = seg_nav.reindex(common).dropna()
            seg_bench = seg_bench.reindex(common).dropna()
            if len(seg_nav) < 2 or len(seg_bench) < 2:
                continue

            stats = summarize(seg_nav, periods=periods)
            excess = summarize_excess(seg_nav, seg_bench, periods=periods)
            row: dict[str, Any] = {
                "strategy": str(strategy),
                "regime": str(regime),
                "n_days": int(len(seg_nav)),
                "start_date": pd.Timestamp(seg_nav.index.min()),
                "end_date": pd.Timestamp(seg_nav.index.max()),
            }
            row.update(stats)
            row.update(excess)
            rows.append({col: row.get(col, np.nan) for col in REGIME_DETAIL_COLUMNS})

    if not rows:
        return pd.DataFrame(columns=REGIME_DETAIL_COLUMNS)
    return pd.DataFrame(rows, columns=REGIME_DETAIL_COLUMNS).sort_values(
        ["strategy", "regime"]
    )


def summarize_strategy_regime_robustness(detail: pd.DataFrame) -> pd.DataFrame:
    """把每条策略的分段表现压缩成一行稳健性摘要。"""
    if detail.empty:
        return pd.DataFrame(columns=REGIME_SUMMARY_COLUMNS)

    rows: list[dict[str, Any]] = []
    for strategy, group in detail.groupby("strategy", dropna=False):
        by_regime = {str(r["regime"]): r for r in group.to_dict("records")}
        excess = pd.to_numeric(group["excess_ann_return"], errors="coerce")
        dd = pd.to_numeric(group["max_drawdown"], errors="coerce")
        positive_rate = float((excess > 0).mean()) if len(excess.dropna()) else np.nan

        bear_excess = by_regime.get("BEAR", {}).get("excess_ann_return", np.nan)
        bull_excess = by_regime.get("BULL", {}).get("excess_ann_return", np.nan)
        if positive_rate >= 2.0 / 3.0 and (
            not np.isfinite(float(bear_excess)) or float(bear_excess) >= -0.10
        ):
            status = "ROBUST"
        elif positive_rate >= 0.5 or (
            np.isfinite(float(bull_excess)) and float(bull_excess) > 0
        ):
            status = "WATCH"
        else:
            status = "UNSTABLE"

        rows.append(
            {
                "strategy": str(strategy),
                "n_regimes": int(group["regime"].nunique()),
                "bull_ann_return": by_regime.get("BULL", {}).get("ann_return", np.nan),
                "bear_ann_return": by_regime.get("BEAR", {}).get("ann_return", np.nan),
                "sideways_ann_return": by_regime.get("SIDEWAYS", {}).get("ann_return", np.nan),
                "bull_excess_ann_return": by_regime.get("BULL", {}).get(
                    "excess_ann_return", np.nan
                ),
                "bear_excess_ann_return": by_regime.get("BEAR", {}).get(
                    "excess_ann_return", np.nan
                ),
                "sideways_excess_ann_return": by_regime.get("SIDEWAYS", {}).get(
                    "excess_ann_return", np.nan
                ),
                "worst_regime_drawdown": float(dd.min()) if dd.notna().any() else np.nan,
                "positive_excess_regime_rate": positive_rate,
                "status": status,
            }
        )

    return pd.DataFrame(rows, columns=REGIME_SUMMARY_COLUMNS).sort_values(
        ["status", "positive_excess_regime_rate", "strategy"],
        ascending=[True, False, True],
    )


def save_market_regime_outputs(
    output_dir: str | Path,
    regime_frame: pd.DataFrame,
    regime_days: pd.DataFrame,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Path]:
    """保存市场状态标签、状态覆盖、策略分段表现和稳健性摘要。"""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "regime_frame": base / "market_regime_frame.csv",
        "regime_days": base / "market_regime_days.csv",
        "strategy_regime_performance": base / "strategy_regime_performance.csv",
        "strategy_regime_summary": base / "strategy_regime_summary.csv",
    }
    regime_frame.to_csv(paths["regime_frame"], index=False, date_format="%Y-%m-%d")
    regime_days.to_csv(paths["regime_days"], index=False, date_format="%Y-%m-%d")
    detail.to_csv(paths["strategy_regime_performance"], index=False, date_format="%Y-%m-%d")
    summary.to_csv(paths["strategy_regime_summary"], index=False)
    return paths
