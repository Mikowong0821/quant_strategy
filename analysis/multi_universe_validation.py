"""
多股票池验证。

本模块不重新跑回测，而是读取多个股票池已经生成的 output 目录，
汇总策略绩效和因子多头超额，用来判断策略/因子是否只在某一个股票池里有效。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STRATEGY_UNIVERSE_COLUMNS = [
    "universe",
    "strategy",
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
    "avg_turnover",
    "avg_effective_n",
    "output_dir",
]

STRATEGY_ROBUSTNESS_COLUMNS = [
    "strategy",
    "n_universes",
    "avg_final_nav",
    "avg_ann_return",
    "min_ann_return",
    "positive_ann_return_rate",
    "avg_excess_ann_return",
    "min_excess_ann_return",
    "positive_excess_rate",
    "avg_information_ratio",
    "positive_information_ratio_rate",
    "worst_max_drawdown",
    "avg_turnover",
    "avg_effective_n",
    "status",
]

FACTOR_UNIVERSE_COLUMNS = [
    "universe",
    "factor",
    "ann_return",
    "excess_ann_return",
    "tracking_error",
    "information_ratio",
    "n_rebalances",
    "output_dir",
]

FACTOR_ROBUSTNESS_COLUMNS = [
    "factor",
    "n_universes",
    "avg_ann_return",
    "avg_excess_ann_return",
    "min_excess_ann_return",
    "positive_excess_rate",
    "avg_information_ratio",
    "positive_information_ratio_rate",
    "avg_tracking_error",
    "status",
]


def _as_path_map(universe_outputs: Mapping[str, str | Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for name, path in universe_outputs.items():
        universe = str(name).strip()
        if not universe:
            raise ValueError("universe 名称不能为空")
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(p)
        out[universe] = p
    if not out:
        raise ValueError("至少需要一个股票池输出目录")
    return out


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _status_from_rates(
    *,
    n_universes: int,
    positive_rate: float,
    positive_excess_rate: float | None = None,
    avg_excess: float | None = None,
) -> str:
    if n_universes <= 1:
        return "INSUFFICIENT"
    primary = positive_rate
    if positive_excess_rate is not None and np.isfinite(positive_excess_rate):
        primary = min(primary, positive_excess_rate)
    if primary >= 2.0 / 3.0 and (avg_excess is None or not np.isfinite(avg_excess) or avg_excess > 0):
        return "ROBUST"
    if primary >= 0.5:
        return "WATCH"
    return "UNSTABLE"


def collect_strategy_universe_performance(
    universe_outputs: Mapping[str, str | Path],
    *,
    strategies: Sequence[str] | None = None,
) -> pd.DataFrame:
    """读取多个 output/performance_summary.csv，合并为跨股票池策略绩效表。"""
    paths = _as_path_map(universe_outputs)
    selected = set(str(x) for x in strategies) if strategies is not None else None
    frames: list[pd.DataFrame] = []
    for universe, base in paths.items():
        path = base / "performance_summary.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if frame.empty or "strategy" not in frame.columns:
            continue
        frame["strategy"] = frame["strategy"].astype(str)
        if selected is not None:
            frame = frame[frame["strategy"].isin(selected)].copy()
        frame.insert(0, "universe", universe)
        frame["output_dir"] = str(base)
        for col in STRATEGY_UNIVERSE_COLUMNS:
            if col not in frame.columns:
                frame[col] = np.nan
        frames.append(frame[STRATEGY_UNIVERSE_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=STRATEGY_UNIVERSE_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["strategy", "universe"]).reset_index(drop=True)


def summarize_strategy_universe_robustness(strategy_perf: pd.DataFrame) -> pd.DataFrame:
    """按策略汇总跨股票池稳健性。"""
    if strategy_perf.empty:
        return pd.DataFrame(columns=STRATEGY_ROBUSTNESS_COLUMNS)
    rows: list[dict[str, Any]] = []
    for strategy, g in strategy_perf.groupby("strategy", sort=True):
        ann = _numeric(g["ann_return"])
        excess = _numeric(g["excess_ann_return"]) if "excess_ann_return" in g else pd.Series(dtype=float)
        ir = _numeric(g["information_ratio"]) if "information_ratio" in g else pd.Series(dtype=float)
        dd = _numeric(g["max_drawdown"]) if "max_drawdown" in g else pd.Series(dtype=float)
        n = int(g["universe"].nunique())
        positive_ann_rate = float((ann > 0).mean()) if len(ann) else np.nan
        positive_excess_rate = float((excess > 0).mean()) if len(excess.dropna()) else np.nan
        avg_excess = float(excess.mean()) if excess.notna().any() else np.nan
        status = (
            "BENCHMARK"
            if str(strategy) == "BENCH_EQUAL_WEIGHT"
            else _status_from_rates(
                n_universes=n,
                positive_rate=positive_ann_rate,
                positive_excess_rate=positive_excess_rate,
                avg_excess=avg_excess,
            )
        )
        rows.append(
            {
                "strategy": str(strategy),
                "n_universes": n,
                "avg_final_nav": float(_numeric(g["final_nav"]).mean()) if "final_nav" in g else np.nan,
                "avg_ann_return": float(ann.mean()) if ann.notna().any() else np.nan,
                "min_ann_return": float(ann.min()) if ann.notna().any() else np.nan,
                "positive_ann_return_rate": positive_ann_rate,
                "avg_excess_ann_return": avg_excess,
                "min_excess_ann_return": float(excess.min()) if excess.notna().any() else np.nan,
                "positive_excess_rate": positive_excess_rate,
                "avg_information_ratio": float(ir.mean()) if ir.notna().any() else np.nan,
                "positive_information_ratio_rate": float((ir > 0).mean()) if len(ir.dropna()) else np.nan,
                "worst_max_drawdown": float(dd.min()) if dd.notna().any() else np.nan,
                "avg_turnover": float(_numeric(g["avg_turnover"]).mean()) if "avg_turnover" in g else np.nan,
                "avg_effective_n": float(_numeric(g["avg_effective_n"]).mean()) if "avg_effective_n" in g else np.nan,
                "status": status,
            }
        )
    return pd.DataFrame(rows)[STRATEGY_ROBUSTNESS_COLUMNS].sort_values(
        ["status", "avg_excess_ann_return", "avg_ann_return"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def collect_factor_universe_performance(
    universe_outputs: Mapping[str, str | Path],
    *,
    factors: Sequence[str] | None = None,
) -> pd.DataFrame:
    """读取多个 output/factor_diagnostics/long_excess_summary.csv，合并为跨股票池因子表现表。"""
    paths = _as_path_map(universe_outputs)
    selected = set(str(x) for x in factors) if factors is not None else None
    frames: list[pd.DataFrame] = []
    for universe, base in paths.items():
        path = base / "factor_diagnostics" / "long_excess_summary.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if frame.empty or "factor" not in frame.columns:
            continue
        frame["factor"] = frame["factor"].astype(str)
        if selected is not None:
            frame = frame[frame["factor"].isin(selected)].copy()
        frame.insert(0, "universe", universe)
        frame["output_dir"] = str(base)
        for col in FACTOR_UNIVERSE_COLUMNS:
            if col not in frame.columns:
                frame[col] = np.nan
        frames.append(frame[FACTOR_UNIVERSE_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=FACTOR_UNIVERSE_COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values(["factor", "universe"]).reset_index(drop=True)


def summarize_factor_universe_robustness(factor_perf: pd.DataFrame) -> pd.DataFrame:
    """按因子汇总跨股票池稳健性。"""
    if factor_perf.empty:
        return pd.DataFrame(columns=FACTOR_ROBUSTNESS_COLUMNS)
    rows: list[dict[str, Any]] = []
    for factor, g in factor_perf.groupby("factor", sort=True):
        ann = _numeric(g["ann_return"])
        excess = _numeric(g["excess_ann_return"])
        ir = _numeric(g["information_ratio"])
        n = int(g["universe"].nunique())
        positive_excess_rate = float((excess > 0).mean()) if len(excess) else np.nan
        avg_excess = float(excess.mean()) if excess.notna().any() else np.nan
        status = _status_from_rates(
            n_universes=n,
            positive_rate=positive_excess_rate,
            positive_excess_rate=positive_excess_rate,
            avg_excess=avg_excess,
        )
        rows.append(
            {
                "factor": str(factor),
                "n_universes": n,
                "avg_ann_return": float(ann.mean()) if ann.notna().any() else np.nan,
                "avg_excess_ann_return": avg_excess,
                "min_excess_ann_return": float(excess.min()) if excess.notna().any() else np.nan,
                "positive_excess_rate": positive_excess_rate,
                "avg_information_ratio": float(ir.mean()) if ir.notna().any() else np.nan,
                "positive_information_ratio_rate": float((ir > 0).mean()) if len(ir.dropna()) else np.nan,
                "avg_tracking_error": float(_numeric(g["tracking_error"]).mean()) if "tracking_error" in g else np.nan,
                "status": status,
            }
        )
    order = {"ROBUST": 0, "WATCH": 1, "UNSTABLE": 2, "INSUFFICIENT": 3}
    out = pd.DataFrame(rows)
    out["_order"] = out["status"].map(order).fillna(9).astype(int)
    return out.sort_values(
        ["_order", "positive_excess_rate", "avg_excess_ann_return", "factor"],
        ascending=[True, False, False, True],
    ).drop(columns=["_order"])[FACTOR_ROBUSTNESS_COLUMNS].reset_index(drop=True)


def save_multi_universe_validation_outputs(
    output_dir: str | Path,
    *,
    strategy_performance: pd.DataFrame,
    strategy_robustness: pd.DataFrame,
    factor_performance: pd.DataFrame,
    factor_robustness: pd.DataFrame,
) -> dict[str, Path]:
    """保存多股票池验证输出。"""
    base = Path(output_dir).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "strategy_universe_performance": base / "strategy_universe_performance.csv",
        "strategy_universe_robustness": base / "strategy_universe_robustness.csv",
        "factor_universe_performance": base / "factor_universe_performance.csv",
        "factor_universe_robustness": base / "factor_universe_robustness.csv",
    }
    strategy_performance.to_csv(paths["strategy_universe_performance"], index=False)
    strategy_robustness.to_csv(paths["strategy_universe_robustness"], index=False)
    factor_performance.to_csv(paths["factor_universe_performance"], index=False)
    factor_robustness.to_csv(paths["factor_universe_robustness"], index=False)
    return paths
