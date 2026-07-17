"""
参数敏感性分析。

本模块不负责重新构建因子，也不拉取外部数据；它接收已经准备好的价格宽表
和某一列信号，在同一份数据上改变少量策略参数，观察绩效、超额、换手和集中度
是否对参数过度敏感。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.benchmark import equal_weight_benchmark_nav, summarize_excess
from analysis.performance import summarize
from analysis.risk_exposure import summarize_concentration
from analysis.turnover import summarize_turnover
from backtest.backtest_single import run_single_backtest
from config import Settings


PARAMETER_DETAIL_COLUMNS = [
    "variant",
    "changed_parameter",
    "changed_value",
    "factor_name",
    "top_k",
    "rebalance_freq",
    "portfolio_weighting",
    "max_position_weight",
    "max_rebalance_turnover",
    "target_volatility",
    "min_positions",
    "min_positions_exposure",
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
    "max_turnover",
    "avg_effective_n",
    "min_effective_n",
]

PARAMETER_SUMMARY_COLUMNS = [
    "changed_parameter",
    "n_variants",
    "avg_ann_return",
    "min_ann_return",
    "max_ann_return",
    "ann_return_range",
    "avg_excess_ann_return",
    "min_excess_ann_return",
    "positive_excess_rate",
    "avg_information_ratio",
    "worst_max_drawdown",
    "avg_turnover",
    "avg_effective_n",
    "status",
]


def build_one_way_parameter_variants(
    base_settings: Settings,
    parameter_grid: Mapping[str, Sequence[Any]],
) -> list[dict[str, Any]]:
    """
    构造“一次只改一个参数”的实验清单。

    第一行永远是 baseline；后续每行只覆盖一个参数，便于定位敏感来源。
    与 baseline 完全相同的取值会跳过，避免重复实验。
    """
    variants: list[dict[str, Any]] = [
        {
            "variant": "baseline",
            "changed_parameter": "BASELINE",
            "changed_value": "",
            "overrides": {},
        }
    ]
    for param, values in parameter_grid.items():
        base_value = getattr(base_settings, param)
        for value in values:
            if value == base_value:
                continue
            variants.append(
                {
                    "variant": f"{param}={value}",
                    "changed_parameter": str(param),
                    "changed_value": value,
                    "overrides": {str(param): value},
                }
            )
    return variants


def _variant_settings(base_settings: Settings, overrides: Mapping[str, Any]) -> Settings:
    allowed = set(base_settings.__dataclass_fields__)  # type: ignore[attr-defined]
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError("未知 Settings 参数: %s" % ", ".join(unknown))
    return replace(base_settings, persist_run_outputs=False, **dict(overrides))


def _row_for_variant(
    *,
    variant: Mapping[str, Any],
    settings: Settings,
    factor_name: str,
    nav: pd.Series,
    benchmark_nav: pd.Series,
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    stats = summarize(nav, periods=settings.trading_days_per_year)
    excess = summarize_excess(nav, benchmark_nav, periods=settings.trading_days_per_year)
    turnover = summarize_turnover(
        list(meta.get("rebalance_log") or []),
        commission_rate=float(settings.commission_rate),
    )
    concentration = summarize_concentration(list(meta.get("rebalance_log") or []))

    row: dict[str, Any] = {
        "variant": variant.get("variant", ""),
        "changed_parameter": variant.get("changed_parameter", ""),
        "changed_value": variant.get("changed_value", ""),
        "factor_name": factor_name,
        "top_k": int(settings.top_k),
        "rebalance_freq": settings.rebalance_freq,
        "portfolio_weighting": settings.portfolio_weighting,
        "max_position_weight": float(settings.max_position_weight),
        "max_rebalance_turnover": float(settings.max_rebalance_turnover),
        "target_volatility": float(settings.target_volatility),
        "min_positions": int(settings.min_positions),
        "min_positions_exposure": float(settings.min_positions_exposure),
    }
    row.update(stats)
    row.update(excess)
    row.update(turnover)
    row.update(concentration)
    return {col: row.get(col, np.nan) for col in PARAMETER_DETAIL_COLUMNS}


def run_parameter_sensitivity(
    *,
    prices: pd.DataFrame,
    factor_values: pd.Series,
    base_settings: Settings,
    variants: Sequence[Mapping[str, Any]],
    factor_name: str,
) -> pd.DataFrame:
    """对每个参数变体运行一条轻量回测，并返回明细表。"""
    rows: list[dict[str, Any]] = []
    prices = prices.sort_index().sort_index(axis=1).astype(float)
    factor_values = factor_values.sort_index().astype(float)

    for variant in variants:
        settings = _variant_settings(base_settings, variant.get("overrides", {}))
        nav, meta = run_single_backtest(
            factor_name,
            factor_values=factor_values,
            prices=prices,
            settings=settings,
            top_k=settings.top_k,
        )
        benchmark = equal_weight_benchmark_nav(prices, dates=nav.index)
        rows.append(
            _row_for_variant(
                variant=variant,
                settings=settings,
                factor_name=factor_name,
                nav=nav,
                benchmark_nav=benchmark,
                meta=meta,
            )
        )

    if not rows:
        return pd.DataFrame(columns=PARAMETER_DETAIL_COLUMNS)
    return pd.DataFrame(rows, columns=PARAMETER_DETAIL_COLUMNS)


def summarize_parameter_sensitivity(detail: pd.DataFrame) -> pd.DataFrame:
    """按参数汇总敏感性，输出 ROBUST / WATCH / UNSTABLE 状态。"""
    if detail.empty:
        return pd.DataFrame(columns=PARAMETER_SUMMARY_COLUMNS)

    rows: list[dict[str, Any]] = []
    work = detail[detail["changed_parameter"] != "BASELINE"].copy()
    for param, group in work.groupby("changed_parameter", dropna=False):
        ann = group["ann_return"].astype(float)
        excess = group["excess_ann_return"].astype(float)
        ir = group["information_ratio"].astype(float)
        mdd = group["max_drawdown"].astype(float)
        positive_rate = float((excess > 0).mean()) if len(excess) else np.nan
        avg_excess = float(excess.mean()) if len(excess) else np.nan
        min_excess = float(excess.min()) if len(excess) else np.nan
        ann_range = float(ann.max() - ann.min()) if len(ann) else np.nan

        if len(group) == 0:
            status = "INSUFFICIENT"
        elif positive_rate >= 0.75 and avg_excess > 0 and ann_range < 0.5:
            status = "ROBUST"
        elif positive_rate >= 0.5 and avg_excess > 0:
            status = "WATCH"
        else:
            status = "UNSTABLE"

        rows.append(
            {
                "changed_parameter": param,
                "n_variants": int(len(group)),
                "avg_ann_return": float(ann.mean()) if len(ann) else np.nan,
                "min_ann_return": float(ann.min()) if len(ann) else np.nan,
                "max_ann_return": float(ann.max()) if len(ann) else np.nan,
                "ann_return_range": ann_range,
                "avg_excess_ann_return": avg_excess,
                "min_excess_ann_return": min_excess,
                "positive_excess_rate": positive_rate,
                "avg_information_ratio": float(ir.mean()) if len(ir) else np.nan,
                "worst_max_drawdown": float(mdd.min()) if len(mdd) else np.nan,
                "avg_turnover": float(group["avg_turnover"].astype(float).mean()),
                "avg_effective_n": float(group["avg_effective_n"].astype(float).mean()),
                "status": status,
            }
        )

    if not rows:
        return pd.DataFrame(columns=PARAMETER_SUMMARY_COLUMNS)
    return pd.DataFrame(rows, columns=PARAMETER_SUMMARY_COLUMNS).sort_values(
        ["status", "changed_parameter"]
    )


def save_parameter_sensitivity_outputs(
    output_dir: str | Path,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Path]:
    """保存参数敏感性明细和汇总。"""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    detail_path = base / "parameter_sensitivity_detail.csv"
    summary_path = base / "parameter_sensitivity_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    return {"detail": detail_path, "summary": summary_path}
