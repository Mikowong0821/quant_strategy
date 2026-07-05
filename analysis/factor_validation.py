"""
样本外验证与因子失效监控。

本模块把已有的 IC、Top-K 多头超额、分组收益诊断按时间切成训练段和验证段，
用于回答：训练期看起来有效的因子，在样本外是否仍然有效。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.factor_diagnostics import batch_factor_group_returns, batch_factor_long_excess
from analysis.ic import daily_ic_spearman, ic_distribution_summary
from config import Settings


VALIDATION_COLUMNS = [
    "factor",
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "train_ic_mean",
    "validation_ic_mean",
    "ic_mean_delta",
    "train_positive_rate",
    "validation_positive_rate",
    "positive_rate_delta",
    "train_excess_ann_return",
    "validation_excess_ann_return",
    "excess_ann_delta",
    "train_information_ratio",
    "validation_information_ratio",
    "train_top_minus_bottom_ann",
    "validation_top_minus_bottom_ann",
    "top_minus_bottom_delta",
    "train_monotonicity_score",
    "validation_monotonicity_score",
    "monotonicity_delta",
    "train_n_days",
    "validation_n_days",
]

MONITOR_COLUMNS = [
    "factor",
    "status",
    "severity",
    "reasons",
    "validation_ic_mean",
    "validation_positive_rate",
    "validation_excess_ann_return",
    "validation_top_minus_bottom_ann",
    "validation_monotonicity_score",
    "ic_mean_delta",
    "excess_ann_delta",
    "top_minus_bottom_delta",
]


def split_train_validation_dates(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    train_ratio: float = 0.5,
) -> tuple[pd.Index, pd.Index]:
    """按日期交集切分训练段和样本外验证段。"""
    if not isinstance(panel.index, pd.MultiIndex):
        raise TypeError("panel 须为 MultiIndex(date, symbol)")
    panel_dates = pd.Index(panel.index.get_level_values("date").unique()).sort_values()
    dates = pd.Index(pd.to_datetime(prices.index)).intersection(panel_dates).sort_values()
    if len(dates) < 4:
        raise ValueError("可用于样本外验证的日期太少")
    ratio = min(max(float(train_ratio), 0.2), 0.8)
    split_pos = int(len(dates) * ratio)
    split_pos = min(max(split_pos, 2), len(dates) - 1)
    return dates[:split_pos], dates[split_pos:]


def _panel_on_dates(panel: pd.DataFrame, dates: pd.Index) -> pd.DataFrame:
    mask = panel.index.get_level_values("date").isin(pd.Index(dates))
    return panel.loc[mask]


def _factor_level_group_summary(group_summary: pd.DataFrame) -> pd.DataFrame:
    if group_summary.empty:
        return pd.DataFrame(
            columns=["factor", "top_minus_bottom_ann", "monotonicity_score"]
        )
    out = (
        group_summary.sort_values(["factor", "group"])
        .groupby("factor", as_index=False)
        .tail(1)
        .copy()
    )
    keep = ["factor", "top_minus_bottom_ann", "monotonicity_score"]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    return out[keep].reset_index(drop=True)


def _segment_metrics(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    factors: list[str],
    settings: Settings,
    segment: str,
) -> pd.DataFrame:
    ic_by_name: dict[str, pd.Series] = {}
    for factor in factors:
        if factor not in panel.columns:
            continue
        ser = panel[factor]
        if ser.notna().sum() == 0:
            continue
        try:
            ic_by_name[factor] = daily_ic_spearman(
                ser,
                prices,
                forward_days=settings.ic_forward_days,
            )
        except Exception:
            continue

    ic_summary = ic_distribution_summary(ic_by_name)
    if ic_summary.empty:
        ic_summary = pd.DataFrame(columns=["factor", "mean_ic", "positive_rate", "n_days"])
    ic_summary = ic_summary.rename(
        columns={
            "mean_ic": f"{segment}_ic_mean",
            "positive_rate": f"{segment}_positive_rate",
            "n_days": f"{segment}_n_days",
        }
    )

    long_summary, _ = batch_factor_long_excess(
        panel,
        prices,
        factors=factors,
        top_k=settings.top_k,
        rebalance_freq=settings.rebalance_freq,
        price_col=settings.price_col,
        periods=settings.trading_days_per_year,
    )
    if long_summary.empty:
        long_summary = pd.DataFrame(
            columns=["factor", "excess_ann_return", "information_ratio"]
        )
    long_summary = long_summary.rename(
        columns={
            "excess_ann_return": f"{segment}_excess_ann_return",
            "information_ratio": f"{segment}_information_ratio",
        }
    )

    _, group_summary = batch_factor_group_returns(
        panel,
        prices,
        factors=factors,
        group_count=settings.factor_group_count,
        rebalance_freq=settings.rebalance_freq,
        price_col=settings.price_col,
        trading_days_per_year=settings.trading_days_per_year,
    )
    group_level = _factor_level_group_summary(group_summary).rename(
        columns={
            "top_minus_bottom_ann": f"{segment}_top_minus_bottom_ann",
            "monotonicity_score": f"{segment}_monotonicity_score",
        }
    )

    out = pd.DataFrame({"factor": factors})
    out = out.merge(
        ic_summary[
            [
                c
                for c in ["factor", f"{segment}_ic_mean", f"{segment}_positive_rate", f"{segment}_n_days"]
                if c in ic_summary.columns
            ]
        ],
        on="factor",
        how="left",
    )
    out = out.merge(
        long_summary[
            [
                c
                for c in ["factor", f"{segment}_excess_ann_return", f"{segment}_information_ratio"]
                if c in long_summary.columns
            ]
        ],
        on="factor",
        how="left",
    )
    out = out.merge(group_level, on="factor", how="left")
    return out


def build_out_of_sample_validation(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    settings: Settings,
    *,
    factors: list[str] | None = None,
    train_ratio: float | None = None,
) -> pd.DataFrame:
    """生成训练段与样本外验证段的因子评价对照表。"""
    if not isinstance(panel.index, pd.MultiIndex):
        raise TypeError("panel 须为 MultiIndex(date, symbol)")
    factor_list = [str(x) for x in (factors or list(panel.columns)) if str(x) in panel.columns]
    factor_list = [x for x in factor_list if panel[x].notna().sum() > 0]
    if not factor_list:
        return pd.DataFrame(columns=VALIDATION_COLUMNS)

    ratio = float(train_ratio if train_ratio is not None else settings.factor_weight_train_ratio)
    train_dates, validation_dates = split_train_validation_dates(panel, prices, train_ratio=ratio)
    train_panel = _panel_on_dates(panel[factor_list], train_dates)
    validation_panel = _panel_on_dates(panel[factor_list], validation_dates)
    train_prices = prices.loc[pd.Index(prices.index).isin(train_dates)]
    validation_prices = prices.loc[pd.Index(prices.index).isin(validation_dates)]

    train = _segment_metrics(train_panel, train_prices, factors=factor_list, settings=settings, segment="train")
    validation = _segment_metrics(
        validation_panel,
        validation_prices,
        factors=factor_list,
        settings=settings,
        segment="validation",
    )
    out = train.merge(validation, on="factor", how="outer")
    out.insert(1, "train_start", pd.Timestamp(train_dates[0]).strftime("%Y-%m-%d"))
    out.insert(2, "train_end", pd.Timestamp(train_dates[-1]).strftime("%Y-%m-%d"))
    out.insert(3, "validation_start", pd.Timestamp(validation_dates[0]).strftime("%Y-%m-%d"))
    out.insert(4, "validation_end", pd.Timestamp(validation_dates[-1]).strftime("%Y-%m-%d"))

    pairs = [
        ("ic_mean", "train_ic_mean", "validation_ic_mean"),
        ("positive_rate", "train_positive_rate", "validation_positive_rate"),
        ("excess_ann", "train_excess_ann_return", "validation_excess_ann_return"),
        ("top_minus_bottom", "train_top_minus_bottom_ann", "validation_top_minus_bottom_ann"),
        ("monotonicity", "train_monotonicity_score", "validation_monotonicity_score"),
    ]
    for prefix, train_col, val_col in pairs:
        if train_col not in out.columns:
            out[train_col] = np.nan
        if val_col not in out.columns:
            out[val_col] = np.nan
        out[f"{prefix}_delta"] = out[val_col].astype(float) - out[train_col].astype(float)

    for col in VALIDATION_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[VALIDATION_COLUMNS].reset_index(drop=True)


def build_factor_decay_monitor(
    validation: pd.DataFrame,
    *,
    min_validation_ic: float = 0.0,
    min_positive_rate: float = 0.5,
    min_excess_ann_return: float = 0.0,
    min_top_minus_bottom_ann: float = 0.0,
    min_monotonicity: float = 0.5,
) -> pd.DataFrame:
    """
    根据样本外验证结果生成失效监控状态。

    status:
    - OK：样本外主要指标仍然为正；
    - WATCH：有指标转弱，但未全面失效；
    - DEGRADED：训练期为正、验证期明显转弱；
    - FAILED：IC、多头超额和 Top-Bottom 同时为负。
    """
    if validation.empty:
        return pd.DataFrame(columns=MONITOR_COLUMNS)

    rows: list[dict[str, Any]] = []
    for rec in validation.to_dict("records"):
        factor = str(rec.get("factor", ""))
        val_ic = float(rec.get("validation_ic_mean", np.nan))
        val_pos = float(rec.get("validation_positive_rate", np.nan))
        val_excess = float(rec.get("validation_excess_ann_return", np.nan))
        val_tb = float(rec.get("validation_top_minus_bottom_ann", np.nan))
        val_mono = float(rec.get("validation_monotonicity_score", np.nan))
        ic_delta = float(rec.get("ic_mean_delta", np.nan))
        excess_delta = float(rec.get("excess_ann_delta", np.nan))
        tb_delta = float(rec.get("top_minus_bottom_delta", np.nan))

        reasons: list[str] = []
        if np.isfinite(val_ic) and val_ic < min_validation_ic:
            reasons.append("validation_ic_below_threshold")
        if np.isfinite(val_pos) and val_pos < min_positive_rate:
            reasons.append("positive_rate_below_threshold")
        if np.isfinite(val_excess) and val_excess < min_excess_ann_return:
            reasons.append("long_excess_below_threshold")
        if np.isfinite(val_tb) and val_tb < min_top_minus_bottom_ann:
            reasons.append("top_bottom_below_threshold")
        if np.isfinite(val_mono) and val_mono < min_monotonicity:
            reasons.append("monotonicity_below_threshold")
        if np.isfinite(ic_delta) and ic_delta < -0.02:
            reasons.append("ic_deteriorated")
        if np.isfinite(excess_delta) and excess_delta < -0.20:
            reasons.append("long_excess_deteriorated")
        if np.isfinite(tb_delta) and tb_delta < -0.20:
            reasons.append("top_bottom_deteriorated")

        negative_core = sum(
            [
                np.isfinite(val_ic) and val_ic < min_validation_ic,
                np.isfinite(val_excess) and val_excess < min_excess_ann_return,
                np.isfinite(val_tb) and val_tb < min_top_minus_bottom_ann,
            ]
        )
        if negative_core >= 3:
            status, severity = "FAILED", 3
        elif len(reasons) >= 4 or (
            np.isfinite(excess_delta) and excess_delta < -0.30 and np.isfinite(tb_delta) and tb_delta < -0.30
        ):
            status, severity = "DEGRADED", 2
        elif reasons:
            status, severity = "WATCH", 1
        else:
            status, severity = "OK", 0

        rows.append(
            {
                "factor": factor,
                "status": status,
                "severity": severity,
                "reasons": ";".join(reasons),
                "validation_ic_mean": val_ic,
                "validation_positive_rate": val_pos,
                "validation_excess_ann_return": val_excess,
                "validation_top_minus_bottom_ann": val_tb,
                "validation_monotonicity_score": val_mono,
                "ic_mean_delta": ic_delta,
                "excess_ann_delta": excess_delta,
                "top_minus_bottom_delta": tb_delta,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["severity", "factor"], ascending=[False, True]).reset_index(drop=True)
    return out[MONITOR_COLUMNS]


def save_factor_validation_outputs(
    settings: Settings,
    validation: pd.DataFrame,
    monitor: pd.DataFrame,
) -> dict[str, Path]:
    """写入 output/factor_validation/ 下的样本外验证和失效监控表。"""
    base = settings.output_dir / "factor_validation"
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "out_of_sample_validation": base / "out_of_sample_validation.csv",
        "factor_decay_monitor": base / "factor_decay_monitor.csv",
    }
    validation.to_csv(paths["out_of_sample_validation"], index=False)
    monitor.to_csv(paths["factor_decay_monitor"], index=False)
    return paths
