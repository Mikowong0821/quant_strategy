#!/usr/bin/env python3
"""Analyze announcement event factors by event type across stock universes."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.factor_diagnostics import batch_factor_group_returns, batch_factor_long_excess
from analysis.ic import daily_ic_spearman, summarize_ic
from analysis.plotting import _pyplot_zh
from backtest.backtest_utils import long_to_wide
from config import get_settings
from factors.factor_events import (
    ANNOUNCEMENT_EVENT_TYPE_PREFIX,
    EVENT_TYPE_RULES,
    calc_announcement_event_type_scores,
    classify_announcement_event,
    load_announcement_events,
)
from factors.preprocess import preprocess_factor_panel
from live.data_feed import load_prices_from_csv
from main import _attach_industry_to_long_df, _industry_series_from_long_df


DEFAULT_CATEGORIES = tuple(EVENT_TYPE_RULES.keys()) + ("OTHER",)
POSITIVE_EVENT_GROUPS = {
    "BUYBACK",
    "HOLDER_INCREASE",
    "PERFORMANCE_POSITIVE",
    "DIVIDEND",
    "CONTRACT_PROJECT",
}
NEGATIVE_EVENT_GROUPS = {
    "HOLDER_REDUCTION",
    "INQUIRY_PENALTY",
    "PERFORMANCE_NEGATIVE",
    "PLEDGE_FREEZE",
    "LITIGATION",
}
BROAD_EVENT_GROUPS = {"OTHER", "GOVERNANCE", "REFINANCE_MA"}


def _parse_universe(value: str) -> dict[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--universe 需要格式 name=prices|stock_pool|events")
    name, payload = value.split("=", 1)
    parts = payload.split("|")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--universe 需要格式 name=prices|stock_pool|events")
    return {
        "name": name.strip(),
        "prices": parts[0].strip(),
        "stock_pool": parts[1].strip(),
        "events": parts[2].strip(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="公告事件类型分层因子诊断")
    parser.add_argument(
        "--universe",
        action="append",
        type=_parse_universe,
        required=True,
        help="股票池，格式 name=prices|stock_pool|events，可重复传入",
    )
    parser.add_argument("--start", default="2025-01-01", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default="2026-06-23", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--top-k", type=int, default=None, help="Top-K 多头数量；默认使用 Settings.top_k")
    parser.add_argument(
        "--output-dir",
        default="output/announcement_event_type_analysis",
        help="输出目录",
    )
    return parser.parse_args()


def _filter_long_prices(long_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    out = long_df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    mask = (out["trade_date"] >= pd.Timestamp(start)) & (out["trade_date"] <= pd.Timestamp(end))
    return out.loc[mask].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _event_count_by_group(events: pd.DataFrame, universe: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["universe", "event_group", "event_count", "event_symbols"])
    frame = events.copy()
    frame["event_group"] = [
        classify_announcement_event(event_type, title)
        for event_type, title in zip(frame["event_type"], frame["title"], strict=True)
    ]
    rows: list[dict[str, Any]] = []
    for event_group, sub in frame.groupby("event_group"):
        rows.append(
            {
                "universe": universe,
                "event_group": event_group,
                "event_count": int(sub.shape[0]),
                "event_symbols": int(sub["symbol"].nunique()),
                "avg_event_score": float(pd.to_numeric(sub["event_score"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["universe", "event_group"]).reset_index(drop=True)


def _coverage_rows(universe: str, raw_panel: pd.DataFrame, z_panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for factor in raw_panel.columns:
        raw = raw_panel[factor]
        z = z_panel[factor] if factor in z_panel.columns else pd.Series(index=raw.index, dtype=float)
        total = int(raw.shape[0])
        raw_nonzero = raw.fillna(0.0).abs() > 1e-12
        z_nonzero = z.fillna(0.0).abs() > 1e-12
        event_group = str(factor).replace(ANNOUNCEMENT_EVENT_TYPE_PREFIX, "", 1)
        rows.append(
            {
                "universe": universe,
                "factor": factor,
                "event_group": event_group,
                "rows": total,
                "raw_nonzero_rows": int(raw_nonzero.sum()),
                "raw_nonzero_coverage": float(raw_nonzero.mean()) if total else 0.0,
                "raw_nonzero_symbols": int(raw[raw_nonzero].index.get_level_values("symbol").nunique())
                if total
                else 0,
                "zscore_nonzero_rows": int(z_nonzero.sum()),
                "zscore_nonzero_coverage": float(z_nonzero.mean()) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _ic_rows(universe: str, panel_z: pd.DataFrame, prices: pd.DataFrame, settings: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for factor in panel_z.columns:
        ser = panel_z[factor]
        if ser.notna().sum() == 0 or ser.fillna(0.0).abs().sum() <= 1e-12:
            continue
        try:
            ic = daily_ic_spearman(ser, prices, forward_days=settings.ic_forward_days)
            stats = summarize_ic(ic)
        except Exception:
            stats = {}
        event_group = str(factor).replace(ANNOUNCEMENT_EVENT_TYPE_PREFIX, "", 1)
        rows.append({"universe": universe, "factor": factor, "event_group": event_group, **stats})
    return pd.DataFrame(rows)


def _prepare_universe(
    universe: dict[str, str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    settings = get_settings()
    settings = replace(
        settings,
        stock_pool_path=Path(universe["stock_pool"]),
        tushare_price_cache_path=Path(universe["prices"]),
        announcement_event_path=Path(universe["events"]),
        backtest_start=args.start,
        backtest_end=args.end,
        persist_run_outputs=False,
    )
    long_df = load_prices_from_csv(Path(universe["prices"]))
    long_df = _filter_long_prices(long_df, args.start, args.end)
    long_df = _attach_industry_to_long_df(long_df, settings)
    prices = long_to_wide(long_df, settings.price_col)
    events = load_announcement_events(Path(universe["events"]))
    raw_panel = calc_announcement_event_type_scores(
        events,
        long_df,
        effective_days=int(settings.announcement_event_effective_days),
        categories=DEFAULT_CATEGORIES,
    )
    industry_ser = _industry_series_from_long_df(long_df, raw_panel.index, settings)
    by_industry = bool(settings.factor_standardize_by_industry and industry_ser is not None)
    z_panel = preprocess_factor_panel(
        raw_panel,
        industry=industry_ser,
        industry_col=settings.industry_col,
        by_industry=by_industry,
        min_industry_count=settings.factor_industry_min_count,
    )
    return raw_panel, z_panel, prices, events, settings


def _normalize_factor_table(universe: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out.insert(0, "universe", universe)
    if "factor" in out.columns and "event_group" not in out.columns:
        out.insert(
            2,
            "event_group",
            out["factor"].astype(str).str.replace(ANNOUNCEMENT_EVENT_TYPE_PREFIX, "", regex=False),
        )
    return out


def _run_universe(universe: dict[str, str], args: argparse.Namespace, output_dir: Path) -> dict[str, pd.DataFrame]:
    name = str(universe["name"])
    raw_panel, z_panel, prices, events, settings = _prepare_universe(universe, args)
    top_k = int(args.top_k or settings.top_k)
    coverage = _coverage_rows(name, raw_panel, z_panel)
    counts = _event_count_by_group(events, name)
    valid_factors = [
        col
        for col in raw_panel.columns
        if raw_panel[col].fillna(0.0).abs().sum() > 1e-12
        and z_panel[col].fillna(0.0).abs().sum() > 1e-12
    ]
    ic = _ic_rows(name, z_panel, prices, settings)
    long_summary, _ = batch_factor_long_excess(
        z_panel,
        prices,
        factors=valid_factors,
        top_k=top_k,
        rebalance_freq=settings.rebalance_freq,
        price_col=settings.price_col,
        periods=settings.trading_days_per_year,
    )
    long_summary = _normalize_factor_table(name, long_summary)
    group_detail, group_summary = batch_factor_group_returns(
        z_panel,
        prices,
        factors=valid_factors,
        group_count=settings.factor_group_count,
        rebalance_freq=settings.rebalance_freq,
        price_col=settings.price_col,
        trading_days_per_year=settings.trading_days_per_year,
    )
    group_detail = _normalize_factor_table(name, group_detail)
    group_summary = _normalize_factor_table(name, group_summary)

    uni_dir = output_dir / name
    uni_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(uni_dir / "type_event_counts.csv", index=False)
    coverage.to_csv(uni_dir / "type_factor_coverage.csv", index=False)
    ic.to_csv(uni_dir / "type_factor_ic_summary.csv", index=False)
    long_summary.to_csv(uni_dir / "type_factor_long_excess.csv", index=False)
    group_detail.to_csv(uni_dir / "type_factor_group_detail.csv", index=False)
    group_summary.to_csv(uni_dir / "type_factor_group_summary.csv", index=False)
    return {
        "counts": counts,
        "coverage": coverage,
        "ic": ic,
        "long": long_summary,
        "group": group_summary,
    }


def _combined_decision_table(
    counts: pd.DataFrame,
    coverage: pd.DataFrame,
    ic: pd.DataFrame,
    long_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
) -> pd.DataFrame:
    if coverage.empty:
        return pd.DataFrame()
    top_rows = pd.DataFrame()
    if not group_summary.empty and "group" in group_summary.columns:
        top_rows = group_summary.sort_values("group").groupby(["universe", "factor"], as_index=False).tail(1)
        top_rows = top_rows[
            [
                c
                for c in [
                    "universe",
                    "factor",
                    "top_minus_bottom_ann",
                    "top_minus_bottom_hit_rate",
                    "monotonicity_score",
                ]
                if c in top_rows.columns
            ]
        ]
    out = coverage.merge(
        counts,
        on=["universe", "event_group"],
        how="left",
    )
    if not ic.empty:
        out = out.merge(
            ic[
                [
                    c
                    for c in ["universe", "factor", "mean_ic", "ic_ir", "hit_rate", "n_days"]
                    if c in ic.columns
                ]
            ],
            on=["universe", "factor"],
            how="left",
        )
    if not long_summary.empty:
        out = out.merge(
            long_summary[
                [
                    c
                    for c in [
                        "universe",
                        "factor",
                        "ann_return",
                        "excess_ann_return",
                        "information_ratio",
                        "max_drawdown",
                    ]
                    if c in long_summary.columns
                ]
            ],
            on=["universe", "factor"],
            how="left",
        )
    if not top_rows.empty:
        out = out.merge(top_rows, on=["universe", "factor"], how="left")
    out["event_count"] = out["event_count"].fillna(0).astype(int)
    out["event_symbols"] = out["event_symbols"].fillna(0).astype(int)
    out["suggestion"] = out.apply(_suggestion, axis=1)
    sort_cols = ["universe", "suggestion", "event_count", "raw_nonzero_coverage"]
    return out.sort_values(sort_cols, ascending=[True, True, False, False]).reset_index(drop=True)


def _suggestion(row: pd.Series) -> str:
    event_group = str(row.get("event_group", ""))
    coverage = float(row.get("raw_nonzero_coverage", 0.0) or 0.0)
    event_count = int(row.get("event_count", 0) or 0)
    mean_ic = float(row.get("mean_ic", np.nan))
    excess = float(row.get("excess_ann_return", np.nan))
    top_bottom = float(row.get("top_minus_bottom_ann", np.nan))
    if coverage <= 0.001 or event_count < 5:
        return "INSUFFICIENT"
    if event_group in BROAD_EVENT_GROUPS:
        return "OBSERVE"
    if event_group in NEGATIVE_EVENT_GROUPS:
        if np.isfinite(mean_ic) and np.isfinite(excess) and mean_ic > 0 and excess > 0:
            return "RISK_FILTER_WATCH"
        if np.isfinite(mean_ic) and mean_ic < 0 or np.isfinite(excess) and excess < 0 or np.isfinite(top_bottom) and top_bottom < 0:
            return "RISK_WATCH"
        return "OBSERVE"
    if np.isfinite(mean_ic) and np.isfinite(excess) and mean_ic > 0 and excess > 0:
        return "ALPHA_WATCH"
    if event_group in POSITIVE_EVENT_GROUPS and (
        np.isfinite(mean_ic) and mean_ic < 0 or np.isfinite(excess) and excess < 0
    ):
        return "UNIVERSE_DEPENDENT"
    return "OBSERVE"


def _plot_bar(frame: pd.DataFrame, *, value_col: str, title: str, ylabel: str, output_path: Path) -> None:
    if frame.empty or value_col not in frame.columns:
        return
    plot_df = frame.pivot_table(index="event_group", columns="universe", values=value_col, aggfunc="mean")
    plot_df = plot_df.dropna(how="all")
    if plot_df.empty:
        return
    order = plot_df.abs().mean(axis=1).sort_values(ascending=False).head(10).index
    plot_df = plot_df.loc[order]
    plt = _pyplot_zh(output_path)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    plot_df.plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("公告类型")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collected: dict[str, list[pd.DataFrame]] = {
        "counts": [],
        "coverage": [],
        "ic": [],
        "long": [],
        "group": [],
    }
    for universe in args.universe:
        result = _run_universe(universe, args, output_dir)
        for key, frame in result.items():
            if not frame.empty:
                collected[key].append(frame)

    counts = pd.concat(collected["counts"], ignore_index=True) if collected["counts"] else pd.DataFrame()
    coverage = pd.concat(collected["coverage"], ignore_index=True) if collected["coverage"] else pd.DataFrame()
    ic = pd.concat(collected["ic"], ignore_index=True) if collected["ic"] else pd.DataFrame()
    long_summary = pd.concat(collected["long"], ignore_index=True) if collected["long"] else pd.DataFrame()
    group_summary = pd.concat(collected["group"], ignore_index=True) if collected["group"] else pd.DataFrame()
    decision = _combined_decision_table(counts, coverage, ic, long_summary, group_summary)

    counts.to_csv(output_dir / "type_event_counts.csv", index=False)
    coverage.to_csv(output_dir / "type_factor_coverage.csv", index=False)
    ic.to_csv(output_dir / "type_factor_ic_summary.csv", index=False)
    long_summary.to_csv(output_dir / "type_factor_long_excess.csv", index=False)
    group_summary.to_csv(output_dir / "type_factor_group_summary.csv", index=False)
    decision.to_csv(output_dir / "type_factor_decision_table.csv", index=False)

    _plot_bar(decision, value_col="mean_ic", title="公告类型分层因子 IC", ylabel="mean IC", output_path=output_dir / "type_factor_ic.png")
    _plot_bar(
        decision,
        value_col="excess_ann_return",
        title="公告类型分层因子多头超额",
        ylabel="excess ann return",
        output_path=output_dir / "type_factor_excess.png",
    )
    _plot_bar(
        decision,
        value_col="top_minus_bottom_ann",
        title="公告类型分层因子 Top-Bottom",
        ylabel="top-bottom ann return",
        output_path=output_dir / "type_factor_top_bottom.png",
    )

    print("decision_table=%s" % (output_dir / "type_factor_decision_table.csv"))
    if not decision.empty:
        cols = [
            "universe",
            "event_group",
            "event_count",
            "raw_nonzero_coverage",
            "mean_ic",
            "excess_ann_return",
            "top_minus_bottom_ann",
            "monotonicity_score",
            "suggestion",
        ]
        print(decision[[c for c in cols if c in decision.columns]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
