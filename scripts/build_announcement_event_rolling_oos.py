#!/usr/bin/env python3
"""Run rolling out-of-sample validation for ANNOUNCEMENT_EVENT_SCORE."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.factor_validation import (
    build_rolling_out_of_sample_validation,
    summarize_rolling_out_of_sample_validation,
)
from analysis.plotting import _pyplot_zh
from backtest.backtest_utils import long_to_wide
from config import get_settings
from factors.factor_events import (
    ANNOUNCEMENT_EVENT_SCORE,
    calc_announcement_event_score,
    load_announcement_events,
)
from factors.preprocess import preprocess_factor_panel
from live.data_feed import load_prices_from_csv
from main import _attach_industry_to_long_df, _industry_series_from_long_df


def _parse_universe(value: str) -> dict[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--universe 需要格式 name=prices|stock_pool|events|fina")
    name, payload = value.split("=", 1)
    parts = payload.split("|")
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError("--universe 需要格式 name=prices|stock_pool|events|fina")
    return {
        "name": name.strip(),
        "prices": parts[0].strip(),
        "stock_pool": parts[1].strip(),
        "events": parts[2].strip(),
        "fina": parts[3].strip() if len(parts) == 4 else "",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="公告事件因子滚动样本外验证")
    parser.add_argument(
        "--universe",
        action="append",
        type=_parse_universe,
        required=True,
        help="股票池，格式 name=prices|stock_pool|events|fina，可重复传入",
    )
    parser.add_argument("--start", default="2025-01-01", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default="2026-06-23", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--train-days", type=int, default=120, help="滚动训练窗口交易日数")
    parser.add_argument("--validation-days", type=int, default=40, help="滚动验证窗口交易日数")
    parser.add_argument("--step-days", type=int, default=40, help="滚动步长交易日数")
    parser.add_argument("--min-validation-days", type=int, default=20, help="最小验证窗口交易日数")
    parser.add_argument(
        "--output-dir",
        default="output/announcement_event_rolling_oos",
        help="输出目录",
    )
    return parser.parse_args()


def _filter_long_prices(long_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    out = long_df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    mask = (out["trade_date"] >= pd.Timestamp(start)) & (out["trade_date"] <= pd.Timestamp(end))
    return out.loc[mask].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _build_event_panel(
    universe: dict[str, str],
    *,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    settings = get_settings()
    settings = replace(
        settings,
        stock_pool_path=Path(universe["stock_pool"]),
        tushare_price_cache_path=Path(universe["prices"]),
        fina_indicator_cache_path=Path(universe["fina"]) if universe.get("fina") else None,
        announcement_event_path=Path(universe["events"]),
        backtest_start=start,
        backtest_end=end,
        persist_run_outputs=False,
    )
    long_df = load_prices_from_csv(Path(universe["prices"]))
    long_df = _filter_long_prices(long_df, start, end)
    long_df = _attach_industry_to_long_df(long_df, settings)
    prices = long_to_wide(long_df, settings.price_col)
    events = load_announcement_events(Path(universe["events"]))
    event_score = calc_announcement_event_score(
        events,
        long_df,
        effective_days=int(settings.announcement_event_effective_days),
    )
    panel = pd.DataFrame({ANNOUNCEMENT_EVENT_SCORE: event_score})
    industry_ser = _industry_series_from_long_df(long_df, panel.index, settings)
    by_industry = bool(settings.factor_standardize_by_industry and industry_ser is not None)
    panel_z = preprocess_factor_panel(
        panel,
        industry=industry_ser,
        industry_col=settings.industry_col,
        by_industry=by_industry,
        min_industry_count=settings.factor_industry_min_count,
    )
    return panel_z, prices, settings


def _event_coverage(universe: str, panel: pd.DataFrame) -> dict[str, Any]:
    ser = panel[ANNOUNCEMENT_EVENT_SCORE]
    total = int(ser.shape[0])
    nonzero = int((ser.fillna(0.0).abs() > 1e-12).sum())
    return {
        "universe": universe,
        "rows": total,
        "zscore_nonzero_rows": nonzero,
        "zscore_nonzero_coverage": nonzero / total if total else 0.0,
    }


def _run_universe(
    universe: dict[str, str],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    name = str(universe["name"])
    panel, prices, settings = _build_event_panel(universe, start=args.start, end=args.end)
    rolling = build_rolling_out_of_sample_validation(
        panel,
        prices,
        settings,
        factors=[ANNOUNCEMENT_EVENT_SCORE],
        train_days=args.train_days,
        validation_days=args.validation_days,
        step_days=args.step_days,
        min_validation_days=args.min_validation_days,
    )
    summary = summarize_rolling_out_of_sample_validation(rolling)
    if not rolling.empty:
        rolling.insert(0, "universe", name)
    if not summary.empty:
        summary.insert(0, "universe", name)

    uni_dir = output_dir / name
    uni_dir.mkdir(parents=True, exist_ok=True)
    rolling.to_csv(uni_dir / "rolling_out_of_sample_validation.csv", index=False)
    summary.to_csv(uni_dir / "rolling_out_of_sample_summary.csv", index=False)
    return rolling, summary, _event_coverage(name, panel)


def _plot_metric(
    rolling: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if rolling.empty or value_col not in rolling.columns:
        return
    plot_df = rolling.pivot(index="window_id", columns="universe", values=value_col).sort_index()
    if plot_df.empty:
        return
    plt = _pyplot_zh(output_path)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for col in plot_df.columns:
        ax.plot(plot_df.index, plot_df[col], marker="o", linewidth=1.4, label=str(col))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("滚动窗口")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rolling_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    for universe in args.universe:
        rolling, summary, coverage = _run_universe(universe, args, output_dir)
        if not rolling.empty:
            rolling_parts.append(rolling)
        if not summary.empty:
            summary_parts.append(summary)
        coverage_rows.append(coverage)

    rolling_all = pd.concat(rolling_parts, ignore_index=True) if rolling_parts else pd.DataFrame()
    summary_all = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    coverage = pd.DataFrame(coverage_rows)
    rolling_all.to_csv(output_dir / "rolling_out_of_sample_validation.csv", index=False)
    summary_all.to_csv(output_dir / "rolling_out_of_sample_summary.csv", index=False)
    coverage.to_csv(output_dir / "event_factor_coverage.csv", index=False)

    _plot_metric(
        rolling_all,
        value_col="validation_ic_mean",
        title="公告因子滚动样本外 IC",
        ylabel="validation IC mean",
        output_path=output_dir / "rolling_oos_ic.png",
    )
    _plot_metric(
        rolling_all,
        value_col="validation_excess_ann_return",
        title="公告因子滚动样本外多头超额",
        ylabel="validation excess ann return",
        output_path=output_dir / "rolling_oos_excess.png",
    )
    _plot_metric(
        rolling_all,
        value_col="validation_top_minus_bottom_ann",
        title="公告因子滚动样本外分组多空差",
        ylabel="validation top-bottom ann return",
        output_path=output_dir / "rolling_oos_top_bottom.png",
    )

    print("rolling_validation=%s" % (output_dir / "rolling_out_of_sample_validation.csv"))
    print("rolling_summary=%s" % (output_dir / "rolling_out_of_sample_summary.csv"))
    print("coverage=%s" % (output_dir / "event_factor_coverage.csv"))
    if not summary_all.empty:
        print(summary_all.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
