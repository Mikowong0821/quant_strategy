#!/usr/bin/env python3
"""Summarize announcement-event backtest comparisons across stock universes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.plotting import _pyplot_zh


def _parse_universe(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--universe 需要格式 name=output_dir")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("股票池名称不能为空")
    return name, Path(path).expanduser()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="公告事件因子多股票池验证汇总")
    parser.add_argument(
        "--universe",
        action="append",
        type=_parse_universe,
        required=True,
        help="股票池输出目录，格式 name=output_dir，可重复传入",
    )
    parser.add_argument(
        "--output-dir",
        default="output/announcement_event_multi_universe",
        help="汇总输出目录",
    )
    return parser.parse_args()


def _load_performance(universe: str, output_dir: Path) -> pd.DataFrame:
    path = output_dir / "performance_summary.csv"
    if not path.is_file():
        raise FileNotFoundError("缺少绩效文件: %s" % path)
    frame = pd.read_csv(path)
    frame.insert(0, "universe", universe)
    return frame


def _load_validation(universe: str, output_dir: Path) -> pd.DataFrame:
    path = output_dir / "announcement_event_factor_validation.csv"
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame.insert(0, "universe", universe)
    return frame


def _mode_from_strategy(strategy: str) -> str:
    value = str(strategy)
    if value.startswith("ROLLING_"):
        return "ROLLING"
    if value.startswith("EQUAL_"):
        return "EQUAL"
    return value.split("_", 1)[0]


def _has_event(strategy: str) -> bool:
    return str(strategy).endswith("_WITH_EVENT")


def _incremental_effect(perf: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for universe, uni_df in perf.groupby("universe", sort=False):
        uni = uni_df.copy()
        uni["mode"] = uni["strategy"].map(_mode_from_strategy)
        uni["has_event"] = uni["strategy"].map(_has_event)
        for mode, mode_df in uni.groupby("mode", sort=False):
            if set(mode_df["has_event"]) != {False, True}:
                continue
            no_event = mode_df.loc[~mode_df["has_event"]].iloc[0]
            with_event = mode_df.loc[mode_df["has_event"]].iloc[0]
            rows.append(
                {
                    "universe": universe,
                    "mode": mode,
                    "no_event_strategy": no_event["strategy"],
                    "with_event_strategy": with_event["strategy"],
                    "no_event_final_nav": float(no_event["final_nav"]),
                    "with_event_final_nav": float(with_event["final_nav"]),
                    "delta_final_nav": float(with_event["final_nav"] - no_event["final_nav"]),
                    "no_event_total_return": float(no_event["total_return"]),
                    "with_event_total_return": float(with_event["total_return"]),
                    "delta_total_return": float(with_event["total_return"] - no_event["total_return"]),
                    "no_event_ann_return": float(no_event["ann_return"]),
                    "with_event_ann_return": float(with_event["ann_return"]),
                    "delta_ann_return": float(with_event["ann_return"] - no_event["ann_return"]),
                    "no_event_max_drawdown": float(no_event["max_drawdown"]),
                    "with_event_max_drawdown": float(with_event["max_drawdown"]),
                    "delta_max_drawdown": float(with_event["max_drawdown"] - no_event["max_drawdown"]),
                    "no_event_information_ratio": float(no_event["information_ratio"]),
                    "with_event_information_ratio": float(with_event["information_ratio"]),
                    "delta_information_ratio": float(
                        with_event["information_ratio"] - no_event["information_ratio"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _plot_incremental_effect(delta: pd.DataFrame, output_dir: Path) -> Path | None:
    if delta.empty:
        return None
    plot_df = delta.pivot(index="universe", columns="mode", values="delta_total_return").sort_index()
    if plot_df.empty:
        return None
    save_path = output_dir / "announcement_event_delta_return.png"
    plt = _pyplot_zh(save_path)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    plot_df.plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("公告事件因子多股票池增量收益")
    ax.set_xlabel("股票池")
    ax.set_ylabel("加入公告因子后的总收益差")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    universe_outputs = dict(args.universe)
    perf = pd.concat(
        [_load_performance(name, path) for name, path in universe_outputs.items()],
        ignore_index=True,
    )
    validations = [
        _load_validation(name, path)
        for name, path in universe_outputs.items()
    ]
    validation = pd.concat([x for x in validations if not x.empty], ignore_index=True)
    delta = _incremental_effect(perf)

    perf.to_csv(output_dir / "multi_universe_performance_summary.csv", index=False)
    validation.to_csv(output_dir / "multi_universe_event_factor_validation.csv", index=False)
    delta.to_csv(output_dir / "multi_universe_incremental_effect.csv", index=False)
    chart = _plot_incremental_effect(delta, output_dir)

    print("multi_universe_performance=%s" % (output_dir / "multi_universe_performance_summary.csv"))
    print("multi_universe_validation=%s" % (output_dir / "multi_universe_event_factor_validation.csv"))
    print("multi_universe_incremental=%s" % (output_dir / "multi_universe_incremental_effect.csv"))
    if chart is not None:
        print("delta_chart=%s" % chart)
    print(delta.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
