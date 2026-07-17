#!/usr/bin/env python
"""从已有回测缓存生成参数敏感性报告 CSV。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.parameter_sensitivity import (  # noqa: E402
    build_one_way_parameter_variants,
    run_parameter_sensitivity,
    save_parameter_sensitivity_outputs,
    summarize_parameter_sensitivity,
)
from config import get_settings  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", default="output/cache/prices_wide_close.csv")
    parser.add_argument("--panel", default="output/cache/factor_panel_zscore.csv")
    parser.add_argument("--factor", default="MOMENTUM_60D")
    parser.add_argument("--output-dir", default="output/parameter_sensitivity")
    parser.add_argument("--top-k", type=int, action="append", default=None)
    parser.add_argument("--rebalance-freq", action="append", default=None)
    parser.add_argument("--portfolio-weighting", action="append", default=None)
    parser.add_argument("--max-position-weight", type=float, action="append", default=None)
    parser.add_argument("--max-rebalance-turnover", type=float, action="append", default=None)
    parser.add_argument("--target-volatility", type=float, action="append", default=None)
    parser.add_argument("--min-positions", type=int, action="append", default=None)
    parser.add_argument("--min-positions-exposure", type=float, default=0.5)
    return parser.parse_args()


def _load_prices(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["trade_date"])
    return df.set_index("trade_date").sort_index()


def _load_factor(path: str | Path, factor: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"])
    if factor not in df.columns:
        available = ", ".join([c for c in df.columns if c not in {"date", "symbol"}])
        raise ValueError(f"factor_panel 中不存在因子 {factor}；可用因子: {available}")
    out = df.set_index(["date", "symbol"])[factor].astype(float).sort_index()
    out.name = factor
    return out


def main() -> int:
    args = _parse_args()
    settings = get_settings()
    prices = _load_prices(args.prices)
    factor_values = _load_factor(args.panel, args.factor)

    grid = {
        "top_k": args.top_k or [3, settings.top_k, 8, 10],
        "rebalance_freq": args.rebalance_freq or [settings.rebalance_freq, "QE"],
        "portfolio_weighting": args.portfolio_weighting
        or [settings.portfolio_weighting, "equal", "risk_parity"],
        "max_position_weight": args.max_position_weight
        or [0.25, settings.max_position_weight, 0.60],
        "max_rebalance_turnover": args.max_rebalance_turnover
        or [0.50, 0.75, settings.max_rebalance_turnover],
        "target_volatility": args.target_volatility or [0.0, 0.15, 0.25],
    }
    variants = build_one_way_parameter_variants(settings, grid)
    min_positions_values = args.min_positions or [max(settings.top_k + 2, 8)]
    for value in min_positions_values:
        if value <= 0:
            continue
        variants.append(
            {
                "variant": f"min_positions={value},exposure={args.min_positions_exposure}",
                "changed_parameter": "min_positions_rule",
                "changed_value": f"{value}|{args.min_positions_exposure}",
                "overrides": {
                    "min_positions": int(value),
                    "min_positions_exposure": float(args.min_positions_exposure),
                },
            }
        )
    detail = run_parameter_sensitivity(
        prices=prices,
        factor_values=factor_values,
        base_settings=settings,
        variants=variants,
        factor_name=args.factor,
    )
    summary = summarize_parameter_sensitivity(detail)
    paths = save_parameter_sensitivity_outputs(args.output_dir, detail, summary)

    print("saved:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    if not summary.empty:
        print("\nsummary:")
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
