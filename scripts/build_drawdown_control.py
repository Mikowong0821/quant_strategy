"""生成账户级回撤止损与降仓控制检查表。"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import get_settings
from live.account_state import load_account_state
from live.daily_paper_cli import DEFAULT_STRATEGY, load_latest_prices
from live.drawdown_control import (
    build_current_account_snapshot,
    default_drawdown_rules,
    evaluate_drawdown_control,
    load_account_snapshots,
    load_drawdown_rules,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成回撤止损与降仓控制检查表")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, help="策略名")
    parser.add_argument("--trade-date", default=None, help="检查日期；默认使用价格缓存最新日期")
    parser.add_argument("--prices", type=Path, default=None, help="价格宽表 CSV；默认 output/cache/prices_wide_close.csv")
    parser.add_argument("--rules", type=Path, default=None, help="回撤规则 CSV；未提供时使用默认规则")
    parser.add_argument("--output-dir", type=Path, default=None, help="输出目录；默认 output/drawdown_control/<strategy>")
    parser.add_argument("--write-template", action="store_true", help="同时输出默认规则模板")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    settings = get_settings()
    price_path = args.prices if args.prices is not None else settings.output_dir / "cache" / "prices_wide_close.csv"
    requested = pd.Timestamp(args.trade_date) if args.trade_date is not None else None
    price_date, latest_prices = load_latest_prices(price_path, trade_date=requested)
    trade_date = requested if requested is not None else price_date

    cash, positions = load_account_state(
        settings,
        strategy=args.strategy,
        default_cash=settings.paper_initial_cash,
    )
    control = evaluate_drawdown_control(
        load_drawdown_rules(str(args.rules) if args.rules is not None else None),
        load_account_snapshots(settings, strategy=args.strategy),
        build_current_account_snapshot(
            cash=cash,
            positions=positions,
            latest_prices=latest_prices,
        ),
        pd.Series(dtype=float),
        trade_date=trade_date,
    )

    safe_strategy = str(args.strategy).replace("/", "_")
    out_dir = args.output_dir if args.output_dir is not None else settings.output_dir / "drawdown_control" / safe_strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = pd.Timestamp(trade_date).strftime("%Y%m%d")
    control_path = out_dir / ("drawdown_control_%s.csv" % tag)
    control.to_csv(control_path, index=False)
    print("wrote %s" % control_path)

    if args.write_template:
        rules_path = out_dir / "drawdown_rules_template.csv"
        default_drawdown_rules().to_csv(rules_path, index=False)
        print("wrote %s" % rules_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
