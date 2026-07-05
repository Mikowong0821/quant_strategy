#!/usr/bin/env python3
"""真实成交回填与执行偏差分析入口。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from live.daily_paper_cli import DEFAULT_STRATEGY
from live.execution_feedback import build_execution_feedback, save_execution_feedback


def _default_manual_confirm_path(strategy: str, trade_date: str | None) -> Path:
    settings = get_settings()
    safe = str(strategy).replace("/", "_")
    base = settings.output_dir / "live_orders" / safe
    if trade_date:
        return base / ("%s_manual_confirm.csv" % pd.Timestamp(trade_date).strftime("%Y-%m-%d"))
    candidates = sorted(base.glob("*_manual_confirm.csv"))
    if not candidates:
        return base / "latest_manual_confirm.csv"
    return candidates[-1]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从人工确认单生成真实成交回填与执行偏差报告")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, help="策略名")
    parser.add_argument("--trade-date", default=None, help="交易日期；用于定位默认人工确认单")
    parser.add_argument("--manual-confirm", type=Path, default=None, help="人工确认 CSV；默认 output/live_orders/<strategy>/<date>_manual_confirm.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    path = args.manual_confirm or _default_manual_confirm_path(args.strategy, args.trade_date)
    if not path.exists():
        print("未找到人工确认单: %s" % path, file=sys.stderr)
        return 1
    frame = pd.read_csv(path)
    detail, summary = build_execution_feedback(frame)
    paths = save_execution_feedback(get_settings(), detail, summary)
    print("真实成交回填与执行偏差分析完成")
    print("detail=%s" % paths["detail"])
    print("summary=%s" % paths["summary"])
    print("report=%s" % paths["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
