#!/usr/bin/env python3
"""可交给 cron/launchd 调用的日终纸面交易调度入口。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from live.paper_scheduler import run_scheduled_daily_paper


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行一次日终纸面交易并写调度日志；未识别参数会透传给 run_daily_paper.py"
    )
    parser.add_argument("--log-date", default=None, help="调度日志日期，默认今天")
    return parser


def main(argv: list[str] | None = None) -> int:
    args, daily_args = build_arg_parser().parse_known_args(argv)
    result = run_scheduled_daily_paper(
        get_settings(),
        daily_args=daily_args,
        log_date=args.log_date,
    )
    print("scheduler_log=%s" % result["log_path"])
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
