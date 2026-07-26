#!/usr/bin/env python3
"""从公告事件表生成风险候选和可选黑名单文件。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from live.event_risk_filter import (
    event_risk_candidates_to_blacklist,
    load_event_risk_candidates,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从公告事件表生成风险候选和黑名单候选")
    parser.add_argument("--events", type=Path, default=None, help="公告事件 CSV/XLSX；默认 Settings.announcement_event_path")
    parser.add_argument("--as-of-date", default=None, help="风险扫描日期；默认不过滤未来日期")
    parser.add_argument("--lookback-days", type=int, default=60, help="只扫描 as-of-date 往前多少自然日")
    parser.add_argument("--block-days", type=int, default=20, help="BLACKLIST 事件默认有效自然日")
    parser.add_argument("--watch-days", type=int, default=10, help="WATCH 事件默认有效自然日")
    parser.add_argument("--block-score-threshold", type=float, default=-0.8)
    parser.add_argument("--watch-score-threshold", type=float, default=-0.3)
    parser.add_argument("--output-dir", type=Path, default=None, help="输出目录；默认 output/event_risk_filter")
    parser.add_argument("--write-blacklist", action="store_true", help="额外输出 risk_blacklist.csv")
    parser.add_argument("--include-watch", action="store_true", help="写黑名单时把 WATCH 事件也纳入")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    settings = get_settings()
    event_path = args.events or settings.announcement_event_path
    out_dir = args.output_dir or (settings.output_dir / "event_risk_filter")
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_event_risk_candidates(
        event_path,
        as_of_date=args.as_of_date,
        lookback_days=args.lookback_days,
        block_days=args.block_days,
        watch_days=args.watch_days,
        block_score_threshold=args.block_score_threshold,
        watch_score_threshold=args.watch_score_threshold,
    )
    date_tag = str(args.as_of_date or "all").replace("-", "")
    candidates_path = out_dir / ("event_risk_candidates_%s.csv" % date_tag)
    candidates.to_csv(candidates_path, index=False)
    print("event_risk_candidates=%s" % candidates_path)
    print("risk_candidates=%d" % int(len(candidates)))

    if args.write_blacklist:
        blacklist = event_risk_candidates_to_blacklist(
            candidates,
            include_watch=bool(args.include_watch),
        )
        blacklist_path = out_dir / ("risk_blacklist_%s.csv" % date_tag)
        blacklist.to_csv(blacklist_path, index=False)
        print("risk_blacklist=%s" % blacklist_path)
        print("blacklist_size=%d" % int(len(blacklist)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
