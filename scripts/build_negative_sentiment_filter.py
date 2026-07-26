#!/usr/bin/env python3
"""从新闻/舆情表生成负面舆情风险候选和可选黑名单。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from live.negative_sentiment_filter import (
    build_negative_sentiment_candidates,
    load_sentiment_items,
    negative_sentiment_candidates_to_blacklist,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从新闻/舆情表生成负面风险候选")
    parser.add_argument("--sentiment", type=Path, required=True, help="新闻/舆情 CSV/XLSX")
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--block-days", type=int, default=10)
    parser.add_argument("--watch-days", type=int, default=5)
    parser.add_argument("--block-score-threshold", type=float, default=-1.0)
    parser.add_argument("--watch-score-threshold", type=float, default=-0.4)
    parser.add_argument("--output-dir", type=Path, default=None, help="输出目录；默认 output/negative_sentiment_filter")
    parser.add_argument("--write-blacklist", action="store_true")
    parser.add_argument("--include-watch", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    settings = get_settings()
    out_dir = args.output_dir or (settings.output_dir / "negative_sentiment_filter")
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_sentiment_items(args.sentiment)
    candidates = build_negative_sentiment_candidates(
        items,
        as_of_date=args.as_of_date,
        lookback_days=args.lookback_days,
        block_days=args.block_days,
        watch_days=args.watch_days,
        block_score_threshold=args.block_score_threshold,
        watch_score_threshold=args.watch_score_threshold,
    )
    date_tag = str(args.as_of_date or "all").replace("-", "")
    candidates_path = out_dir / ("negative_sentiment_candidates_%s.csv" % date_tag)
    candidates.to_csv(candidates_path, index=False)
    print("negative_sentiment_candidates=%s" % candidates_path)
    print("risk_candidates=%d" % int(len(candidates)))

    if args.write_blacklist:
        blacklist = negative_sentiment_candidates_to_blacklist(
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
