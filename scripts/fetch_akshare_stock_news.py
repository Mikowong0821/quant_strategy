#!/usr/bin/env python3
"""从 AkShare 东方财富个股新闻接口拉取股票池最近新闻并缓存。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from live.news_source import fetch_akshare_stock_news_items, merge_news_items, save_news_items
from live.negative_sentiment_filter import load_sentiment_items
from live.stock_pool import load_stock_pool_frame, normalize_ts_code


def _parse_symbols(value: str) -> list[str]:
    return [normalize_ts_code(x) for x in value.split(",") if normalize_ts_code(x)]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="拉取 AkShare 个股新闻并保存为统一 news_sentiment 表")
    parser.add_argument("--stock-pool", type=Path, default=None, help="股票池 CSV/XLSX；默认使用配置股票池")
    parser.add_argument("--symbols", default="", help="逗号分隔股票代码；若提供则优先于股票池")
    parser.add_argument("--limit-symbols", type=int, default=0, help="只拉取前 N 只股票，0 表示不限制")
    parser.add_argument("--sleep-seconds", type=float, default=0.3, help="每只股票之间的等待秒数")
    parser.add_argument("--output", type=Path, default=None, help="输出 CSV；默认 data/news_sentiment_akshare.csv")
    parser.add_argument("--merge-existing", action="store_true", help="若输出文件已存在则合并去重")
    return parser


def _symbols_from_args(args: argparse.Namespace) -> list[str]:
    settings = get_settings()
    if args.symbols:
        symbols = _parse_symbols(args.symbols)
    else:
        pool_path = args.stock_pool or settings.stock_pool_path
        pool = load_stock_pool_frame(pool_path)
        pool = pool[pool["enabled"]].copy()
        symbols = [str(x) for x in pool["symbol"] if str(x)]
    if args.limit_symbols and args.limit_symbols > 0:
        symbols = symbols[: int(args.limit_symbols)]
    if not symbols:
        raise ValueError("没有可拉取的股票代码")
    return symbols


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    settings = get_settings()
    out_path = args.output or (settings.data_dir / "news_sentiment_akshare.csv")
    symbols = _symbols_from_args(args)
    incoming = fetch_akshare_stock_news_items(symbols, sleep_seconds=float(args.sleep_seconds))
    if args.merge_existing and out_path.exists():
        existing = load_sentiment_items(out_path)
        incoming = merge_news_items(existing, incoming)
    save_news_items(incoming, out_path)
    print("news_sentiment=%s" % out_path)
    print("symbols=%d" % len(symbols))
    print("rows=%d" % int(len(incoming)))
    if not incoming.empty:
        print("date_min=%s" % incoming["publish_time"].min())
        print("date_max=%s" % incoming["publish_time"].max())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
