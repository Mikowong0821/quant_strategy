#!/usr/bin/env python3
"""从 AkShare/巨潮资讯拉取公告并保存为统一 announcement_events 表。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from live.announcement_source import fetch_akshare_cninfo_announcement_events, save_announcement_events
from live.stock_pool import load_stock_pool_frame


def _symbols_from_args(symbols_text: str, stock_pool: Path | None) -> list[str]:
    if symbols_text.strip():
        return [s.strip() for s in symbols_text.split(",") if s.strip()]
    if stock_pool is None:
        return []
    settings = get_settings()
    pool = load_stock_pool_frame(stock_pool, code_col=settings.stock_pool_code_col)
    return [str(x) for x in pool.loc[pool["enabled"], "symbol"].dropna().astype(str).unique()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从 AkShare/巨潮资讯拉取公告事件表")
    parser.add_argument("--symbols", default="", help="逗号分隔 ts_code；不传则可用 --stock-pool")
    parser.add_argument("--stock-pool", type=Path, default=None, help="股票池 Excel/CSV")
    parser.add_argument("--start", required=True, help="开始日期 YYYY-MM-DD 或 YYYYMMDD")
    parser.add_argument("--end", required=True, help="结束日期 YYYY-MM-DD 或 YYYYMMDD")
    parser.add_argument("--category", default="", help="巨潮公告类型；空表示全部")
    parser.add_argument("--keyword", default="", help="公告标题关键词；空表示不过滤")
    parser.add_argument("--output", type=Path, default=None, help="输出 CSV；默认 Settings.announcement_event_path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    settings = get_settings()
    symbols = _symbols_from_args(args.symbols, args.stock_pool or settings.stock_pool_path)
    if not symbols:
        raise SystemExit("请通过 --symbols 或 --stock-pool 提供股票代码")
    events = fetch_akshare_cninfo_announcement_events(
        symbols,
        args.start,
        args.end,
        category=args.category,
        keyword=args.keyword,
    )
    output = args.output or settings.announcement_event_path or (settings.data_dir / "announcement_events.csv")
    path = save_announcement_events(events, output)
    print("announcement_events=%s" % path)
    print("symbols=%d events=%d" % (len(symbols), len(events)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
