#!/usr/bin/env python3
"""生成实盘前股票池过滤报告与 active universe 确认文件。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from live.data_feed import load_prices_from_csv
from live.stock_pool import (
    active_universe_from_report,
    build_stock_pool_filter_report,
    load_stock_pool_frame,
    save_universe_files,
)


def _load_price_file(path: Path) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=1)
    if {"trade_date", "ts_code", "close"}.issubset(head.columns):
        return load_prices_from_csv(path)
    frame = pd.read_csv(path)
    date_col = "date" if "date" in frame.columns else frame.columns[0]
    return frame.rename(columns={date_col: "date"}).set_index("date")


def _load_trade_status(path: Path | None, trade_date: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    if frame.empty or "date" not in frame.columns or trade_date is None:
        return frame
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    dt = pd.Timestamp(trade_date)
    symbol_col = "symbol" if "symbol" in frame.columns else "ts_code"
    frame = frame[frame["date"] <= dt].sort_values([symbol_col, "date"])
    return frame.groupby(symbol_col, as_index=False).tail(1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从人工股票池生成实盘 active universe 确认文件")
    parser.add_argument("--stock-pool", type=Path, default=None, help="股票池 Excel/CSV；默认读取 Settings.stock_pool_path")
    parser.add_argument("--prices", type=Path, required=True, help="价格缓存，支持 Tushare 长表或价格宽表")
    parser.add_argument("--trade-status", type=Path, default=None, help="可选交易状态 CSV")
    parser.add_argument("--trade-date", default=None, help="确认日期；默认使用价格缓存最后日期")
    parser.add_argument("--min-price-coverage", type=float, default=0.8)
    parser.add_argument("--min-history-days", type=int, default=20)
    parser.add_argument("--liquidity-lookback-days", type=int, default=20)
    parser.add_argument("--min-avg-volume", type=float, default=0.0)
    parser.add_argument("--min-avg-amount", type=float, default=0.0)
    parser.add_argument("--output-subdir", default="live_universe", help="输出到 settings.output_dir 下的子目录")
    parser.add_argument("--allow-limit-up", action="store_true", help="不因涨停剔除股票")
    parser.add_argument("--allow-limit-down", action="store_true", help="不因跌停剔除股票")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    settings = get_settings()
    stock_pool_path = args.stock_pool or settings.stock_pool_path
    price_data = _load_price_file(args.prices)
    pool = load_stock_pool_frame(stock_pool_path, code_col=settings.stock_pool_code_col)
    trade_status = _load_trade_status(args.trade_status, args.trade_date)

    report = build_stock_pool_filter_report(
        pool,
        price_data=price_data,
        trade_status=trade_status,
        as_of_date=args.trade_date,
        min_price_coverage=args.min_price_coverage,
        min_history_days=args.min_history_days,
        liquidity_lookback_days=args.liquidity_lookback_days,
        min_avg_volume=args.min_avg_volume,
        min_avg_amount=args.min_avg_amount,
        exclude_limit_up=not args.allow_limit_up,
        exclude_limit_down=not args.allow_limit_down,
    )
    active = active_universe_from_report(report)
    paths = save_universe_files(settings, report, trade_date=args.trade_date, subdir=args.output_subdir)
    print("filter_report=%s" % paths["filter_report"])
    print("active_universe=%s" % paths["active_universe"])
    print("pool_size=%d active_size=%d" % (len(report), len(active)))
    if len(report) != len(active):
        excluded = report.loc[~report["active"], ["symbol", "name", "exclude_reason"]]
        print("excluded:")
        print(excluded.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
