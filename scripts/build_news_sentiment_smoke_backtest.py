#!/usr/bin/env python3
"""Short-window smoke backtest for news sentiment risk factors."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.performance import summarize
from analysis.plotting import plot_nav
from factors.factor_news import NEWS_NEGATIVE_RISK_SCORE, calc_news_sentiment_factors
from live.negative_sentiment_filter import load_sentiment_items
from live.stock_pool import normalize_ts_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="新闻 / 舆情因子短窗口烟雾回测")
    parser.add_argument("--sentiment", type=Path, required=True, help="统一 news_sentiment CSV/XLSX")
    parser.add_argument("--start", default="2026-07-01", help="行情开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default="2026-07-27", help="行情结束日期 YYYY-MM-DD")
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--effective-days", type=int, default=7)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=Path("output/news_sentiment_smoke_backtest"))
    return parser.parse_args()


def _plain_code(symbol: str) -> str:
    norm = normalize_ts_code(symbol)
    return norm.split(".", 1)[0] if "." in norm else str(symbol).strip()


def _fetch_akshare_prices(symbols: list[str], start: str, end: str, sleep_seconds: float) -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError as exc:
        raise ImportError("需要安装 akshare: pip install akshare") from exc
    frames: list[pd.DataFrame] = []
    start_s = pd.Timestamp(start).strftime("%Y%m%d")
    end_s = pd.Timestamp(end).strftime("%Y%m%d")
    for symbol in symbols:
        norm = normalize_ts_code(symbol)
        if not norm:
            continue
        raw = ak.stock_zh_a_hist(
            symbol=_plain_code(norm),
            period="daily",
            start_date=start_s,
            end_date=end_s,
            adjust="qfq",
        )
        if raw is None or raw.empty:
            continue
        frame = raw.copy()
        rename = {"日期": "trade_date", "收盘": "close", "成交量": "volume", "成交额": "amount"}
        frame = frame.rename(columns=rename)
        if "trade_date" not in frame.columns or "close" not in frame.columns:
            continue
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce", format="mixed")
        frame["ts_code"] = norm
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frames.append(frame[["trade_date", "ts_code", "close"]].dropna(subset=["trade_date", "close"]))
        if sleep_seconds > 0:
            time.sleep(float(sleep_seconds))
    if not frames:
        return pd.DataFrame(columns=["trade_date", "ts_code", "close"])
    return pd.concat(frames, ignore_index=True).sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _equal_weight_nav(returns: pd.DataFrame, eligible: pd.DataFrame | None = None) -> pd.Series:
    nav = [1.0]
    dates = list(returns.index)
    out_dates = [dates[0]]
    for i in range(1, len(dates)):
        date = dates[i]
        ret = returns.loc[date]
        if eligible is None:
            symbols = list(ret.dropna().index)
        else:
            prev_date = dates[i - 1]
            prev_flag = eligible.loc[prev_date].reindex(ret.index).fillna(False)
            symbols = list(prev_flag[prev_flag].index)
        day_ret = float(ret.reindex(symbols).dropna().mean()) if symbols else 0.0
        nav.append(nav[-1] * (1.0 + day_ret))
        out_dates.append(date)
    return pd.Series(nav, index=pd.DatetimeIndex(out_dates), dtype=float)


def _summary(nav_by_name: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, nav in nav_by_name.items():
        rows.append({"strategy": name, **summarize(nav, periods=252)})
    return pd.DataFrame(rows)


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    news = load_sentiment_items(args.sentiment)
    if news.empty:
        raise ValueError("新闻 / 舆情表为空")
    symbols = sorted(set(str(x) for x in news["symbol"] if str(x)))
    prices_long = _fetch_akshare_prices(symbols, args.start, args.end, args.sleep_seconds)
    if prices_long.empty:
        raise ValueError("没有拉到可用行情")
    prices = prices_long.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    returns = prices.pct_change()
    factor_panel = calc_news_sentiment_factors(
        news,
        prices_long,
        effective_days=int(args.effective_days),
        lookback_days=int(args.lookback_days),
    )
    risk = factor_panel[NEWS_NEGATIVE_RISK_SCORE].unstack("symbol").reindex(index=prices.index, columns=prices.columns).fillna(0.0)
    eligible = risk <= 1e-12

    nav_base = _equal_weight_nav(returns).rename("EQUAL_WEIGHT_BASE")
    nav_filtered = _equal_weight_nav(returns, eligible=eligible).rename("NEWS_RISK_FILTERED")
    nav_compare = pd.DataFrame({nav_base.name: nav_base, nav_filtered.name: nav_filtered})
    perf = _summary({nav_base.name: nav_base, nav_filtered.name: nav_filtered})
    factor_nonzero = (factor_panel.fillna(0.0).abs() > 1e-12).sum().rename("nonzero_rows").reset_index()
    factor_nonzero.columns = ["factor", "nonzero_rows"]

    nav_compare.to_csv(output_dir / "nav_compare.csv", index_label="date")
    perf.to_csv(output_dir / "performance_summary.csv", index=False)
    prices_long.to_csv(output_dir / "prices_long.csv", index=False)
    factor_panel.to_csv(output_dir / "news_factor_panel.csv", index=True)
    factor_nonzero.to_csv(output_dir / "news_factor_nonzero.csv", index=False)
    plot_nav(nav_compare, title="新闻 / 舆情风险过滤短窗口烟雾回测", save_path=output_dir / "nav_compare.png")

    print("output_dir=%s" % output_dir)
    print("news_rows=%d" % len(news))
    print("symbols=%d" % len(symbols))
    print("price_rows=%d" % len(prices_long))
    print(perf.to_string(index=False))
    print(factor_nonzero.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
