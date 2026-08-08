#!/usr/bin/env python3
"""A50 short-window weekly backtest with announcement and news factors."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.benchmark import equal_weight_benchmark_nav, summarize_excess
from analysis.performance import summarize
from analysis.plotting import plot_nav
from backtest.backtest_multi import run_multi_backtest
from backtest.backtest_utils import long_to_wide
from config import get_settings
from factors.factor_events import ANNOUNCEMENT_EVENT_SCORE, load_announcement_events
from factors.factor_news import (
    NEWS_HEAT_7D,
    NEWS_NEGATIVE_COUNT_7D,
    NEWS_NEGATIVE_RISK_SCORE,
    NEWS_SENTIMENT_DECAY,
    calc_news_sentiment_factors,
)
from factors.panel_builder import DEFAULT_FACTOR_ORDER, build_four_factor_panel
from factors.preprocess import preprocess_factor_panel
from live.announcement_source import fetch_akshare_cninfo_announcement_events, save_announcement_events
from live.data_feed import fetch_daily_panel, load_prices_from_csv
from live.news_source import fetch_akshare_stock_news_items, normalize_akshare_stock_news, save_news_items
from live.stock_pool import load_stock_pool_frame, normalize_ts_code
from main import (
    _attach_industry_to_long_df,
    _build_rolling_score_weighted_fusion,
    _industry_series_from_long_df,
)
from models.fusion import fuse_equal_weight_zscore


NEWS_NEGATIVE_RISK_AVOID = "NEWS_NEGATIVE_RISK_AVOID"
NEWS_NEGATIVE_COUNT_AVOID = "NEWS_NEGATIVE_COUNT_AVOID"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A50 公告 + 新闻舆情短窗口周频回测")
    parser.add_argument("--stock-pool", default="data/stock_pool_ftse_china_a50_20260710.csv")
    parser.add_argument("--warmup-start", default="2025-01-01", help="因子和权重热身开始日期")
    parser.add_argument("--start", default="2026-07-01", help="绩效评价开始日期")
    parser.add_argument("--end", default="2026-07-26", help="绩效评价结束日期")
    parser.add_argument("--rebalance-freq", default="W-FRI", help="周频调仓默认用每周五")
    parser.add_argument("--fusion-mode", choices=["equal", "rolling"], default="equal")
    parser.add_argument("--portfolio-weighting", default="equal", choices=["equal", "max_sharpe", "risk_parity"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--prices", default="data/prices_ftse_china_a50_real_20250101_20260726.csv")
    parser.add_argument("--fina", default="data/fina_indicator_ftse_china_a50_20250101_20260710.csv")
    parser.add_argument("--events", default="data/announcement_events_a50_20260701_20260726.csv")
    parser.add_argument("--news", default="data/news_sentiment_a50_20260701_20260726.csv")
    parser.add_argument("--refresh-prices", action="store_true")
    parser.add_argument("--refresh-events", action="store_true")
    parser.add_argument("--refresh-news", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--news-timeout-seconds", type=float, default=12.0)
    parser.add_argument("--output-dir", default="output/a50_event_news_weekly_20260701_20260726")
    return parser.parse_args()


def _symbols_and_names(stock_pool_path: Path) -> tuple[list[str], dict[str, str]]:
    pool = load_stock_pool_frame(stock_pool_path)
    pool = pool[pool["enabled"]].copy()
    symbols = [normalize_ts_code(x) for x in pool["symbol"]]
    symbols = [x for x in symbols if x]
    name_map = {str(rec["symbol"]): str(rec.get("name", "") or rec["symbol"]) for rec in pool.to_dict("records")}
    return sorted(dict.fromkeys(symbols)), name_map


def _ensure_prices(args: argparse.Namespace, symbols: list[str]) -> pd.DataFrame:
    path = Path(args.prices)
    if args.refresh_prices or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        prices = fetch_daily_panel(symbols, args.warmup_start, args.end)
        prices.to_csv(path, index=False)
        return prices
    return load_prices_from_csv(path)


def _ensure_events(args: argparse.Namespace, symbols: list[str]) -> pd.DataFrame:
    path = Path(args.events)
    if args.refresh_events or not path.exists():
        events = fetch_akshare_cninfo_announcement_events(symbols, args.start, args.end)
        save_announcement_events(events, path)
        return events
    return load_announcement_events(path)


def _akshare_news_worker(symbol: str, queue: mp.Queue) -> None:
    try:
        import akshare as ak

        raw = ak.stock_news_em(symbol=symbol.split(".", 1)[0])
        frame = normalize_akshare_stock_news(raw, symbol=symbol)
        queue.put(frame.to_dict("records"))
    except Exception as exc:
        queue.put({"error": str(exc)})


def _fetch_akshare_news_with_timeout(
    symbols: list[str],
    *,
    timeout_seconds: float,
    sleep_seconds: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    total = len(symbols)
    for i, symbol in enumerate(symbols, start=1):
        queue: mp.Queue = mp.Queue()
        proc = mp.Process(target=_akshare_news_worker, args=(symbol, queue))
        proc.start()
        proc.join(timeout=float(timeout_seconds))
        if proc.is_alive():
            proc.terminate()
            proc.join()
            print("news_fetch timeout %s (%d/%d)" % (symbol, i, total), flush=True)
        else:
            payload = queue.get() if not queue.empty() else []
            if isinstance(payload, list) and payload:
                frame = pd.DataFrame(payload)
                frames.append(frame)
                print("news_fetch ok %s rows=%d (%d/%d)" % (symbol, len(frame), i, total), flush=True)
            elif isinstance(payload, dict) and payload.get("error"):
                print("news_fetch skip %s error=%s (%d/%d)" % (symbol, payload["error"], i, total), flush=True)
            else:
                print("news_fetch empty %s (%d/%d)" % (symbol, i, total), flush=True)
        queue.close()
        if sleep_seconds > 0:
            time.sleep(float(sleep_seconds))
    if not frames:
        return pd.DataFrame()
    from live.news_source import merge_news_items

    return merge_news_items(*frames)


def _ensure_news(args: argparse.Namespace, symbols: list[str]) -> pd.DataFrame:
    path = Path(args.news)
    if args.refresh_news or not path.exists():
        if float(args.news_timeout_seconds) > 0:
            news = _fetch_akshare_news_with_timeout(
                symbols,
                timeout_seconds=float(args.news_timeout_seconds),
                sleep_seconds=float(args.sleep_seconds),
            )
        else:
            news = fetch_akshare_stock_news_items(symbols, sleep_seconds=float(args.sleep_seconds))
        if not news.empty:
            news = news.copy()
            news["publish_time"] = pd.to_datetime(news["publish_time"], errors="coerce")
            start = pd.Timestamp(args.start)
            end = pd.Timestamp(args.end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            news = news[(news["publish_time"] >= start) & (news["publish_time"] <= end)].copy()
        save_news_items(news, path)
        return news
    from live.negative_sentiment_filter import load_sentiment_items

    return load_sentiment_items(path)


def _filter_prices(long_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    out = long_df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out["ts_code"] = out["ts_code"].map(normalize_ts_code)
    mask = (out["trade_date"] >= pd.Timestamp(start)) & (out["trade_date"] <= pd.Timestamp(end))
    return out.loc[mask].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _scenario_factors(panel_z: pd.DataFrame) -> dict[str, list[str]]:
    base = [
        factor
        for factor in DEFAULT_FACTOR_ORDER
        if factor != ANNOUNCEMENT_EVENT_SCORE and factor in panel_z.columns
    ]
    announcement = [ANNOUNCEMENT_EVENT_SCORE] if ANNOUNCEMENT_EVENT_SCORE in panel_z.columns else []
    news = [
        factor
        for factor in [
            NEWS_SENTIMENT_DECAY,
            NEWS_NEGATIVE_RISK_AVOID,
            NEWS_NEGATIVE_COUNT_AVOID,
            NEWS_HEAT_7D,
        ]
        if factor in panel_z.columns
    ]
    raw = {
        "BASE": base,
        "ANNOUNCEMENT": base + announcement,
        "NEWS": base + news,
        "ANNOUNCEMENT_NEWS": base + announcement + news,
    }
    out: dict[str, list[str]] = {}
    for name, factors in raw.items():
        selected: list[str] = []
        for factor in factors:
            if factor not in panel_z.columns:
                continue
            ser = panel_z[factor]
            if ser.notna().sum() == 0:
                continue
            if ser.fillna(0.0).abs().sum() <= 1e-12:
                continue
            selected.append(factor)
        out[name] = selected
    return out


def _rebase_window(nav: pd.Series, start: str, end: str) -> pd.Series:
    out = nav.copy().dropna()
    out.index = pd.to_datetime(out.index)
    out = out[(out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))]
    if out.empty:
        return out
    return out / float(out.iloc[0])


def _run_scenario(
    name: str,
    factors: list[str],
    panel_z: pd.DataFrame,
    prices: pd.DataFrame,
    long_df: pd.DataFrame,
    settings: Any,
    *,
    fusion_mode: str,
) -> tuple[pd.Series, dict[str, Any], pd.DataFrame]:
    if not factors:
        raise ValueError("%s 没有可用因子" % name)
    if fusion_mode == "rolling":
        fused, weight_log, meta = _build_rolling_score_weighted_fusion(panel_z[factors], prices, settings)
    else:
        fused = fuse_equal_weight_zscore(panel_z[factors])
        weight_log = pd.DataFrame()
        meta = {"fusion_mode": "equal", "factors": list(factors)}
    nav, bt_meta = run_multi_backtest(
        fused=fused,
        prices=prices,
        settings=settings,
        factor_name=name,
        long_prices=long_df,
    )
    bt_meta.update(meta)
    return nav.rename(name), bt_meta, weight_log


def _performance_rows(nav_by_name: dict[str, pd.Series], prices_window: pd.DataFrame, settings: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, nav in nav_by_name.items():
        bench = equal_weight_benchmark_nav(prices_window, dates=nav.index, price_col=settings.price_col)
        stats = summarize(nav, periods=settings.trading_days_per_year)
        excess = summarize_excess(nav, bench, periods=settings.trading_days_per_year)
        rows.append({"strategy": name, **stats, **excess})
    return pd.DataFrame(rows)


def _rebalance_rows(meta_by_name: dict[str, dict[str, Any]], name_map: dict[str, str], start: str, end: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    for strategy, meta in meta_by_name.items():
        for rec in meta.get("rebalance_log", []):
            dt = pd.Timestamp(rec.get("date"))
            if dt < start_ts or dt > end_ts:
                continue
            selected = [str(x) for x in rec.get("selected_picks", [])]
            picks = [str(x) for x in rec.get("picks", [])]
            rows.append(
                {
                    "strategy": strategy,
                    "date": dt,
                    "selected_picks": ",".join(selected),
                    "selected_names": ",".join(name_map.get(x, x) for x in selected),
                    "picks": ",".join(picks),
                    "pick_names": ",".join(name_map.get(x, x) for x in picks),
                    "weights": ",".join("%.6f" % float(x) for x in rec.get("weights", [])),
                    "target_turnover": rec.get("target_turnover", 0.0),
                    "cash_target_weight": rec.get("cash_target_weight", 0.0),
                    "weighting": rec.get("weighting", ""),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["date", "strategy"]).reset_index(drop=True)


def _factor_coverage(panel_raw: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = int(panel_raw.shape[0])
    for factor in factors:
        if factor not in panel_raw.columns:
            continue
        ser = panel_raw[factor]
        non_na = int(ser.notna().sum())
        nonzero = int((ser.fillna(0.0).abs() > 1e-12).sum())
        rows.append(
            {
                "factor": factor,
                "rows": total,
                "non_na_rows": non_na,
                "non_na_coverage": non_na / total if total else 0.0,
                "nonzero_rows": nonzero,
                "nonzero_coverage": nonzero / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _write_summary(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    prices_full: pd.DataFrame,
    prices_window: pd.DataFrame,
    events: pd.DataFrame,
    news: pd.DataFrame,
    perf: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    latest_trade = pd.to_datetime(prices_full["trade_date"]).max() if not prices_full.empty else pd.NaT
    lines = [
        "# A50 公告与新闻舆情短窗口周频回测",
        "",
        "## 测试设定",
        "",
        f"- 评价窗口：{args.start} ~ {args.end}",
        f"- 实际最新交易日：{latest_trade.date() if pd.notna(latest_trade) else ''}",
        f"- 因子热身起点：{args.warmup_start}",
        f"- 调仓频率：{args.rebalance_freq}",
        f"- 融合方式：{args.fusion_mode}",
        f"- 股票池：{args.stock_pool}",
        f"- 股票数：{prices_full['ts_code'].nunique() if not prices_full.empty else 0}",
        f"- 评价窗口行情行数：{len(prices_window)}",
        f"- 公告事件数：{len(events)}",
        f"- 新闻条数：{len(news)}",
        "",
        "## 绩效对比",
        "",
        perf.to_markdown(index=False),
        "",
        "## 信息类因子覆盖",
        "",
        coverage.to_markdown(index=False),
        "",
        "## 说明",
        "",
        "这次是短窗口烟雾回测，不是长期有效性结论。它主要验证：A50 真实行情更新后，公告事件和新闻舆情因子能否在同一周频调仓框架下参与组合排序，并观察短期净值是否出现差异。",
        "",
        "如果某个信息类因子覆盖很低，即使短期收益有变化，也不能直接解释为该因子有效；更稳妥的做法是继续积累新闻和公告缓存，再做更长窗口、多股票池和滚动样本外验证。",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols, name_map = _symbols_and_names(Path(args.stock_pool))
    prices_full = _ensure_prices(args, symbols)
    prices_full = _filter_prices(prices_full, args.warmup_start, args.end)
    events = _ensure_events(args, symbols)
    news = _ensure_news(args, symbols)

    settings = get_settings()
    settings = replace(
        settings,
        stock_pool_path=Path(args.stock_pool),
        tushare_price_cache_path=Path(args.prices),
        fina_indicator_cache_path=Path(args.fina) if args.fina else None,
        announcement_event_path=Path(args.events),
        backtest_start=args.warmup_start,
        backtest_end=args.end,
        rebalance_freq=args.rebalance_freq,
        force_final_rebalance=True,
        top_k=int(args.top_k),
        portfolio_weighting=args.portfolio_weighting,
        persist_run_outputs=False,
    )

    long_df = _attach_industry_to_long_df(prices_full, settings)
    prices_wide = long_to_wide(long_df, settings.price_col)
    panel = build_four_factor_panel(prices_wide, long_df, settings)

    news_panel = calc_news_sentiment_factors(news, long_df).reindex(panel.index)
    news_panel[NEWS_NEGATIVE_RISK_AVOID] = -news_panel[NEWS_NEGATIVE_RISK_SCORE]
    news_panel[NEWS_NEGATIVE_COUNT_AVOID] = -news_panel[NEWS_NEGATIVE_COUNT_7D]
    panel = pd.concat(
        [
            panel,
            news_panel[
                [
                    NEWS_SENTIMENT_DECAY,
                    NEWS_NEGATIVE_RISK_AVOID,
                    NEWS_NEGATIVE_COUNT_AVOID,
                    NEWS_HEAT_7D,
                ]
            ],
        ],
        axis=1,
    ).sort_index()

    industry_ser = _industry_series_from_long_df(long_df, panel.index, settings)
    by_industry = bool(settings.factor_standardize_by_industry and industry_ser is not None)
    panel_z = preprocess_factor_panel(
        panel,
        industry=industry_ser,
        industry_col=settings.industry_col,
        by_industry=by_industry,
        min_industry_count=settings.factor_industry_min_count,
    )

    factor_sets = _scenario_factors(panel_z)
    nav_window_by_name: dict[str, pd.Series] = {}
    meta_by_name: dict[str, dict[str, Any]] = {}
    weight_logs: dict[str, pd.DataFrame] = {}
    for scenario, factors in factor_sets.items():
        nav, meta, weight_log = _run_scenario(
            scenario,
            factors,
            panel_z,
            prices_wide,
            long_df,
            settings,
            fusion_mode=args.fusion_mode,
        )
        nav_window_by_name[scenario] = _rebase_window(nav, args.start, args.end)
        meta_by_name[scenario] = meta
        weight_logs[scenario] = weight_log
        # Small pause to keep console output readable when running under schedulers.
        time.sleep(0.01)

    nav_compare = pd.DataFrame(nav_window_by_name).sort_index()
    nav_compare.to_csv(output_dir / "nav_compare.csv", index_label="date")
    plot_nav(nav_compare, title="A50 公告与新闻舆情短窗口周频回测", save_path=output_dir / "nav_compare.png")

    prices_window = prices_wide.loc[
        (prices_wide.index >= pd.Timestamp(args.start)) & (prices_wide.index <= pd.Timestamp(args.end))
    ]
    perf = _performance_rows(nav_window_by_name, prices_window, settings)
    perf.to_csv(output_dir / "performance_summary.csv", index=False)

    rb = _rebalance_rows(meta_by_name, name_map, args.start, args.end)
    rb.to_csv(output_dir / "rebalance_log_rolling.csv", index=False)

    pd.DataFrame(
        [{"scenario": k, "factors": ",".join(v), "n_factors": len(v)} for k, v in factor_sets.items()]
    ).to_csv(output_dir / "factor_sets.csv", index=False)
    for scenario, frame in weight_logs.items():
        if not frame.empty:
            frame.to_csv(output_dir / ("rolling_factor_weight_log_%s.csv" % scenario.lower()), index=False)

    coverage = _factor_coverage(
        panel,
        [
            ANNOUNCEMENT_EVENT_SCORE,
            NEWS_SENTIMENT_DECAY,
            NEWS_NEGATIVE_RISK_AVOID,
            NEWS_NEGATIVE_COUNT_AVOID,
            NEWS_HEAT_7D,
        ],
    )
    coverage.to_csv(output_dir / "information_factor_coverage.csv", index=False)
    events.to_csv(output_dir / "announcement_events_used.csv", index=False)
    news.to_csv(output_dir / "news_sentiment_used.csv", index=False)
    prices_full.to_csv(output_dir / "prices_used.csv", index=False)

    _write_summary(
        output_dir=output_dir,
        args=args,
        prices_full=prices_full,
        prices_window=prices_full[
            (prices_full["trade_date"] >= pd.Timestamp(args.start))
            & (prices_full["trade_date"] <= pd.Timestamp(args.end))
        ],
        events=events,
        news=news,
        perf=perf,
        coverage=coverage,
    )

    print("output_dir=%s" % output_dir)
    print("symbols=%d" % len(symbols))
    print("price_rows=%d latest_trade=%s" % (len(prices_full), prices_full["trade_date"].max()))
    print("events=%d news=%d" % (len(events), len(news)))
    print(perf.to_string(index=False))


if __name__ == "__main__":
    main()
