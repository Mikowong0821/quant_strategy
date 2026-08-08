#!/usr/bin/env python3
"""Backtest negative sentiment as a risk filter instead of an alpha factor."""
from __future__ import annotations

import argparse
import sys
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
from factors.factor_events import ANNOUNCEMENT_EVENT_SCORE
from factors.panel_builder import DEFAULT_FACTOR_ORDER, build_four_factor_panel
from factors.preprocess import preprocess_factor_panel
from live.data_feed import load_prices_from_csv
from live.negative_sentiment_filter import build_negative_sentiment_candidates, load_sentiment_items
from live.stock_pool import load_stock_pool_frame, normalize_ts_code
from main import _attach_industry_to_long_df, _industry_series_from_long_df
from models.fusion import fuse_equal_weight_zscore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="负面舆情过滤回测")
    parser.add_argument("--prices", default="data/prices_ftse_china_a50_real_20250101_20260726.csv")
    parser.add_argument("--stock-pool", default="data/stock_pool_ftse_china_a50_20260710.csv")
    parser.add_argument("--sentiment", default="data/news_sentiment_a50_20260701_20260726.csv")
    parser.add_argument("--fina", default="data/fina_indicator_ftse_china_a50_20250101_20260710.csv")
    parser.add_argument("--warmup-start", default="2026-04-01")
    parser.add_argument("--start", default="2026-07-01")
    parser.add_argument("--end", default="2026-07-26")
    parser.add_argument("--rebalance-freq", default="W-FRI")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--block-days", type=int, default=10)
    parser.add_argument("--watch-days", type=int, default=5)
    parser.add_argument("--block-score-threshold", type=float, default=-1.0)
    parser.add_argument("--watch-score-threshold", type=float, default=-0.4)
    parser.add_argument("--output-dir", default="output/negative_sentiment_filter_backtest")
    return parser.parse_args()


def _stock_name_map(stock_pool_path: Path) -> dict[str, str]:
    pool = load_stock_pool_frame(stock_pool_path)
    out: dict[str, str] = {}
    for rec in pool.to_dict("records"):
        symbol = normalize_ts_code(rec.get("symbol"))
        if symbol:
            out[symbol] = str(rec.get("name", "") or symbol)
    return out


def _filter_prices(long_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    out = long_df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out["ts_code"] = out["ts_code"].map(normalize_ts_code)
    mask = (out["trade_date"] >= pd.Timestamp(start)) & (out["trade_date"] <= pd.Timestamp(end))
    return out.loc[mask].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _rebalance_dates(prices: pd.DataFrame, freq: str, *, force_final: bool = True) -> pd.DatetimeIndex:
    if prices.empty:
        return pd.DatetimeIndex([])
    dates = pd.DatetimeIndex(
        [group.index[-1] for _, group in prices.dropna(how="all").groupby(pd.Grouper(freq=freq)) if not group.empty]
    )
    if force_final and len(prices.index) > 0:
        dates = dates.union(pd.DatetimeIndex([prices.index[-1]]))
    return dates.sort_values()


def _base_factor_names(panel_z: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for factor in DEFAULT_FACTOR_ORDER:
        if factor == ANNOUNCEMENT_EVENT_SCORE or factor not in panel_z.columns:
            continue
        ser = panel_z[factor]
        if ser.notna().sum() > 0 and ser.fillna(0.0).abs().sum() > 1e-12:
            out.append(factor)
    return out


def _active_negative_symbols(
    candidates: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    *,
    include_watch: bool,
    name_map: dict[str, str],
) -> pd.DataFrame:
    columns = [
        "date",
        "symbol",
        "name",
        "risk_action",
        "publish_time",
        "blacklist_until",
        "sentiment_score",
        "negative_keywords",
        "title",
        "source",
        "url",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    frame = candidates.copy()
    keep = {"BLACKLIST", "WATCH"} if include_watch else {"BLACKLIST"}
    frame = frame[frame["risk_action"].isin(keep)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["publish_time"] = pd.to_datetime(frame["publish_time"], errors="coerce")
    frame["blacklist_until"] = pd.to_datetime(frame["blacklist_until"], errors="coerce")

    rows: list[dict[str, Any]] = []
    for dt in rebalance_dates:
        active = frame[
            (frame["publish_time"] <= pd.Timestamp(dt) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
            & (frame["blacklist_until"] >= pd.Timestamp(dt))
        ].copy()
        if active.empty:
            continue
        active = active.sort_values(["symbol", "risk_action", "publish_time"])
        for symbol, group in active.groupby("symbol", sort=True):
            latest = group.sort_values("publish_time").iloc[-1]
            rows.append(
                {
                    "date": pd.Timestamp(dt),
                    "symbol": str(symbol),
                    "name": name_map.get(str(symbol), str(symbol)),
                    "risk_action": str(latest.get("risk_action", "")),
                    "publish_time": latest.get("publish_time"),
                    "blacklist_until": latest.get("blacklist_until"),
                    "sentiment_score": latest.get("sentiment_score"),
                    "negative_keywords": latest.get("negative_keywords", ""),
                    "title": latest.get("title", ""),
                    "source": latest.get("source", ""),
                    "url": latest.get("url", ""),
                }
            )
    return pd.DataFrame(rows).reindex(columns=columns).sort_values(["date", "symbol"]).reset_index(drop=True)


def _apply_negative_filter(fused: pd.Series, risk_log: pd.DataFrame, *, name: str) -> pd.Series:
    out = fused.copy()
    out.index = out.index.set_names(["date", "symbol"])
    if not risk_log.empty:
        for rec in risk_log.to_dict("records"):
            key = (pd.Timestamp(rec["date"]), str(rec["symbol"]))
            if key in out.index:
                out.loc[key] = pd.NA
    out.name = name
    return out


def _rebase_window(nav: pd.Series, start: str, end: str) -> pd.Series:
    out = nav.dropna().copy()
    out.index = pd.to_datetime(out.index)
    out = out[(out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))]
    if out.empty:
        return out
    return out / float(out.iloc[0])


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
            rows.append(
                {
                    "strategy": strategy,
                    "date": dt,
                    "selected_picks": ",".join(selected),
                    "selected_names": ",".join(name_map.get(x, x) for x in selected),
                    "weights": ",".join("%.6f" % float(x) for x in rec.get("weights", [])),
                    "target_turnover": rec.get("target_turnover", 0.0),
                    "weighting": rec.get("weighting", ""),
                }
            )
    return pd.DataFrame(rows).sort_values(["date", "strategy"]).reset_index(drop=True) if rows else pd.DataFrame()


def _write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    perf: pd.DataFrame,
    candidates: pd.DataFrame,
    block_log: pd.DataFrame,
    watch_log: pd.DataFrame,
    prices: pd.DataFrame,
    sentiment: pd.DataFrame,
) -> None:
    lines = [
        "# 负面舆情过滤短窗口回测",
        "",
        "## 测试设定",
        "",
        f"- 评价窗口：{args.start} ~ {args.end}",
        f"- 实际最新交易日：{pd.to_datetime(prices['trade_date']).max().date() if not prices.empty else ''}",
        f"- 调仓频率：{args.rebalance_freq}",
        f"- 股票池：{args.stock_pool}",
        f"- 新闻舆情记录：{len(sentiment)} 条，覆盖 {sentiment['symbol'].nunique() if not sentiment.empty else 0} 只股票",
        f"- 负面候选：{len(candidates)} 条，覆盖 {candidates['symbol'].nunique() if not candidates.empty else 0} 只股票",
        f"- BLOCK_ONLY 调仓拦截记录：{len(block_log)} 条",
        f"- BLOCK_AND_WATCH 调仓拦截记录：{len(watch_log)} 条",
        "",
        "## 绩效对比",
        "",
        perf.to_markdown(index=False),
        "",
        "## 说明",
        "",
        "这次不把新闻舆情当作 alpha 因子参与加分，而是把负面新闻当作调仓前风险门禁。若某只股票在调仓日仍处于负面舆情有效期内，则从当期候选中剔除。",
        "",
        "短窗口结果只能说明这段样本里的风险过滤效果，不能代表长期有效性。后续需要把这个门禁和公告风险、人工黑名单合并成统一风险门禁。",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name_map = _stock_name_map(Path(args.stock_pool))
    prices_full = _filter_prices(load_prices_from_csv(Path(args.prices)), args.warmup_start, args.end)
    sentiment = load_sentiment_items(Path(args.sentiment))
    candidates = build_negative_sentiment_candidates(
        sentiment,
        as_of_date=args.end,
        lookback_days=40,
        block_days=args.block_days,
        watch_days=args.watch_days,
        block_score_threshold=args.block_score_threshold,
        watch_score_threshold=args.watch_score_threshold,
    )

    settings = get_settings()
    settings = replace(
        settings,
        stock_pool_path=Path(args.stock_pool),
        tushare_price_cache_path=Path(args.prices),
        fina_indicator_cache_path=Path(args.fina) if args.fina else None,
        backtest_start=args.warmup_start,
        backtest_end=args.end,
        rebalance_freq=args.rebalance_freq,
        force_final_rebalance=True,
        top_k=int(args.top_k),
        portfolio_weighting="equal",
        persist_run_outputs=False,
    )

    long_df = _attach_industry_to_long_df(prices_full, settings)
    prices_wide = long_to_wide(long_df, settings.price_col)
    panel = build_four_factor_panel(prices_wide, long_df, settings)
    industry_ser = _industry_series_from_long_df(long_df, panel.index, settings)
    panel_z = preprocess_factor_panel(
        panel,
        industry=industry_ser,
        industry_col=settings.industry_col,
        by_industry=bool(settings.factor_standardize_by_industry and industry_ser is not None),
        min_industry_count=settings.factor_industry_min_count,
    )
    factors = _base_factor_names(panel_z)
    fused = fuse_equal_weight_zscore(panel_z[factors]).rename("BASE")

    rebalance_dates = _rebalance_dates(prices_wide, args.rebalance_freq, force_final=True)
    block_log = _active_negative_symbols(candidates, rebalance_dates, include_watch=False, name_map=name_map)
    watch_log = _active_negative_symbols(candidates, rebalance_dates, include_watch=True, name_map=name_map)

    scenarios = {
        "BASE": fused,
        "NEG_SENTIMENT_BLOCK_ONLY": _apply_negative_filter(fused, block_log, name="NEG_SENTIMENT_BLOCK_ONLY"),
        "NEG_SENTIMENT_BLOCK_AND_WATCH": _apply_negative_filter(
            fused,
            watch_log,
            name="NEG_SENTIMENT_BLOCK_AND_WATCH",
        ),
    }
    nav_window_by_name: dict[str, pd.Series] = {}
    meta_by_name: dict[str, dict[str, Any]] = {}
    for name, score in scenarios.items():
        nav, meta = run_multi_backtest(
            fused=score,
            prices=prices_wide,
            settings=settings,
            factor_name=name,
            long_prices=long_df,
        )
        nav_window_by_name[name] = _rebase_window(nav, args.start, args.end)
        meta_by_name[name] = meta

    nav_compare = pd.DataFrame(nav_window_by_name).sort_index()
    nav_compare.to_csv(output_dir / "nav_compare.csv", index_label="date")
    plot_nav(nav_compare, title="负面舆情过滤短窗口回测", save_path=output_dir / "nav_compare.png")

    prices_window = prices_wide[(prices_wide.index >= pd.Timestamp(args.start)) & (prices_wide.index <= pd.Timestamp(args.end))]
    perf = _performance_rows(nav_window_by_name, prices_window, settings)
    perf.to_csv(output_dir / "performance_summary.csv", index=False)
    _rebalance_rows(meta_by_name, name_map, args.start, args.end).to_csv(output_dir / "rebalance_log.csv", index=False)

    candidates.to_csv(output_dir / "negative_sentiment_candidates.csv", index=False)
    block_log.to_csv(output_dir / "risk_filter_log_block_only.csv", index=False)
    watch_log.to_csv(output_dir / "risk_filter_log_block_and_watch.csv", index=False)
    pd.DataFrame([{"scenario": key, "factors": ",".join(factors), "n_factors": len(factors)} for key in scenarios]).to_csv(
        output_dir / "factor_sets.csv",
        index=False,
    )
    sentiment.to_csv(output_dir / "news_sentiment_used.csv", index=False)

    _write_summary(output_dir, args, perf, candidates, block_log, watch_log, prices_full, sentiment)

    print("output_dir=%s" % output_dir)
    print("sentiment_rows=%d candidates=%d block_log=%d watch_log=%d" % (len(sentiment), len(candidates), len(block_log), len(watch_log)))
    print(perf.to_string(index=False))


if __name__ == "__main__":
    main()
