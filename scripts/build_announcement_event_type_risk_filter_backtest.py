#!/usr/bin/env python3
"""Backtest typed announcement alpha with event-risk candidate filtering."""
from __future__ import annotations

import argparse
import sys
import warnings
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
from factors.factor_events import (
    ANNOUNCEMENT_EVENT_SCORE,
    ANNOUNCEMENT_EVENT_TYPE_PREFIX,
    calc_announcement_event_type_scores,
    classify_announcement_event,
    load_announcement_events,
    normalize_announcement_events,
)
from factors.panel_builder import DEFAULT_FACTOR_ORDER, build_four_factor_panel
from factors.preprocess import preprocess_factor_panel
from live.data_feed import load_prices_from_csv
from live.stock_pool import load_stock_pool_frame, normalize_ts_code
from main import (
    _attach_industry_to_long_df,
    _build_rolling_score_weighted_fusion,
    _industry_series_from_long_df,
)


DEFAULT_ALPHA_GROUPS = ("BUYBACK", "CONTRACT_PROJECT", "PERFORMANCE_POSITIVE")
DEFAULT_RISK_GROUPS = (
    "HOLDER_REDUCTION",
    "INQUIRY_PENALTY",
    "PLEDGE_FREEZE",
    "LITIGATION",
    "PERFORMANCE_NEGATIVE",
)


def _parse_groups(value: str) -> tuple[str, ...]:
    return tuple(x.strip().upper() for x in value.split(",") if x.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="公告类型风险过滤回测")
    parser.add_argument("--prices", required=True, help="行情长表 CSV")
    parser.add_argument("--stock-pool", required=True, help="股票池 CSV/XLSX")
    parser.add_argument("--events", required=True, help="公告事件 CSV/XLSX")
    parser.add_argument("--fina", default="", help="财务指标缓存 CSV，可选")
    parser.add_argument("--start", default="2025-01-01", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default="2026-06-23", help="结束日期 YYYY-MM-DD")
    parser.add_argument(
        "--alpha-groups",
        default=",".join(DEFAULT_ALPHA_GROUPS),
        help="作为收益候选的公告类型，逗号分隔",
    )
    parser.add_argument(
        "--risk-groups",
        default=",".join(DEFAULT_RISK_GROUPS),
        help="作为调仓前风险过滤的公告类型，逗号分隔",
    )
    parser.add_argument(
        "--risk-lookback-days",
        type=int,
        default=20,
        help="调仓日前多少自然日内的风险公告生效",
    )
    parser.add_argument(
        "--include-watch",
        action="store_true",
        help="保留参数位：当前类型风险过滤默认阻断 risk-groups 内全部命中股票",
    )
    parser.add_argument(
        "--output-dir",
        default="output/announcement_event_type_risk_filter_backtest",
        help="输出目录",
    )
    return parser.parse_args()


def _filter_long_prices(long_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    out = long_df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    mask = (out["trade_date"] >= pd.Timestamp(start)) & (out["trade_date"] <= pd.Timestamp(end))
    return out.loc[mask].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _stock_name_map(stock_pool_path: Path) -> dict[str, str]:
    pool = load_stock_pool_frame(stock_pool_path)
    name_cols = ["股票简称", "股票名称", "name", "名称"]
    name_col = next((c for c in name_cols if c in pool.columns), None)
    out: dict[str, str] = {}
    for rec in pool.to_dict("records"):
        symbol = normalize_ts_code(rec.get("symbol"))
        if not symbol:
            continue
        out[symbol] = str(rec.get(name_col, "") or symbol) if name_col else symbol
    return out


def _type_factor(group: str) -> str:
    return f"{ANNOUNCEMENT_EVENT_TYPE_PREFIX}{group}"


def _nonempty_factors(panel_z: pd.DataFrame, factors: list[str]) -> list[str]:
    out: list[str] = []
    for factor in factors:
        if factor not in panel_z.columns:
            continue
        ser = panel_z[factor]
        if ser.notna().sum() == 0:
            continue
        if ser.fillna(0.0).abs().sum() <= 1e-12:
            continue
        out.append(factor)
    return out


def _prepare_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str], Any]:
    settings = get_settings()
    settings = replace(
        settings,
        stock_pool_path=Path(args.stock_pool),
        tushare_price_cache_path=Path(args.prices),
        fina_indicator_cache_path=Path(args.fina) if args.fina else None,
        announcement_event_path=Path(args.events),
        backtest_start=args.start,
        backtest_end=args.end,
        force_final_rebalance=True,
        persist_run_outputs=False,
    )
    long_df = load_prices_from_csv(Path(args.prices))
    long_df = _filter_long_prices(long_df, args.start, args.end)
    long_df = _attach_industry_to_long_df(long_df, settings)
    prices = long_to_wide(long_df, settings.price_col)
    name_map = _stock_name_map(Path(args.stock_pool))

    panel = build_four_factor_panel(prices, long_df, settings)
    groups = tuple(sorted(set(_parse_groups(args.alpha_groups) + _parse_groups(args.risk_groups))))
    events = load_announcement_events(Path(args.events))
    type_panel = calc_announcement_event_type_scores(
        events,
        long_df,
        effective_days=int(settings.announcement_event_effective_days),
        categories=groups,
    )
    panel = pd.concat([panel, type_panel], axis=1).sort_index()
    industry_ser = _industry_series_from_long_df(long_df, panel.index, settings)
    by_industry = bool(settings.factor_standardize_by_industry and industry_ser is not None)
    panel_z = preprocess_factor_panel(
        panel,
        industry=industry_ser,
        industry_col=settings.industry_col,
        by_industry=by_industry,
        min_industry_count=settings.factor_industry_min_count,
    )
    return panel_z, prices, long_df, name_map, settings


def _base_alpha_factors(panel_z: pd.DataFrame, alpha_groups: tuple[str, ...]) -> list[str]:
    base = [
        factor
        for factor in DEFAULT_FACTOR_ORDER
        if factor != ANNOUNCEMENT_EVENT_SCORE and factor in panel_z.columns
    ]
    factors = base + [_type_factor(group) for group in alpha_groups]
    return _nonempty_factors(panel_z, factors)


def _rebalance_dates(prices: pd.DataFrame, settings: Any) -> pd.DatetimeIndex:
    alias = {"M": "ME", "Q": "QE", "A": "YE", "Y": "YE"}.get(settings.rebalance_freq, settings.rebalance_freq)
    if prices.empty:
        return pd.DatetimeIndex([])
    dates = pd.DatetimeIndex(prices.resample(alias).apply(lambda x: x.index[-1]).iloc[:, 0])
    if bool(getattr(settings, "force_final_rebalance", False)) and len(prices.index) > 0:
        dates = dates.union(pd.DatetimeIndex([prices.index[-1]]))
    return dates.sort_values()


def _risk_event_frame(
    events: pd.DataFrame,
    risk_groups: tuple[str, ...],
) -> pd.DataFrame:
    frame = normalize_announcement_events(events)
    if frame.empty:
        return pd.DataFrame(columns=["event_date", "symbol", "event_group", "title"])
    out = frame.copy()
    out["event_group"] = [
        classify_announcement_event(event_type, title)
        for event_type, title in zip(out["event_type"], out["title"], strict=True)
    ]
    out = out[out["event_group"].isin(set(risk_groups))].copy()
    return out.sort_values(["event_date", "symbol"]).reset_index(drop=True)


def _risk_filter_log(
    *,
    events: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    lookback_days: int,
    name_map: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if events.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "name",
                "event_groups",
                "event_count",
                "latest_event_date",
                "latest_title",
                "action",
            ]
        )
    for dt in rebalance_dates:
        start = pd.Timestamp(dt).normalize() - pd.Timedelta(days=int(lookback_days))
        sub = events[(events["event_date"] <= dt) & (events["event_date"] >= start)].copy()
        if sub.empty:
            continue
        for symbol, g in sub.groupby("symbol", sort=True):
            latest = g.sort_values("event_date").iloc[-1]
            rows.append(
                {
                    "date": pd.Timestamp(dt),
                    "symbol": str(symbol),
                    "name": name_map.get(str(symbol), str(symbol)),
                    "event_groups": ",".join(sorted(set(g["event_group"].astype(str)))),
                    "event_count": int(len(g)),
                    "latest_event_date": pd.Timestamp(latest["event_date"]),
                    "latest_title": str(latest.get("title", "")),
                    "action": "EXCLUDE_FROM_REBALANCE_CANDIDATES",
                }
            )
    return pd.DataFrame(rows).sort_values(["date", "symbol"]).reset_index(drop=True)


def _apply_risk_filter(
    fused: pd.Series,
    risk_log: pd.DataFrame,
) -> pd.Series:
    out = fused.copy()
    out.index = out.index.set_names(["date", "symbol"])
    if risk_log.empty or out.empty:
        out.name = "ROLLING_TYPE_ALPHA_RISK_FILTERED"
        return out
    finite = out[pd.notna(out)]
    floor = float(finite.min() - 1_000_000.0) if not finite.empty else -1_000_000.0
    for rec in risk_log.to_dict("records"):
        key = (pd.Timestamp(rec["date"]), str(rec["symbol"]))
        if key in out.index:
            out.loc[key] = floor
    out.name = "ROLLING_TYPE_ALPHA_RISK_FILTERED"
    return out


def _performance_rows(
    nav_by_name: dict[str, pd.Series],
    prices: pd.DataFrame,
    settings: Any,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, nav in nav_by_name.items():
        bench = equal_weight_benchmark_nav(prices, dates=nav.index, price_col=settings.price_col)
        stats = summarize(nav, periods=settings.trading_days_per_year)
        excess = summarize_excess(nav, bench, periods=settings.trading_days_per_year)
        rows.append({"strategy": name, **stats, **excess})
    return pd.DataFrame(rows)


def _rebalance_rows(meta: dict[str, Any], strategy: str, name_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in meta.get("rebalance_log", []):
        picks = [str(x) for x in rec.get("picks", [])]
        selected = [str(x) for x in rec.get("selected_picks", [])]
        rows.append(
            {
                "strategy": strategy,
                "date": pd.Timestamp(rec.get("date")),
                "picks": ",".join(picks),
                "pick_names": ",".join(name_map.get(x, x) for x in picks),
                "selected_picks": ",".join(selected),
                "selected_names": ",".join(name_map.get(x, x) for x in selected),
                "target_turnover": rec.get("target_turnover", 0.0),
                "cash_target_weight": rec.get("cash_target_weight", 0.0),
            }
        )
    return pd.DataFrame(rows)


def _diff_summary(perf: pd.DataFrame) -> pd.DataFrame:
    if perf.empty:
        return pd.DataFrame()
    base = perf[perf["strategy"] == "ROLLING_TYPE_ALPHA"]
    filtered = perf[perf["strategy"] == "ROLLING_TYPE_ALPHA_RISK_FILTERED"]
    if base.empty or filtered.empty:
        return pd.DataFrame()
    b = base.iloc[0]
    f = filtered.iloc[0]
    return pd.DataFrame(
        [
            {
                "base_strategy": b["strategy"],
                "filtered_strategy": f["strategy"],
                "base_final_nav": float(b["final_nav"]),
                "filtered_final_nav": float(f["final_nav"]),
                "delta_final_nav": float(f["final_nav"] - b["final_nav"]),
                "base_total_return": float(b["total_return"]),
                "filtered_total_return": float(f["total_return"]),
                "delta_total_return": float(f["total_return"] - b["total_return"]),
                "base_max_drawdown": float(b["max_drawdown"]),
                "filtered_max_drawdown": float(f["max_drawdown"]),
                "delta_max_drawdown": float(f["max_drawdown"] - b["max_drawdown"]),
                "base_information_ratio": float(b["information_ratio"]),
                "filtered_information_ratio": float(f["information_ratio"]),
                "delta_information_ratio": float(f["information_ratio"] - b["information_ratio"]),
            }
        ]
    )


def main() -> int:
    args = _parse_args()
    warnings.filterwarnings("ignore", message="An input array is constant")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_z, prices, long_df, name_map, settings = _prepare_data(args)
    factors = _base_alpha_factors(panel_z, _parse_groups(args.alpha_groups))
    if not factors:
        raise ValueError("没有可用的基础 + 收益类公告因子")

    rolling_fused, weight_log, fusion_meta = _build_rolling_score_weighted_fusion(panel_z[factors], prices, settings)
    nav_alpha, meta_alpha = run_multi_backtest(
        fused=rolling_fused,
        prices=prices,
        settings=settings,
        factor_name="ROLLING_TYPE_ALPHA",
        long_prices=long_df,
    )
    meta_alpha.update(fusion_meta)

    risk_events = _risk_event_frame(load_announcement_events(Path(args.events)), _parse_groups(args.risk_groups))
    rebal_dates = _rebalance_dates(prices, settings)
    risk_log = _risk_filter_log(
        events=risk_events,
        rebalance_dates=rebal_dates,
        lookback_days=int(args.risk_lookback_days),
        name_map=name_map,
    )
    filtered_fused = _apply_risk_filter(rolling_fused, risk_log)
    nav_filtered, meta_filtered = run_multi_backtest(
        fused=filtered_fused,
        prices=prices,
        settings=settings,
        factor_name="ROLLING_TYPE_ALPHA_RISK_FILTERED",
        long_prices=long_df,
    )

    nav_by_name = {
        "ROLLING_TYPE_ALPHA": nav_alpha.rename("ROLLING_TYPE_ALPHA"),
        "ROLLING_TYPE_ALPHA_RISK_FILTERED": nav_filtered.rename("ROLLING_TYPE_ALPHA_RISK_FILTERED"),
    }
    nav_compare = pd.DataFrame(nav_by_name).sort_index()
    perf = _performance_rows(nav_by_name, prices, settings)
    diff = _diff_summary(perf)
    rebalances = pd.concat(
        [
            _rebalance_rows(meta_alpha, "ROLLING_TYPE_ALPHA", name_map),
            _rebalance_rows(meta_filtered, "ROLLING_TYPE_ALPHA_RISK_FILTERED", name_map),
        ],
        ignore_index=True,
    )

    nav_compare.to_csv(output_dir / "nav_compare.csv", index_label="date")
    perf.to_csv(output_dir / "performance_summary.csv", index=False)
    diff.to_csv(output_dir / "risk_filter_incremental_effect.csv", index=False)
    rebalances.to_csv(output_dir / "rebalance_log_rolling.csv", index=False)
    weight_log.to_csv(output_dir / "rolling_factor_weight_log_type_alpha.csv", index=False)
    risk_events.to_csv(output_dir / "typed_risk_events.csv", index=False)
    risk_log.to_csv(output_dir / "risk_filter_log.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "TYPE_ALPHA",
                "n_factors": len(factors),
                "factors": ",".join(factors),
                "risk_groups": "",
                "risk_lookback_days": "",
            },
            {
                "scenario": "TYPE_ALPHA_RISK_FILTERED",
                "n_factors": len(factors),
                "factors": ",".join(factors),
                "risk_groups": ",".join(_parse_groups(args.risk_groups)),
                "risk_lookback_days": int(args.risk_lookback_days),
            },
        ]
    ).to_csv(output_dir / "scenario_config.csv", index=False)
    plot_nav(nav_compare, title="公告类型收益因子 vs 风险过滤", save_path=output_dir / "nav_compare.png")

    print("performance_summary=%s" % (output_dir / "performance_summary.csv"))
    print("risk_filter_log=%s" % (output_dir / "risk_filter_log.csv"))
    if not perf.empty:
        cols = ["strategy", "final_nav", "total_return", "ann_return", "max_drawdown", "information_ratio"]
        print(perf[[c for c in cols if c in perf.columns]].to_string(index=False))
    if not diff.empty:
        print(diff.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
