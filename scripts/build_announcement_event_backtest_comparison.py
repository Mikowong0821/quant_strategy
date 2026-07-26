"""Compare backtests with and without the real announcement event factor."""
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
from analysis.ic import daily_ic_spearman, summarize_ic
from analysis.performance import summarize
from analysis.plotting import plot_nav
from backtest.backtest_multi import run_multi_backtest
from backtest.backtest_utils import long_to_wide
from config import get_settings
from factors.factor_events import ANNOUNCEMENT_EVENT_SCORE
from factors.panel_builder import DEFAULT_FACTOR_ORDER, build_four_factor_panel
from factors.preprocess import preprocess_factor_panel
from live.data_feed import load_prices_from_csv
from live.stock_pool import load_stock_pool_frame, normalize_ts_code
from main import (
    _attach_industry_to_long_df,
    _build_rolling_score_weighted_fusion,
    _industry_series_from_long_df,
)
from models.fusion import fuse_equal_weight_zscore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="公告事件因子增量回测对比")
    parser.add_argument("--prices", required=True, help="行情长表 CSV")
    parser.add_argument("--stock-pool", required=True, help="股票池 CSV/XLSX")
    parser.add_argument("--events", required=True, help="真实公告事件 CSV/XLSX")
    parser.add_argument("--fina", default="", help="财务指标缓存 CSV，可选")
    parser.add_argument("--start", default="2025-01-01", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default="2026-07-10", help="结束日期 YYYY-MM-DD")
    parser.add_argument(
        "--output-dir",
        default="output/announcement_event_backtest",
        help="输出目录",
    )
    return parser.parse_args()


def _stock_name_map(stock_pool_path: Path) -> dict[str, str]:
    pool = load_stock_pool_frame(stock_pool_path)
    out: dict[str, str] = {}
    name_cols = ["股票简称", "股票名称", "name", "名称"]
    name_col = next((c for c in name_cols if c in pool.columns), None)
    for rec in pool.to_dict("records"):
        symbol = normalize_ts_code(rec.get("symbol"))
        if not symbol:
            continue
        out[symbol] = str(rec.get(name_col, "") or symbol) if name_col else symbol
    return out


def _filter_long_prices(long_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    out = long_df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    mask = (out["trade_date"] >= pd.Timestamp(start)) & (out["trade_date"] <= pd.Timestamp(end))
    return out.loc[mask].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _available_factors(panel: pd.DataFrame, *, include_event: bool) -> list[str]:
    factors: list[str] = []
    for factor in DEFAULT_FACTOR_ORDER:
        if factor == ANNOUNCEMENT_EVENT_SCORE and not include_event:
            continue
        if factor not in panel.columns:
            continue
        col = panel[factor]
        if col.notna().sum() == 0:
            continue
        factors.append(factor)
    return factors


def _run_strategy_set(
    panel_z: pd.DataFrame,
    prices: pd.DataFrame,
    long_df: pd.DataFrame,
    settings: Any,
    *,
    include_event: bool,
) -> dict[str, Any]:
    suffix = "WITH_EVENT" if include_event else "NO_EVENT"
    factors = _available_factors(panel_z, include_event=include_event)
    if not factors:
        raise ValueError("没有可用因子")

    equal_fused = fuse_equal_weight_zscore(panel_z[factors])
    equal_nav, equal_meta = run_multi_backtest(
        fused=equal_fused,
        prices=prices,
        settings=settings,
        factor_name=f"EQUAL_{suffix}",
        long_prices=long_df,
    )

    rolling_fused, rolling_weight_log, rolling_fusion_meta = _build_rolling_score_weighted_fusion(
        panel_z[factors],
        prices,
        settings,
    )
    rolling_nav, rolling_meta = run_multi_backtest(
        fused=rolling_fused,
        prices=prices,
        settings=settings,
        factor_name=f"ROLLING_{suffix}",
        long_prices=long_df,
    )
    rolling_meta.update(rolling_fusion_meta)

    return {
        "factors": factors,
        "equal_nav": equal_nav.rename(f"EQUAL_{suffix}"),
        "equal_meta": equal_meta,
        "rolling_nav": rolling_nav.rename(f"ROLLING_{suffix}"),
        "rolling_meta": rolling_meta,
        "rolling_weight_log": rolling_weight_log,
    }


def _performance_rows(nav_by_name: dict[str, pd.Series], prices: pd.DataFrame, settings: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, nav in nav_by_name.items():
        bench = equal_weight_benchmark_nav(prices, dates=nav.index, price_col=settings.price_col)
        stats = summarize(nav, periods=settings.trading_days_per_year)
        excess = summarize_excess(nav, bench, periods=settings.trading_days_per_year)
        rows.append(
            {
                "strategy": name,
                **stats,
                **excess,
            }
        )
    return pd.DataFrame(rows)


def _rebalance_rows(meta: dict[str, Any], strategy: str, name_map: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in meta.get("rebalance_log", []):
        picks = [str(x) for x in rec.get("picks", [])]
        selected = [str(x) for x in rec.get("selected_picks", [])]
        weights = [float(x) for x in rec.get("weights", [])]
        rows.append(
            {
                "strategy": strategy,
                "date": pd.Timestamp(rec.get("date")),
                "picks": ",".join(picks),
                "pick_names": ",".join(name_map.get(x, x) for x in picks),
                "selected_picks": ",".join(selected),
                "selected_names": ",".join(name_map.get(x, x) for x in selected),
                "weights": ",".join("%.6f" % x for x in weights),
                "weighting": rec.get("weighting", ""),
                "cash_target_weight": rec.get("cash_target_weight", 0.0),
                "target_turnover": rec.get("target_turnover", 0.0),
            }
        )
    return pd.DataFrame(rows)


def _event_validation(panel_raw: pd.DataFrame, panel_z: pd.DataFrame, prices: pd.DataFrame, settings: Any) -> pd.DataFrame:
    if ANNOUNCEMENT_EVENT_SCORE not in panel_z.columns:
        return pd.DataFrame()
    raw_event = panel_raw[ANNOUNCEMENT_EVENT_SCORE]
    z_event = panel_z[ANNOUNCEMENT_EVENT_SCORE]
    total = int(raw_event.shape[0])
    raw_nonzero = int((raw_event.fillna(0.0).abs() > 1e-12).sum())
    raw_symbols = int(raw_event[raw_event.fillna(0.0).abs() > 1e-12].index.get_level_values("symbol").nunique())
    z_nonzero = int((z_event.fillna(0.0).abs() > 1e-12).sum())
    try:
        ic = daily_ic_spearman(z_event, prices, forward_days=settings.ic_forward_days)
        ic_stats = summarize_ic(ic)
    except Exception:
        ic_stats = {}
    return pd.DataFrame(
        [
            {
                "factor": ANNOUNCEMENT_EVENT_SCORE,
                "rows": total,
                "raw_event_nonzero_rows": raw_nonzero,
                "raw_event_coverage": raw_nonzero / total if total else 0.0,
                "raw_event_symbols": raw_symbols,
                "zscore_nonzero_rows": z_nonzero,
                "zscore_nonzero_coverage": z_nonzero / total if total else 0.0,
                **ic_stats,
            }
        ]
    )


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    industry_ser = _industry_series_from_long_df(long_df, panel.index, settings)
    by_industry = bool(settings.factor_standardize_by_industry and industry_ser is not None)
    panel_z = preprocess_factor_panel(
        panel,
        industry=industry_ser,
        industry_col=settings.industry_col,
        by_industry=by_industry,
        min_industry_count=settings.factor_industry_min_count,
    )

    no_event = _run_strategy_set(panel_z, prices, long_df, settings, include_event=False)
    with_event = _run_strategy_set(panel_z, prices, long_df, settings, include_event=True)

    nav_by_name = {
        "ROLLING_NO_EVENT": no_event["rolling_nav"],
        "ROLLING_WITH_EVENT": with_event["rolling_nav"],
        "EQUAL_NO_EVENT": no_event["equal_nav"],
        "EQUAL_WITH_EVENT": with_event["equal_nav"],
    }
    nav_compare = pd.DataFrame(nav_by_name).sort_index()
    nav_compare.to_csv(output_dir / "nav_compare.csv", index_label="date")
    plot_nav(
        nav_compare,
        title="公告事件因子增量回测对比",
        save_path=output_dir / "nav_compare.png",
    )

    perf = _performance_rows(nav_by_name, prices, settings)
    perf.to_csv(output_dir / "performance_summary.csv", index=False)

    rb = pd.concat(
        [
            _rebalance_rows(no_event["rolling_meta"], "ROLLING_NO_EVENT", name_map),
            _rebalance_rows(with_event["rolling_meta"], "ROLLING_WITH_EVENT", name_map),
        ],
        ignore_index=True,
    )
    rb.to_csv(output_dir / "rebalance_log_rolling.csv", index=False)

    no_event["rolling_weight_log"].to_csv(output_dir / "rolling_factor_weight_log_no_event.csv", index=False)
    with_event["rolling_weight_log"].to_csv(output_dir / "rolling_factor_weight_log_with_event.csv", index=False)

    validation = _event_validation(panel, panel_z, prices, settings)
    validation.to_csv(output_dir / "announcement_event_factor_validation.csv", index=False)

    factor_sets = pd.DataFrame(
        [
            {"scenario": "NO_EVENT", "factors": ",".join(no_event["factors"]), "n_factors": len(no_event["factors"])},
            {
                "scenario": "WITH_EVENT",
                "factors": ",".join(with_event["factors"]),
                "n_factors": len(with_event["factors"]),
            },
        ]
    )
    factor_sets.to_csv(output_dir / "factor_sets.csv", index=False)

    print("output_dir=%s" % output_dir)
    print("standardize_by_industry=%s" % by_industry)
    print("date_range=%s~%s prices_days=%d symbols=%d" % (args.start, args.end, prices.shape[0], prices.shape[1]))
    print(perf.to_string(index=False))
    if not validation.empty:
        print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
