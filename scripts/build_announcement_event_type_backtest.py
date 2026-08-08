#!/usr/bin/env python3
"""Compare announcement total-score vs typed announcement factors in backtests."""
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
    load_announcement_events,
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
from models.fusion import fuse_equal_weight_zscore


DEFAULT_ALPHA_GROUPS = ("BUYBACK", "CONTRACT_PROJECT", "PERFORMANCE_POSITIVE")
DEFAULT_RISK_GROUPS = (
    "HOLDER_REDUCTION",
    "INQUIRY_PENALTY",
    "PLEDGE_FREEZE",
    "LITIGATION",
    "PERFORMANCE_NEGATIVE",
)


def _parse_universe(value: str) -> dict[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--universe 需要格式 name=prices|stock_pool|events|fina")
    name, payload = value.split("=", 1)
    parts = payload.split("|")
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError("--universe 需要格式 name=prices|stock_pool|events|fina")
    return {
        "name": name.strip(),
        "prices": parts[0].strip(),
        "stock_pool": parts[1].strip(),
        "events": parts[2].strip(),
        "fina": parts[3].strip() if len(parts) == 4 else "",
    }


def _parse_groups(value: str) -> tuple[str, ...]:
    return tuple(x.strip().upper() for x in value.split(",") if x.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="公告类型因子组合回测")
    parser.add_argument(
        "--universe",
        action="append",
        type=_parse_universe,
        required=True,
        help="股票池，格式 name=prices|stock_pool|events|fina，可重复传入",
    )
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
        help="作为风险过滤候选的公告类型，逗号分隔",
    )
    parser.add_argument(
        "--output-dir",
        default="output/announcement_event_type_backtest",
        help="输出目录",
    )
    parser.add_argument(
        "--include-equal",
        action="store_true",
        help="额外运行 EQUAL 等权融合口径；默认只运行 ROLLING 主策略口径",
    )
    return parser.parse_args()


def _filter_long_prices(long_df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    out = long_df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    mask = (out["trade_date"] >= pd.Timestamp(start)) & (out["trade_date"] <= pd.Timestamp(end))
    return out.loc[mask].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


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


def _scenario_factor_sets(
    panel_z: pd.DataFrame,
    *,
    alpha_groups: tuple[str, ...],
    risk_groups: tuple[str, ...],
) -> dict[str, list[str]]:
    base = [
        factor
        for factor in DEFAULT_FACTOR_ORDER
        if factor != ANNOUNCEMENT_EVENT_SCORE and factor in panel_z.columns
    ]
    total = base + [ANNOUNCEMENT_EVENT_SCORE]
    alpha = base + [_type_factor(group) for group in alpha_groups]
    alpha_risk = alpha + [_type_factor(group) for group in risk_groups]
    return {
        "NO_EVENT": _nonempty_factors(panel_z, base),
        "TOTAL_EVENT": _nonempty_factors(panel_z, total),
        "TYPE_ALPHA": _nonempty_factors(panel_z, alpha),
        "TYPE_ALPHA_RISK_AWARE": _nonempty_factors(panel_z, alpha_risk),
    }


def _run_strategy(
    panel_z: pd.DataFrame,
    prices: pd.DataFrame,
    long_df: pd.DataFrame,
    settings: Any,
    *,
    scenario: str,
    factors: list[str],
    include_equal: bool,
) -> dict[str, Any]:
    if not factors:
        raise ValueError("%s 没有可用因子" % scenario)
    equal_nav = pd.Series(dtype=float, name=f"EQUAL_{scenario}")
    equal_meta: dict[str, Any] = {}
    if include_equal:
        equal_fused = fuse_equal_weight_zscore(panel_z[factors])
        equal_nav, equal_meta = run_multi_backtest(
            fused=equal_fused,
            prices=prices,
            settings=settings,
            factor_name=f"EQUAL_{scenario}",
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
        factor_name=f"ROLLING_{scenario}",
        long_prices=long_df,
    )
    rolling_meta.update(rolling_fusion_meta)
    return {
        "factors": factors,
        "equal_nav": equal_nav.rename(f"EQUAL_{scenario}"),
        "equal_meta": equal_meta,
        "rolling_nav": rolling_nav.rename(f"ROLLING_{scenario}"),
        "rolling_meta": rolling_meta,
        "rolling_weight_log": rolling_weight_log,
    }


def _performance_rows(
    universe: str,
    nav_by_name: dict[str, pd.Series],
    prices: pd.DataFrame,
    settings: Any,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, nav in nav_by_name.items():
        bench = equal_weight_benchmark_nav(prices, dates=nav.index, price_col=settings.price_col)
        stats = summarize(nav, periods=settings.trading_days_per_year)
        excess = summarize_excess(nav, bench, periods=settings.trading_days_per_year)
        rows.append({"universe": universe, "strategy": name, **stats, **excess})
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


def _type_factor_coverage(panel_raw: pd.DataFrame, panel_z: pd.DataFrame, universe: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    type_cols = [c for c in panel_raw.columns if str(c).startswith(ANNOUNCEMENT_EVENT_TYPE_PREFIX)]
    for factor in type_cols:
        raw = panel_raw[factor]
        z = panel_z[factor] if factor in panel_z.columns else pd.Series(index=raw.index, dtype=float)
        total = int(raw.shape[0])
        raw_nonzero = raw.fillna(0.0).abs() > 1e-12
        z_nonzero = z.fillna(0.0).abs() > 1e-12
        rows.append(
            {
                "universe": universe,
                "factor": factor,
                "event_group": str(factor).replace(ANNOUNCEMENT_EVENT_TYPE_PREFIX, "", 1),
                "rows": total,
                "raw_nonzero_rows": int(raw_nonzero.sum()),
                "raw_nonzero_coverage": float(raw_nonzero.mean()) if total else 0.0,
                "zscore_nonzero_rows": int(z_nonzero.sum()),
                "zscore_nonzero_coverage": float(z_nonzero.mean()) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _prepare_universe(
    universe: dict[str, str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str], Any]:
    settings = get_settings()
    settings = replace(
        settings,
        stock_pool_path=Path(universe["stock_pool"]),
        tushare_price_cache_path=Path(universe["prices"]),
        fina_indicator_cache_path=Path(universe["fina"]) if universe.get("fina") else None,
        announcement_event_path=Path(universe["events"]),
        backtest_start=args.start,
        backtest_end=args.end,
        force_final_rebalance=True,
        persist_run_outputs=False,
    )
    long_df = load_prices_from_csv(Path(universe["prices"]))
    long_df = _filter_long_prices(long_df, args.start, args.end)
    long_df = _attach_industry_to_long_df(long_df, settings)
    prices = long_to_wide(long_df, settings.price_col)
    name_map = _stock_name_map(Path(universe["stock_pool"]))

    panel = build_four_factor_panel(prices, long_df, settings)
    groups = tuple(sorted(set(_parse_groups(args.alpha_groups) + _parse_groups(args.risk_groups))))
    events = load_announcement_events(Path(universe["events"]))
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
    return panel, panel_z, prices, long_df, name_map, settings


def _run_universe(universe: dict[str, str], args: argparse.Namespace, output_dir: Path) -> dict[str, pd.DataFrame]:
    name = str(universe["name"])
    panel_raw, panel_z, prices, long_df, name_map, settings = _prepare_universe(universe, args)
    factor_sets = _scenario_factor_sets(
        panel_z,
        alpha_groups=_parse_groups(args.alpha_groups),
        risk_groups=_parse_groups(args.risk_groups),
    )
    nav_by_name: dict[str, pd.Series] = {}
    rebalances: list[pd.DataFrame] = []
    factor_set_rows: list[dict[str, Any]] = []
    for scenario, factors in factor_sets.items():
        result = _run_strategy(
            panel_z,
            prices,
            long_df,
            settings,
            scenario=scenario,
            factors=factors,
            include_equal=bool(args.include_equal),
        )
        if bool(args.include_equal) and not result["equal_nav"].empty:
            nav_by_name[result["equal_nav"].name] = result["equal_nav"]
        nav_by_name[result["rolling_nav"].name] = result["rolling_nav"]
        rebalances.append(_rebalance_rows(result["rolling_meta"], f"ROLLING_{scenario}", name_map))
        result["rolling_weight_log"].to_csv(output_dir / name / f"rolling_factor_weight_log_{scenario.lower()}.csv", index=False)
        factor_set_rows.append(
            {
                "universe": name,
                "scenario": scenario,
                "n_factors": len(factors),
                "factors": ",".join(factors),
            }
        )

    uni_dir = output_dir / name
    uni_dir.mkdir(parents=True, exist_ok=True)
    nav_compare = pd.DataFrame(nav_by_name).sort_index()
    nav_compare.to_csv(uni_dir / "nav_compare.csv", index_label="date")
    plot_nav(nav_compare, title=f"{name} 公告类型因子组合回测", save_path=uni_dir / "nav_compare.png")
    perf = _performance_rows(name, nav_by_name, prices, settings)
    perf.to_csv(uni_dir / "performance_summary.csv", index=False)
    rb = pd.concat(rebalances, ignore_index=True) if rebalances else pd.DataFrame()
    rb.to_csv(uni_dir / "rebalance_log_rolling.csv", index=False)
    factor_sets_df = pd.DataFrame(factor_set_rows)
    factor_sets_df.to_csv(uni_dir / "factor_sets.csv", index=False)
    coverage = _type_factor_coverage(panel_raw, panel_z, name)
    coverage.to_csv(uni_dir / "type_factor_coverage.csv", index=False)
    return {"performance": perf, "factor_sets": factor_sets_df, "coverage": coverage}


def _scenario_mode(strategy: str) -> tuple[str, str]:
    text = str(strategy)
    if text.startswith("ROLLING_"):
        return "ROLLING", text.replace("ROLLING_", "", 1)
    if text.startswith("EQUAL_"):
        return "EQUAL", text.replace("EQUAL_", "", 1)
    return "", text


def _incremental_effect(perf: pd.DataFrame) -> pd.DataFrame:
    if perf.empty:
        return pd.DataFrame()
    frame = perf.copy()
    parsed = frame["strategy"].map(_scenario_mode)
    frame["mode"] = [x[0] for x in parsed]
    frame["scenario"] = [x[1] for x in parsed]
    rows: list[dict[str, Any]] = []
    for (universe, mode), sub in frame.groupby(["universe", "mode"]):
        base = sub[sub["scenario"] == "NO_EVENT"]
        if base.empty:
            continue
        base_row = base.iloc[0]
        for _, rec in sub[sub["scenario"] != "NO_EVENT"].iterrows():
            rows.append(
                {
                    "universe": universe,
                    "mode": mode,
                    "scenario": rec["scenario"],
                    "base_strategy": base_row["strategy"],
                    "strategy": rec["strategy"],
                    "base_final_nav": float(base_row["final_nav"]),
                    "final_nav": float(rec["final_nav"]),
                    "delta_final_nav": float(rec["final_nav"] - base_row["final_nav"]),
                    "base_total_return": float(base_row["total_return"]),
                    "total_return": float(rec["total_return"]),
                    "delta_total_return": float(rec["total_return"] - base_row["total_return"]),
                    "base_max_drawdown": float(base_row["max_drawdown"]),
                    "max_drawdown": float(rec["max_drawdown"]),
                    "delta_max_drawdown": float(rec["max_drawdown"] - base_row["max_drawdown"]),
                    "base_information_ratio": float(base_row["information_ratio"]),
                    "information_ratio": float(rec["information_ratio"]),
                    "delta_information_ratio": float(rec["information_ratio"] - base_row["information_ratio"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["universe", "mode", "scenario"]).reset_index(drop=True)


def _plot_incremental(incremental: pd.DataFrame, output_path: Path) -> None:
    if incremental.empty:
        return
    plot_df = incremental[incremental["mode"] == "ROLLING"].copy()
    if plot_df.empty:
        return
    pivot = plot_df.pivot(index="universe", columns="scenario", values="delta_total_return")
    plt = __import__("analysis.plotting", fromlist=["_pyplot_zh"])._pyplot_zh(output_path)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("公告类型因子组合回测：相对无公告基线的收益差")
    ax.set_xlabel("股票池")
    ax.set_ylabel("delta total return")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    warnings.filterwarnings("ignore", message="An input array is constant")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    perf_parts: list[pd.DataFrame] = []
    factor_parts: list[pd.DataFrame] = []
    coverage_parts: list[pd.DataFrame] = []
    for universe in args.universe:
        uni_dir = output_dir / str(universe["name"])
        uni_dir.mkdir(parents=True, exist_ok=True)
        result = _run_universe(universe, args, output_dir)
        perf_parts.append(result["performance"])
        factor_parts.append(result["factor_sets"])
        coverage_parts.append(result["coverage"])

    perf = pd.concat(perf_parts, ignore_index=True) if perf_parts else pd.DataFrame()
    factor_sets = pd.concat(factor_parts, ignore_index=True) if factor_parts else pd.DataFrame()
    coverage = pd.concat(coverage_parts, ignore_index=True) if coverage_parts else pd.DataFrame()
    incremental = _incremental_effect(perf)
    perf.to_csv(output_dir / "performance_summary.csv", index=False)
    factor_sets.to_csv(output_dir / "factor_sets.csv", index=False)
    coverage.to_csv(output_dir / "type_factor_coverage.csv", index=False)
    incremental.to_csv(output_dir / "incremental_effect.csv", index=False)
    _plot_incremental(incremental, output_dir / "rolling_incremental_return.png")

    print("performance_summary=%s" % (output_dir / "performance_summary.csv"))
    print("incremental_effect=%s" % (output_dir / "incremental_effect.csv"))
    if not incremental.empty:
        cols = [
            "universe",
            "mode",
            "scenario",
            "final_nav",
            "delta_total_return",
            "max_drawdown",
            "information_ratio",
        ]
        print(incremental[[c for c in cols if c in incremental.columns]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
