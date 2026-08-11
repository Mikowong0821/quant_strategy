"""生成组合压力测试与情景分析报告。"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from config import Settings, get_settings
from live.daily_paper_cli import DEFAULT_STRATEGY, load_latest_target_weights, load_optional_csv
from live.stress_test import (
    default_stress_scenarios,
    load_stress_scenarios,
    run_portfolio_stress_tests,
    summarize_stress_tests,
)


def _write_summary(path: Path, checks: pd.DataFrame) -> None:
    status, detail = summarize_stress_tests(checks)
    lines = [
        "# 组合压力测试摘要",
        "",
        "- 整体状态：`%s`" % status,
        "- 摘要：%s" % detail,
        "",
    ]
    if not checks.empty:
        display = checks.copy()
        cols = [
            "scenario_id",
            "category",
            "status",
            "shock_value",
            "affected_weight",
            "estimated_loss_pct",
            "estimated_loss_amount",
            "details",
            "action",
        ]
        lines.extend(
            [
                "| 情景 | 类别 | 状态 | 冲击 | 受影响权重 | 预估损失率 | 预估损失金额 | 说明 | 处理动作 |",
                "|---|---|---|---:|---:|---:|---:|---|---|",
            ]
        )
        for rec in display[cols].to_dict("records"):
            lines.append(
                "| {scenario_id} | {category} | {status} | {shock_value:.4f} | {affected_weight:.4f} | {estimated_loss_pct:.4f} | {estimated_loss_amount:.2f} | {details} | {action} |".format(
                    **{
                        **rec,
                        "shock_value": float(rec["shock_value"]) if pd.notna(rec["shock_value"]) else float("nan"),
                        "affected_weight": float(rec["affected_weight"]) if pd.notna(rec["affected_weight"]) else float("nan"),
                        "estimated_loss_pct": float(rec["estimated_loss_pct"]) if pd.notna(rec["estimated_loss_pct"]) else float("nan"),
                        "estimated_loss_amount": float(rec["estimated_loss_amount"])
                        if pd.notna(rec["estimated_loss_amount"])
                        else float("nan"),
                    }
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_portfolio_stress_tests(
    settings: Settings,
    *,
    strategy: str,
    trade_date: Any = None,
    rebalance_log_path: Path | None = None,
    scenarios_path: Path | None = None,
    industry_path: Path | None = None,
    total_asset: float = 1_000_000.0,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    rebalance_path = (
        rebalance_log_path
        if rebalance_log_path is not None
        else settings.output_dir / "rebalance_logs" / ("%s.csv" % strategy.replace("/", "_"))
    )
    target_date, target_weights = load_latest_target_weights(rebalance_path, trade_date=trade_date)
    run_date = pd.Timestamp(trade_date) if trade_date is not None else target_date
    scenarios = load_stress_scenarios(str(scenarios_path) if scenarios_path is not None else None)
    industry = load_optional_csv(industry_path)
    checks = run_portfolio_stress_tests(
        scenarios,
        target_weights,
        trade_date=run_date,
        total_asset=total_asset,
        industry=industry,
    )

    safe_strategy = str(strategy).replace("/", "_")
    base = output_dir or (settings.output_dir / "stress_tests" / safe_strategy)
    base.mkdir(parents=True, exist_ok=True)
    tag = pd.Timestamp(run_date).strftime("%Y%m%d")
    checks_path = base / ("portfolio_stress_tests_%s.csv" % tag)
    summary_path = base / ("portfolio_stress_test_summary_%s.md" % tag)
    checks.to_csv(checks_path, index=False)
    _write_summary(summary_path, checks)
    if scenarios_path is None:
        default_stress_scenarios().to_csv(base / "stress_scenarios_template.csv", index=False)
    return {"checks": checks_path, "summary": summary_path}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成组合压力测试与情景分析")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, help="策略名")
    parser.add_argument("--trade-date", default=None, help="压力测试日期；默认使用最近目标权重日期")
    parser.add_argument("--rebalance-log", type=Path, default=None, help="调仓日志 CSV；默认 output/rebalance_logs/<strategy>.csv")
    parser.add_argument("--scenarios", type=Path, default=None, help="压力测试情景表 CSV；默认使用内置情景")
    parser.add_argument("--industry", type=Path, default=None, help="行业映射 CSV，用于行业冲击情景")
    parser.add_argument("--total-asset", type=float, default=1_000_000.0, help="用于估算损失金额的组合资产")
    parser.add_argument("--output-dir", type=Path, default=None, help="输出目录，默认 output/stress_tests/<strategy>")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = build_portfolio_stress_tests(
        get_settings(),
        strategy=args.strategy,
        trade_date=args.trade_date,
        rebalance_log_path=args.rebalance_log,
        scenarios_path=args.scenarios,
        industry_path=args.industry,
        total_asset=args.total_asset,
        output_dir=args.output_dir,
    )
    for key, value in paths.items():
        print("%s=%s" % (key, value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
