#!/usr/bin/env python3
"""生成容量与冲击成本估算报告。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from live.capacity_impact import (
    default_capacity_rules,
    evaluate_capacity_impact,
    load_capacity_rules,
    load_liquidity_history,
    summarize_capacity_impact,
)


DEFAULT_STRATEGY = "FUSED_ROLLING_SCORE_WEIGHTED"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按订单计划估算容量、参与率和冲击成本")
    parser.add_argument("--trade-date", required=True, help="检查日期")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, help="策略名")
    parser.add_argument("--orders", type=Path, required=True, help="订单计划 CSV，含 symbol/side/estimated_amount")
    parser.add_argument("--liquidity-history", type=Path, default=None, help="流动性历史 CSV，默认 output/cache/prices_long.csv")
    parser.add_argument("--rules", type=Path, default=None, help="容量规则 CSV")
    parser.add_argument("--lookback-days", type=int, default=20, help="平均成交额窗口")
    parser.add_argument("--impact-coefficient-bps", type=float, default=100.0, help="冲击成本估算系数")
    parser.add_argument("--write-default-rules", action="store_true", help="同时输出默认规则模板")
    parser.add_argument("--output-dir", type=Path, default=None, help="输出目录，默认 output/capacity_impact/<strategy>")
    return parser


def _markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str], max_rows: int = 30) -> str:
    cols = [c for c in columns if c in frame.columns]
    if frame.empty or not cols:
        return "无\n"
    if len(headers) != len(cols):
        headers = cols
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for rec in frame.loc[:, cols].head(max_rows).to_dict("records"):
        vals: list[str] = []
        for col in cols:
            value = rec.get(col, "")
            if isinstance(value, float):
                vals.append("%.4f" % value)
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    if len(frame) > max_rows:
        lines.extend(["", "仅展示前 %d 行，共 %d 行。" % (max_rows, len(frame))])
    return "\n".join(lines) + "\n"


def _write_summary(path: Path, *, args: argparse.Namespace, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    status, text = summarize_capacity_impact(summary)
    lines = [
        "# 容量与冲击成本估算",
        "",
        "## 运行范围",
        "",
        "- 策略：`%s`" % args.strategy,
        "- 检查日期：%s" % args.trade_date,
        "- 订单计划：%s" % args.orders,
        "- 流动性历史：%s" % (args.liquidity_history or get_settings().output_dir / "cache" / "prices_long.csv"),
        "- 平均成交额窗口：%d" % int(args.lookback_days),
        "- 冲击成本系数：%.2f bps" % float(args.impact_coefficient_bps),
        "",
        "## 总结",
        "",
        "- 总状态：`%s`" % status,
        "- 摘要：%s" % text,
        "",
        _markdown_table(
            summary,
            [
                "status",
                "n_orders",
                "n_with_liquidity",
                "n_missing_liquidity",
                "max_participation_rate",
                "total_order_amount",
                "estimated_impact_cost_amount",
                "estimated_impact_cost_bps",
                "portfolio_capacity_multiplier_at_warning",
            ],
            ["状态", "订单数", "有流动性数据", "缺流动性数据", "最大参与率", "订单总金额", "冲击成本", "冲击bps", "容量倍数"],
        ),
        "## 逐订单明细",
        "",
        _markdown_table(
            detail,
            [
                "symbol",
                "side",
                "estimated_amount",
                "avg_amount",
                "participation_rate",
                "impact_cost_bps",
                "impact_cost_amount",
                "capacity_multiplier_at_warning",
                "status",
                "details",
            ],
            ["标的", "方向", "订单金额", "平均成交额", "参与率", "冲击bps", "冲击金额", "容量倍数", "状态", "说明"],
            max_rows=50,
        ),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    settings = get_settings()
    liquidity_path = args.liquidity_history or settings.output_dir / "cache" / "prices_long.csv"
    orders = pd.read_csv(args.orders)
    detail, summary = evaluate_capacity_impact(
        orders,
        load_liquidity_history(liquidity_path),
        trade_date=args.trade_date,
        rules=load_capacity_rules(args.rules),
        lookback_days=args.lookback_days,
        impact_coefficient_bps=args.impact_coefficient_bps,
    )
    output_dir = args.output_dir or settings.output_dir / "capacity_impact" / args.strategy.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = str(args.trade_date).replace("-", "")
    detail_path = output_dir / ("capacity_impact_detail_%s.csv" % tag)
    summary_path = output_dir / ("capacity_impact_summary_%s.csv" % tag)
    report_path = output_dir / ("capacity_impact_summary_%s.md" % tag)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    _write_summary(report_path, args=args, detail=detail, summary=summary)
    if args.write_default_rules:
        default_capacity_rules().to_csv(output_dir / "capacity_rules_template.csv", index=False)
    status, text = summarize_capacity_impact(summary)
    print("capacity_impact_detail=%s" % detail_path)
    print("capacity_impact_summary=%s" % summary_path)
    print("capacity_impact_report=%s" % report_path)
    print("status=%s detail=%s" % (status, text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
