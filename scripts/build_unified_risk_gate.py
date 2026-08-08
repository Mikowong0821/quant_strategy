#!/usr/bin/env python3
"""合并公告风险、负面舆情和人工黑名单，输出统一风险门禁。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from live.event_risk_filter import build_event_risk_candidates, load_event_risk_candidates
from live.negative_sentiment_filter import build_negative_sentiment_candidates, load_sentiment_items
from live.risk_blacklist import load_risk_blacklist
from live.risk_gate import build_unified_risk_gate, risk_gate_to_blacklist, summarize_risk_gate_for_report
from live.stock_pool import load_stock_pool_frame


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成公告 + 舆情 + 人工黑名单统一风险门禁")
    parser.add_argument("--trade-date", required=True, help="门禁评估日期")
    parser.add_argument("--stock-pool", type=Path, default=Path("data/stock_pool_ftse_china_a50_20260710.csv"))
    parser.add_argument("--events", type=Path, default=Path("data/announcement_events_a50_20260701_20260726.csv"))
    parser.add_argument("--sentiment", type=Path, default=Path("data/news_sentiment_a50_20260701_20260726.csv"))
    parser.add_argument("--manual-risk", type=Path, default=Path("data/risk_blacklist.csv"))
    parser.add_argument("--event-lookback-days", type=int, default=60)
    parser.add_argument("--event-block-days", type=int, default=20)
    parser.add_argument("--event-watch-days", type=int, default=10)
    parser.add_argument("--sentiment-lookback-days", type=int, default=7)
    parser.add_argument("--sentiment-block-days", type=int, default=10)
    parser.add_argument("--sentiment-watch-days", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--write-blacklist", action="store_true")
    parser.add_argument("--include-watch-in-blacklist", action="store_true")
    return parser


def _load_events(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return load_event_risk_candidates(
        path,
        as_of_date=args.trade_date,
        lookback_days=args.event_lookback_days,
        block_days=args.event_block_days,
        watch_days=args.event_watch_days,
    )


def _load_sentiment(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    items = load_sentiment_items(path)
    return build_negative_sentiment_candidates(
        items,
        as_of_date=args.trade_date,
        lookback_days=args.sentiment_lookback_days,
        block_days=args.sentiment_block_days,
        watch_days=args.sentiment_watch_days,
    )


def _write_summary(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    gate: pd.DataFrame,
    details: pd.DataFrame,
    blacklist: pd.DataFrame | None,
) -> None:
    report_status, report_detail = summarize_risk_gate_for_report(gate)
    status_counts = gate["gate_status"].astype(str).value_counts().to_dict() if not gate.empty else {}
    source_counts = details["source"].astype(str).value_counts().head(10).to_dict() if not details.empty else {}
    top = gate[gate["gate_status"].isin(["BLOCK", "WATCH"])].head(12)

    lines = [
        "# 统一风险门禁运行摘要",
        "",
        "## 运行范围",
        "",
        "- 门禁日期：%s" % args.trade_date,
        "- 股票池：%s" % args.stock_pool,
        "- 公告风险候选：%s" % args.events,
        "- 舆情风险候选：%s" % args.sentiment,
        "- 人工黑名单：%s" % args.manual_risk,
        "",
        "## 门禁结果",
        "",
        "- 总状态：%s" % report_status,
        "- 明细：%s" % report_detail,
        "- 状态分布：%s" % status_counts,
        "- 来源分布：%s" % source_counts,
    ]
    if blacklist is not None:
        lines.append("- 导出黑名单：%d 只" % int(len(blacklist)))
    lines.extend(["", "## 风险命中样例", ""])
    if top.empty:
        lines.append("没有 BLOCK / WATCH 命中。")
    else:
        display_cols = ["symbol", "name", "gate_status", "risk_count", "sources", "reason", "expires_at"]
        lines.append(top[display_cols].to_markdown(index=False))
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    settings = get_settings()
    output_dir = args.output_dir or (settings.output_dir / "unified_risk_gate")
    output_dir.mkdir(parents=True, exist_ok=True)

    pool = load_stock_pool_frame(args.stock_pool)
    manual = load_risk_blacklist(args.manual_risk)
    event_candidates = _load_events(args.events, args)
    sentiment_candidates = _load_sentiment(args.sentiment, args)

    gate, details = build_unified_risk_gate(
        trade_date=args.trade_date,
        symbols=pool,
        manual_blacklist=manual,
        event_candidates=event_candidates,
        sentiment_candidates=sentiment_candidates,
        include_pass=True,
    )

    tag = str(args.trade_date).replace("-", "")
    gate_path = output_dir / ("risk_gate_%s.csv" % tag)
    detail_path = output_dir / ("risk_gate_details_%s.csv" % tag)
    gate.to_csv(gate_path, index=False)
    details.to_csv(detail_path, index=False)
    print("risk_gate=%s" % gate_path)
    print("risk_gate_details=%s" % detail_path)

    blacklist = None
    if args.write_blacklist:
        blacklist = risk_gate_to_blacklist(gate, include_watch=args.include_watch_in_blacklist)
        blacklist_path = output_dir / ("risk_blacklist_%s.csv" % tag)
        blacklist.to_csv(blacklist_path, index=False)
        print("risk_blacklist=%s" % blacklist_path)
        print("blacklist_size=%d" % int(len(blacklist)))

    _write_summary(output_dir, args=args, gate=gate, details=details, blacklist=blacklist)
    status, detail = summarize_risk_gate_for_report(gate)
    print("status=%s detail=%s" % (status, detail))
    print("event_candidates=%d sentiment_candidates=%d manual_rows=%d" % (len(event_candidates), len(sentiment_candidates), len(manual)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
