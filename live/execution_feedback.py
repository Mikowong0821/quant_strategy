"""
真实成交回填与执行偏差分析。

读取人工确认实盘单中回填的真实成交字段，比较系统建议订单和真实执行结果，
输出成交数量、成交价格、金额和状态差异。该模块不连接券商、不修改账户状态。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings


FEEDBACK_COLUMNS = [
    "date",
    "strategy",
    "symbol",
    "side",
    "suggested_qty",
    "executed_qty",
    "qty_diff",
    "suggested_price",
    "executed_price",
    "price_diff",
    "price_slippage_pct",
    "suggested_amount",
    "executed_amount",
    "amount_diff",
    "check_status",
    "manual_action",
    "execution_status",
    "operator",
    "confirmed_at",
    "execution_note",
]


SUMMARY_COLUMNS = [
    "date",
    "strategy",
    "n_orders",
    "n_filled",
    "n_partial",
    "n_not_executed",
    "n_blocked",
    "suggested_buy_amount",
    "executed_buy_amount",
    "suggested_sell_amount",
    "executed_sell_amount",
    "net_executed_cash_flow",
    "avg_abs_price_slippage_pct",
    "max_abs_price_slippage_pct",
]


def execution_feedback_dir(settings: Settings, strategy: str) -> Path:
    safe = str(strategy).replace("/", "_")
    return settings.output_dir / "execution_feedback" / safe


def _date_to_str(value: Any) -> str:
    if value is None or value == "":
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(out):
        return default
    return out


def _execution_status(row: pd.Series) -> str:
    check_status = str(row.get("check_status", "")).upper()
    action = str(row.get("manual_action", "")).upper()
    suggested_qty = abs(_num(row.get("suggested_qty", 0.0)))
    executed_qty = abs(_num(row.get("executed_qty", 0.0)))
    executed_price = _num(row.get("executed_price", 0.0))
    if check_status == "BLOCK" or action == "DO_NOT_EXECUTE":
        return "BLOCKED"
    if executed_qty <= 0 or executed_price <= 0:
        return "NOT_EXECUTED"
    if suggested_qty > 0 and executed_qty < suggested_qty:
        return "PARTIAL"
    if suggested_qty > 0 and executed_qty > suggested_qty:
        return "OVERFILLED"
    return "FILLED"


def build_execution_feedback(manual_confirmation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """由人工确认单生成逐笔执行偏差和汇总表。"""
    if manual_confirmation.empty:
        empty_detail = pd.DataFrame(columns=FEEDBACK_COLUMNS)
        empty_summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
        return empty_detail, empty_summary

    required = {"date", "strategy", "symbol", "side", "delta_shares", "price", "estimated_amount"}
    missing = required - set(manual_confirmation.columns)
    if missing:
        raise ValueError("人工确认单缺少必要列: %s" % ", ".join(sorted(missing)))

    rows: list[dict[str, Any]] = []
    for rec in manual_confirmation.to_dict("records"):
        side = str(rec.get("side", "")).upper()
        suggested_qty = abs(int(round(_num(rec.get("delta_shares", 0.0)))))
        executed_qty = abs(int(round(_num(rec.get("executed_qty", 0.0)))))
        suggested_price = _num(rec.get("price", 0.0))
        executed_price = _num(rec.get("executed_price", 0.0))
        suggested_amount = abs(_num(rec.get("estimated_amount", suggested_qty * suggested_price)))
        executed_amount = executed_qty * executed_price if executed_qty > 0 and executed_price > 0 else 0.0
        price_diff = executed_price - suggested_price if executed_amount > 0 else 0.0
        price_slippage_pct = price_diff / suggested_price if suggested_price > 0 and executed_amount > 0 else 0.0
        row = {
            "date": _date_to_str(rec.get("date", "")),
            "strategy": str(rec.get("strategy", "")),
            "symbol": str(rec.get("symbol", "")),
            "side": side,
            "suggested_qty": suggested_qty,
            "executed_qty": executed_qty,
            "qty_diff": executed_qty - suggested_qty,
            "suggested_price": suggested_price,
            "executed_price": executed_price,
            "price_diff": price_diff,
            "price_slippage_pct": price_slippage_pct,
            "suggested_amount": suggested_amount,
            "executed_amount": executed_amount,
            "amount_diff": executed_amount - suggested_amount,
            "check_status": str(rec.get("check_status", "")),
            "manual_action": str(rec.get("manual_action", "")),
            "execution_status": "",
            "operator": str(rec.get("operator", "")),
            "confirmed_at": str(rec.get("confirmed_at", "")),
            "execution_note": str(rec.get("execution_note", "")),
        }
        row["execution_status"] = _execution_status(pd.Series(row))
        rows.append(row)

    detail = pd.DataFrame(rows, columns=FEEDBACK_COLUMNS)
    dates = detail["date"].dropna().astype(str)
    strategies = detail["strategy"].dropna().astype(str)
    buy = detail[detail["side"] == "BUY"]
    sell = detail[detail["side"] == "SELL"]
    executed_buy = float(buy["executed_amount"].sum()) if not buy.empty else 0.0
    executed_sell = float(sell["executed_amount"].sum()) if not sell.empty else 0.0
    filled_mask = detail["execution_status"].isin(["FILLED", "OVERFILLED"])
    slippage = detail.loc[detail["executed_amount"] > 0, "price_slippage_pct"].astype(float).abs()
    summary = pd.DataFrame(
        [
            {
                "date": dates.iloc[0] if len(dates) else "",
                "strategy": strategies.iloc[0] if len(strategies) else "",
                "n_orders": int(len(detail)),
                "n_filled": int(filled_mask.sum()),
                "n_partial": int((detail["execution_status"] == "PARTIAL").sum()),
                "n_not_executed": int((detail["execution_status"] == "NOT_EXECUTED").sum()),
                "n_blocked": int((detail["execution_status"] == "BLOCKED").sum()),
                "suggested_buy_amount": float(buy["suggested_amount"].sum()) if not buy.empty else 0.0,
                "executed_buy_amount": executed_buy,
                "suggested_sell_amount": float(sell["suggested_amount"].sum()) if not sell.empty else 0.0,
                "executed_sell_amount": executed_sell,
                "net_executed_cash_flow": executed_sell - executed_buy,
                "avg_abs_price_slippage_pct": float(slippage.mean()) if not slippage.empty else 0.0,
                "max_abs_price_slippage_pct": float(slippage.max()) if not slippage.empty else 0.0,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )
    return detail, summary


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 30) -> str:
    if frame.empty:
        return "无\n"
    rows = frame.head(max_rows)
    lines = [
        "| " + " | ".join(rows.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(rows.columns)) + " |",
    ]
    for rec in rows.to_dict("records"):
        vals: list[str] = []
        for col in rows.columns:
            value = rec.get(col, "")
            if isinstance(value, float):
                vals.append("%.4f" % value)
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    if len(frame) > max_rows:
        lines.append("")
        lines.append("仅展示前 %d 行，共 %d 行。" % (max_rows, len(frame)))
    return "\n".join(lines) + "\n"


def build_execution_feedback_report(detail: pd.DataFrame, summary: pd.DataFrame) -> str:
    """生成真实成交回填与执行偏差 Markdown 报告。"""
    rec = summary.iloc[0].to_dict() if not summary.empty else {}
    strategy = str(rec.get("strategy", ""))
    date_s = str(rec.get("date", ""))
    lines = [
        "# 真实成交回填与执行偏差分析 - %s - %s" % (strategy, date_s),
        "",
        "## 摘要",
        "",
        "- 订单数：%s" % rec.get("n_orders", 0),
        "- 完全成交：%s" % rec.get("n_filled", 0),
        "- 部分成交：%s" % rec.get("n_partial", 0),
        "- 未执行：%s" % rec.get("n_not_executed", 0),
        "- 阻断：%s" % rec.get("n_blocked", 0),
        "- 建议买入金额：%.2f" % float(rec.get("suggested_buy_amount", 0.0) or 0.0),
        "- 实际买入金额：%.2f" % float(rec.get("executed_buy_amount", 0.0) or 0.0),
        "- 建议卖出金额：%.2f" % float(rec.get("suggested_sell_amount", 0.0) or 0.0),
        "- 实际卖出金额：%.2f" % float(rec.get("executed_sell_amount", 0.0) or 0.0),
        "- 实际净现金流：%.2f" % float(rec.get("net_executed_cash_flow", 0.0) or 0.0),
        "- 平均绝对滑点：%.4f%%" % (float(rec.get("avg_abs_price_slippage_pct", 0.0) or 0.0) * 100.0),
        "- 最大绝对滑点：%.4f%%" % (float(rec.get("max_abs_price_slippage_pct", 0.0) or 0.0) * 100.0),
        "",
        "## 逐笔执行偏差",
        "",
        _markdown_table(detail),
        "",
        "## 说明",
        "",
        "`price_slippage_pct = (executed_price - suggested_price) / suggested_price`。",
        "",
        "买入时正滑点通常表示成交价高于建议价；卖出时正滑点通常表示成交价高于建议价。该表只记录执行偏差，不判断交易是否应该发生。",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def save_execution_feedback(
    settings: Settings,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Path]:
    """保存执行偏差 CSV 与 Markdown 报告。"""
    if summary.empty:
        strategy = "UNKNOWN"
        date_s = "unknown"
    else:
        rec = summary.iloc[0].to_dict()
        strategy = str(rec.get("strategy", "UNKNOWN") or "UNKNOWN")
        date_s = str(rec.get("date", "unknown") or "unknown")
    base = execution_feedback_dir(settings, strategy)
    base.mkdir(parents=True, exist_ok=True)
    detail_path = base / ("%s_execution_feedback.csv" % date_s)
    summary_path = base / ("%s_execution_summary.csv" % date_s)
    report_path = base / ("%s_execution_feedback.md" % date_s)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    report_path.write_text(build_execution_feedback_report(detail, summary), encoding="utf-8")
    return {
        "detail": detail_path,
        "summary": summary_path,
        "report": report_path,
    }
