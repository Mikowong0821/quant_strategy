"""
纸面交易日报：把单日纸面运行结果整理成 Markdown。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from config import Settings


def paper_report_dir(settings: Settings, strategy: str) -> Path:
    safe = str(strategy).replace("/", "_")
    return settings.output_dir / "paper_reports" / safe


def _date_to_str(value: Any) -> str:
    if value is None or value == "":
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _fmt_money(value: Any) -> str:
    try:
        return "%.2f" % float(value)
    except (TypeError, ValueError):
        return ""


def _fmt_float(value: Any, digits: int = 4) -> str:
    try:
        return ("%." + str(digits) + "f") % float(value)
    except (TypeError, ValueError):
        return ""


def _read_previous_snapshot(result: dict[str, Any]) -> dict[str, Any] | None:
    paths = result.get("paths", {})
    account_paths = paths.get("account_state") if isinstance(paths, dict) else None
    if not isinstance(account_paths, dict):
        return None
    snapshots_path = account_paths.get("snapshots")
    if snapshots_path is None or not Path(snapshots_path).exists():
        return None
    snapshots = pd.read_csv(snapshots_path)
    if snapshots.empty or "date" not in snapshots.columns:
        return None
    trade_date = pd.Timestamp(result["trade_date"])
    snapshots = snapshots.copy()
    snapshots["date"] = pd.to_datetime(snapshots["date"], errors="coerce")
    prev = snapshots[snapshots["date"] < trade_date].sort_values("date")
    if prev.empty:
        return None
    return prev.iloc[-1].to_dict()


def _markdown_table(
    frame: pd.DataFrame,
    columns: Iterable[str],
    headers: Iterable[str] | None = None,
    *,
    max_rows: int = 20,
) -> str:
    cols = [c for c in columns if c in frame.columns]
    if frame.empty or not cols:
        return "无\n"
    headers_l = list(headers) if headers is not None else cols
    if len(headers_l) != len(cols):
        headers_l = cols
    rows = frame.loc[:, cols].head(max_rows)

    lines = [
        "| " + " | ".join(str(x) for x in headers_l) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for rec in rows.to_dict("records"):
        vals: list[str] = []
        for col in cols:
            value = rec.get(col, "")
            if isinstance(value, float):
                vals.append(_fmt_float(value))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    if len(frame) > max_rows:
        lines.append("")
        lines.append("仅展示前 %d 行，共 %d 行。" % (max_rows, len(frame)))
    return "\n".join(lines) + "\n"


def build_daily_paper_report(result: dict[str, Any]) -> str:
    """根据 `run_daily_paper_trade` / `run_daily_paper_from_outputs` 结果生成 Markdown。"""
    strategy = str(result["strategy"])
    trade_date = _date_to_str(result["trade_date"])
    target_date = _date_to_str(result.get("target_date", result["trade_date"]))
    price_date = _date_to_str(result.get("price_date", result["trade_date"]))
    orders = result["orders"]
    checks = result["order_checks"]
    trades = result["paper_trades"]
    positions = result["positions"]
    snapshot = result["account_snapshot"]
    broker_orders = result.get("broker_orders", pd.DataFrame())
    execution_mode = str(result.get("execution_mode", "paper_trading"))
    previous = _read_previous_snapshot(result)

    n_orders = int(len(orders))
    n_pass = int((checks["check_status"] == "PASS").sum()) if not checks.empty else 0
    n_block = int((checks["check_status"] == "BLOCK").sum()) if not checks.empty else 0
    n_filled = int((trades["fill_status"] == "FILLED").sum()) if not trades.empty else 0
    n_skipped = int((trades["fill_status"] == "SKIPPED").sum()) if not trades.empty else 0
    guard_issues = list(result.get("guard_issues", []) or [])

    lines: list[str] = [
        "# 纸面交易日报 - %s - %s" % (strategy, trade_date),
        "",
        "## 运行摘要",
        "",
        "- 策略：`%s`" % strategy,
        "- 执行模式：`%s`" % execution_mode,
        "- 运行日期：%s" % trade_date,
        "- 目标权重日期：%s" % target_date,
        "- 价格日期：%s" % price_date,
        "- 订单数：%d" % n_orders,
        "- 预检查通过：%d" % n_pass,
        "- 预检查阻断：%d" % n_block,
        "- 成交：%d" % n_filled,
        "- 跳过：%d" % n_skipped,
        "",
        "## 账户快照",
        "",
        "- 现金：%s" % _fmt_money(snapshot.get("cash", 0.0)),
        "- 持仓市值：%s" % _fmt_money(snapshot.get("market_value", 0.0)),
        "- 总资产：%s" % _fmt_money(snapshot.get("total_asset", 0.0)),
        "- 持仓数量：%d" % int(float(snapshot.get("n_positions", 0.0))),
        "",
    ]

    if guard_issues:
        lines.extend(["## 运行检查", ""])
        for issue in guard_issues:
            lines.append("- [%s] `%s`：%s" % (issue.severity, issue.code, issue.message))
        lines.append("")

    if previous is not None:
        prev_asset = float(previous.get("total_asset", 0.0))
        cur_asset = float(snapshot.get("total_asset", 0.0))
        diff = cur_asset - prev_asset
        ret = diff / prev_asset if abs(prev_asset) > 1e-12 else 0.0
        lines.extend(
            [
                "## 较上一快照变化",
                "",
                "- 上一快照日期：%s" % _date_to_str(previous.get("date")),
                "- 上一总资产：%s" % _fmt_money(prev_asset),
                "- 资产变化：%s" % _fmt_money(diff),
                "- 资产变化率：%.2f%%" % (ret * 100.0),
                "",
            ]
        )
    else:
        lines.extend(["## 较上一快照变化", "", "暂无上一快照。", ""])

    blocked = checks[checks["check_status"] == "BLOCK"] if not checks.empty else checks
    lines.extend(
        [
            "## 今日订单",
            "",
            _markdown_table(
                orders,
                ["symbol", "side", "delta_shares", "price", "estimated_amount", "target_weight", "trade_reason"],
                ["标的", "方向", "股数变化", "价格", "预估金额", "目标权重", "原因"],
            ),
            "## 被阻断订单",
            "",
            _markdown_table(
                blocked,
                ["symbol", "side", "delta_shares", "estimated_amount", "check_reason"],
                ["标的", "方向", "股数变化", "预估金额", "阻断原因"],
            ),
            "## 纸面成交",
            "",
            _markdown_table(
                trades,
                ["symbol", "side", "qty", "price", "gross_amount", "commission", "cash_after", "fill_status", "fill_reason"],
                ["标的", "方向", "数量", "价格", "成交金额", "手续费", "成交后现金", "状态", "原因"],
            ),
            "## 券商订单回报",
            "",
            _markdown_table(
                broker_orders,
                ["symbol", "side", "qty", "price", "status", "reason", "filled_qty", "avg_price", "cash_after"],
                ["标的", "方向", "数量", "价格", "状态", "原因", "成交数量", "均价", "成交后现金"],
            ),
            "## 当前持仓",
            "",
            _markdown_table(
                positions,
                ["symbol", "shares", "available_shares", "updated_at"],
                ["标的", "持仓股数", "可用股数", "更新时间"],
            ),
        ]
    )

    paths = result.get("paths", {})
    if paths:
        lines.extend(["## 输出文件", ""])
        for key, value in paths.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    lines.append("- `%s.%s`：%s" % (key, sub_key, sub_value))
            else:
                lines.append("- `%s`：%s" % (key, value))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def save_daily_paper_report(settings: Settings, result: dict[str, Any]) -> Path:
    """保存纸面交易日报到 output/paper_reports/<strategy>/<date>.md。"""
    strategy = str(result["strategy"])
    trade_date = _date_to_str(result["trade_date"])
    base = paper_report_dir(settings, strategy)
    base.mkdir(parents=True, exist_ok=True)
    path = base / ("%s.md" % trade_date)
    path.write_text(build_daily_paper_report(result), encoding="utf-8")
    return path
