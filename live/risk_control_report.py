"""风险总控日报：汇总日终纸面交易中的关键风控状态。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from live.capacity_impact import summarize_capacity_impact
from live.drawdown_control import summarize_drawdown_control
from live.paper_guard import GuardIssue
from live.risk_blacklist import summarize_risk_blacklist_for_report
from live.risk_gate import summarize_risk_gate_for_report
from live.risk_limits import summarize_risk_limit_checks
from live.stress_test import summarize_stress_tests


RISK_CONTROL_COLUMNS = [
    "trade_date",
    "module",
    "status",
    "severity_rank",
    "summary",
    "action",
]

_STATUS_RANK = {"BLOCK": 0, "ERROR": 0, "WATCH": 1, "WARNING": 1, "NA": 2, "PASS": 3, "OK": 3}


@dataclass(frozen=True)
class RiskControlItem:
    """风险总控日报的一行。"""

    trade_date: str
    module: str
    status: str
    severity_rank: int
    summary: str
    action: str


def _normalize_status(status: Any) -> str:
    text = str(status or "NA").strip().upper()
    if text == "ERROR":
        return "BLOCK"
    if text == "WARNING":
        return "WATCH"
    if text == "OK":
        return "PASS"
    if text in {"BLOCK", "WATCH", "PASS", "NA"}:
        return text
    return "NA"


def _rank(status: Any) -> int:
    return int(_STATUS_RANK.get(str(status or "NA").strip().upper(), 2))


def _action_for_status(status: str) -> str:
    if status == "BLOCK":
        return "暂停自动执行，先人工复核或重新生成订单。"
    if status == "WATCH":
        return "允许继续纸面观察，但需要人工确认风险来源。"
    if status == "NA":
        return "补齐缺失输入后再判断，不能当作通过。"
    return "无需额外动作，继续监控。"


def summarize_guard_issues_for_report(issues: list[GuardIssue] | None) -> tuple[str, str]:
    """把运行检查 issue 压缩成总控状态。"""
    rows = list(issues or [])
    if not rows:
        return "PASS", "运行检查无 ERROR/WARNING"
    errors = [x for x in rows if str(x.severity).upper() == "ERROR"]
    warnings = [x for x in rows if str(x.severity).upper() == "WARNING"]
    if errors:
        return "BLOCK", "ERROR=%d; WARNING=%d; %s" % (
            len(errors),
            len(warnings),
            "; ".join("%s:%s" % (x.code, x.message) for x in errors[:3]),
        )
    return "WATCH", "WARNING=%d; %s" % (
        len(warnings),
        "; ".join("%s:%s" % (x.code, x.message) for x in warnings[:3]),
    )


def summarize_order_checks_for_report(checks: pd.DataFrame | None) -> tuple[str, str]:
    """把订单预检查结果压缩成总控状态。"""
    if checks is None or checks.empty or "check_status" not in checks.columns:
        return "NA", "未找到订单预检查结果"
    status = checks["check_status"].astype(str).str.upper()
    block = int((status == "BLOCK").sum())
    passed = int((status == "PASS").sum())
    total = int(len(checks))
    if block > 0:
        reasons = ""
        if "check_reason" in checks.columns:
            reasons = "; ".join(
                str(x)
                for x in checks.loc[status == "BLOCK", "check_reason"].dropna().astype(str).head(3).tolist()
            )
        return "BLOCK", "BLOCK=%d; PASS=%d; total=%d; %s" % (block, passed, total, reasons)
    return "PASS", "全部通过；PASS=%d; total=%d" % (passed, total)


def _promote_na_status(status: Any, summary: str, frame: pd.DataFrame | None) -> tuple[str, str]:
    """总控层更保守：子模块明细里有 NA 时，不能把整体当成完全通过。"""
    normalized = _normalize_status(status)
    if normalized != "PASS" or frame is None or frame.empty or "status" not in frame.columns:
        return normalized, summary
    statuses = frame["status"].astype(str).str.upper()
    if int((statuses == "NA").sum()) > 0:
        return "NA", summary
    return normalized, summary


def build_risk_control_report(
    *,
    trade_date: Any,
    guard_issues: list[GuardIssue] | None = None,
    risk_gate: pd.DataFrame | None = None,
    risk_blacklist: pd.DataFrame | None = None,
    drawdown_control: pd.DataFrame | None = None,
    capacity_impact_summary: pd.DataFrame | None = None,
    order_checks: pd.DataFrame | None = None,
    risk_limit_checks: pd.DataFrame | None = None,
    stress_tests: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """构建风险总控日报明细表。"""
    date_s = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    risk_limit_status, risk_limit_summary = _promote_na_status(
        *summarize_risk_limit_checks(risk_limit_checks),
        risk_limit_checks,
    )
    stress_status, stress_summary = _promote_na_status(
        *summarize_stress_tests(stress_tests),
        stress_tests,
    )
    specs = [
        ("运行检查", *summarize_guard_issues_for_report(guard_issues)),
        ("统一风险门禁", *summarize_risk_gate_for_report(risk_gate)),
        ("风险黑名单", *summarize_risk_blacklist_for_report(risk_blacklist)),
        ("回撤止损与降仓", *summarize_drawdown_control(drawdown_control)),
        ("容量与冲击成本", *summarize_capacity_impact(capacity_impact_summary)),
        ("订单预检查", *summarize_order_checks_for_report(order_checks)),
        ("组合风险限额", risk_limit_status, risk_limit_summary),
        ("组合压力测试", stress_status, stress_summary),
    ]
    rows: list[RiskControlItem] = []
    for module, status_raw, summary in specs:
        status = _normalize_status(status_raw)
        rows.append(
            RiskControlItem(
                trade_date=date_s,
                module=module,
                status=status,
                severity_rank=_rank(status),
                summary=str(summary),
                action=_action_for_status(status),
            )
        )
    out = pd.DataFrame([r.__dict__ for r in rows], columns=RISK_CONTROL_COLUMNS)
    return out.sort_values(["severity_rank", "module"]).reset_index(drop=True)


def summarize_risk_control_report(report: pd.DataFrame | None) -> tuple[str, str]:
    """生成命令行和 Markdown 可读的总控摘要。"""
    if report is None or report.empty or "status" not in report.columns:
        return "NA", "未生成风险总控日报"
    status = report["status"].astype(str).str.upper()
    block = int((status == "BLOCK").sum())
    watch = int((status == "WATCH").sum())
    na = int((status == "NA").sum())
    passed = int((status == "PASS").sum())
    overall = "BLOCK" if block else "WATCH" if watch else "NA" if na else "PASS"
    focus = report[report["status"].astype(str).str.upper().isin({"BLOCK", "WATCH", "NA"})]
    focus_text = "；".join(
        "%s=%s" % (str(row.get("module", "")), str(row.get("status", "")))
        for row in focus.head(5).to_dict("records")
    )
    if not focus_text:
        focus_text = "无需要特别关注的风险模块"
    detail = "BLOCK=%d，WATCH=%d，NA=%d，PASS=%d；%s" % (block, watch, na, passed, focus_text)
    return overall, detail
