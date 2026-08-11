"""
纸面交易日报：把单日纸面运行结果整理成 Markdown。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from config import Settings
from live.factor_health_report import summarize_factor_health_report
from live.manual_confirmation import FACTOR_HEALTH_SEVERITY, summarize_factor_health
from live.risk_blacklist import summarize_risk_blacklist_for_report
from live.risk_gate import summarize_risk_gate_for_report
from live.risk_limits import summarize_risk_limit_checks
from live.stress_test import summarize_stress_tests
from live.style_exposure_monitor import summarize_style_exposure_for_report


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


def _factor_health_section(factor_monitor: pd.DataFrame | None) -> list[str]:
    status, reasons = summarize_factor_health(factor_monitor)
    lines = [
        "## 因子健康与失效监控",
        "",
        "- 整体状态：`%s`" % status,
        "- 状态原因：%s" % reasons,
    ]
    if factor_monitor is None or factor_monitor.empty:
        lines.extend(
            [
                "- 监控明细：未找到因子失效监控表，默认路径为 `output/factor_validation/factor_decay_monitor.csv`。",
                "",
            ]
        )
        return lines

    monitor = factor_monitor.copy()
    if "status" in monitor.columns:
        monitor["status"] = monitor["status"].astype(str).str.upper()
        monitor["_severity_rank"] = monitor["status"].map(FACTOR_HEALTH_SEVERITY).fillna(0).astype(int)
    else:
        monitor["status"] = "UNKNOWN"
        monitor["_severity_rank"] = 0
    counts = monitor["status"].value_counts().to_dict()
    risky_count = int((monitor["_severity_rank"] >= FACTOR_HEALTH_SEVERITY["WATCH"]).sum())
    count_parts = ["%s=%d" % (key, int(counts.get(key, 0))) for key in ["OK", "WATCH", "DEGRADED", "FAILED"]]
    lines.extend(
        [
            "- 因子数量：%d" % int(len(monitor)),
            "- 风险因子数量：%d" % risky_count,
            "- 状态分布：%s" % "，".join(count_parts),
            "",
        ]
    )

    sort_cols = ["_severity_rank"]
    ascending = [False]
    if "factor" in monitor.columns:
        sort_cols.append("factor")
        ascending.append(True)
    display = monitor.sort_values(sort_cols, ascending=ascending)
    lines.extend(
        [
            _markdown_table(
                display,
                [
                    "factor",
                    "status",
                    "reasons",
                    "validation_ic_mean",
                    "validation_positive_rate",
                    "validation_excess_ann_return",
                    "validation_top_minus_bottom_ann",
                    "validation_monotonicity_score",
                ],
                [
                    "因子",
                    "状态",
                    "原因",
                    "验证期IC均值",
                    "验证期IC胜率",
                    "验证期多头超额",
                    "验证期Top-Bottom",
                    "单调性",
                ],
                max_rows=30,
            )
        ]
    )
    return lines


def _style_exposure_section(style_exposure: pd.DataFrame | None) -> list[str]:
    status, detail = summarize_style_exposure_for_report(style_exposure)
    lines = [
        "## 组合风格暴露",
        "",
        "- 主导风格：`%s`" % status,
        "- 摘要：%s" % detail,
    ]
    if style_exposure is None or style_exposure.empty:
        lines.extend(
            [
                "- 暴露明细：未找到风格暴露表，默认路径为 `output/factor_diagnostics/style_exposure.csv`。",
                "",
            ]
        )
        return lines

    display = style_exposure.copy()
    if "date" in display.columns:
        display["date"] = pd.to_datetime(display["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    lines.extend(
        [
            "",
            _markdown_table(
                display,
                [
                    "date",
                    "style",
                    "weighted_exposure",
                    "score_coverage",
                    "n_positions",
                    "n_scored_positions",
                ],
                [
                    "暴露日期",
                    "风格",
                    "加权暴露",
                    "得分覆盖率",
                    "持仓数",
                    "有分数持仓数",
                ],
                max_rows=20,
            ),
        ]
    )
    return lines


def _enhanced_factor_health_section(health_report: pd.DataFrame | None) -> list[str]:
    status, detail = summarize_factor_health_report(health_report)
    lines = [
        "## 增强因子健康总览",
        "",
        "- 整体状态：`%s`" % status,
        "- 摘要：%s" % detail,
    ]
    if health_report is None or health_report.empty:
        lines.extend(
            [
                "- 明细：未找到增强因子健康总览，默认读取 `output/factor_validation/`、`output/factor_diagnostics/` 和 `output/market_regime/`。",
                "",
            ]
        )
        return lines

    display = health_report.copy()
    lines.extend(
        [
            "",
            _markdown_table(
                display,
                ["category", "status", "summary", "detail", "action"],
                ["类别", "状态", "摘要", "明细", "处理动作"],
                max_rows=20,
            ),
        ]
    )
    return lines


def _risk_blacklist_section(risk_blacklist: pd.DataFrame | None) -> list[str]:
    status, detail = summarize_risk_blacklist_for_report(risk_blacklist)
    lines = [
        "## 风险预警与黑名单",
        "",
        "- 整体状态：`%s`" % status,
        "- 摘要：%s" % detail,
    ]
    if risk_blacklist is None or risk_blacklist.empty:
        lines.extend(
            [
                "- 明细：未找到有效黑名单；默认读取 `data/risk_blacklist.csv`，也可通过 `--risk-blacklist` 指定。",
                "",
            ]
        )
        return lines

    display = risk_blacklist.copy()
    for col in ["created_at", "expires_at"]:
        if col in display.columns:
            display[col] = pd.to_datetime(display[col], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    lines.extend(
        [
            "",
            _markdown_table(
                display,
                ["symbol", "name", "severity", "reason", "source", "created_at", "expires_at"],
                ["标的", "名称", "风险等级", "原因", "来源", "生效日", "失效日"],
                max_rows=30,
            ),
        ]
    )
    return lines


def _risk_gate_section(risk_gate: pd.DataFrame | None) -> list[str]:
    status, detail = summarize_risk_gate_for_report(risk_gate)
    lines = [
        "## 统一风险门禁",
        "",
        "- 整体状态：`%s`" % status,
        "- 摘要：%s" % detail,
    ]
    if risk_gate is None or risk_gate.empty:
        lines.extend(
            [
                "- 明细：未找到统一风险门禁表；可先运行 `scripts/build_unified_risk_gate.py`，再通过 `--risk-gate` 指定。",
                "",
            ]
        )
        return lines

    display = risk_gate.copy()
    for col in ["trade_date", "latest_triggered_at", "expires_at"]:
        if col in display.columns:
            display[col] = pd.to_datetime(display[col], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    if "gate_status" in display.columns:
        rank = {"BLOCK": 0, "WATCH": 1, "PASS": 2}
        display["_rank"] = display["gate_status"].astype(str).str.upper().map(rank).fillna(9)
        display = display.sort_values(["_rank", "symbol"]).drop(columns="_rank")
    lines.extend(
        [
            "",
            _markdown_table(
                display,
                [
                    "symbol",
                    "name",
                    "gate_status",
                    "risk_count",
                    "sources",
                    "reason",
                    "latest_triggered_at",
                    "expires_at",
                ],
                ["标的", "名称", "门禁", "风险条数", "来源", "原因", "最近触发", "失效日"],
                max_rows=30,
            ),
        ]
    )
    return lines


def _risk_limit_section(risk_limit_checks: pd.DataFrame | None) -> list[str]:
    status, detail = summarize_risk_limit_checks(risk_limit_checks)
    lines = [
        "## 组合风险限额",
        "",
        "- 整体状态：`%s`" % status,
        "- 摘要：%s" % detail,
    ]
    if risk_limit_checks is None or risk_limit_checks.empty:
        lines.extend(
            [
                "- 明细：未找到组合风险限额检查表；日终脚本会默认计算，也可通过 `--risk-limits` 指定自定义限额表。",
                "",
            ]
        )
        return lines

    display = risk_limit_checks.copy()
    if "status" in display.columns:
        rank = {"BLOCK": 0, "WATCH": 1, "PASS": 2, "NA": 3}
        display["_rank"] = display["status"].astype(str).str.upper().map(rank).fillna(9)
        display = display.sort_values(["_rank", "category", "limit_id"]).drop(columns="_rank")
    risky = display[display["status"].astype(str).str.upper().isin(["BLOCK", "WATCH", "NA"])]
    lines.extend(
        [
            "",
            "### 需要关注的限额",
            "",
            _markdown_table(
                risky,
                [
                    "limit_id",
                    "category",
                    "status",
                    "observed_value",
                    "warning_threshold",
                    "block_threshold",
                    "details",
                    "action",
                ],
                ["限额", "类别", "状态", "当前值", "提醒阈值", "阻断阈值", "说明", "处理动作"],
                max_rows=20,
            ),
            "### 全部限额",
            "",
            _markdown_table(
                display,
                [
                    "limit_id",
                    "category",
                    "status",
                    "observed_value",
                    "warning_threshold",
                    "block_threshold",
                    "direction",
                    "details",
                ],
                ["限额", "类别", "状态", "当前值", "提醒阈值", "阻断阈值", "方向", "说明"],
                max_rows=30,
            ),
        ]
    )
    return lines


def _stress_test_section(stress_tests: pd.DataFrame | None) -> list[str]:
    status, detail = summarize_stress_tests(stress_tests)
    lines = [
        "## 组合压力测试",
        "",
        "- 整体状态：`%s`" % status,
        "- 摘要：%s" % detail,
    ]
    if stress_tests is None or stress_tests.empty:
        lines.extend(
            [
                "- 明细：未找到组合压力测试结果；日终脚本会默认计算，也可通过 `--stress-scenarios` 指定自定义情景表。",
                "",
            ]
        )
        return lines

    display = stress_tests.copy()
    if "status" in display.columns:
        rank = {"BLOCK": 0, "WATCH": 1, "PASS": 2, "NA": 3}
        display["_rank"] = display["status"].astype(str).str.upper().map(rank).fillna(9)
        display = display.sort_values(["_rank", "category", "scenario_id"]).drop(columns="_rank")
    risky = display[display["status"].astype(str).str.upper().isin(["BLOCK", "WATCH", "NA"])]
    lines.extend(
        [
            "",
            "### 需要关注的压力情景",
            "",
            _markdown_table(
                risky,
                [
                    "scenario_id",
                    "category",
                    "status",
                    "shock_value",
                    "affected_weight",
                    "estimated_loss_pct",
                    "estimated_loss_amount",
                    "details",
                    "action",
                ],
                ["情景", "类别", "状态", "冲击", "受影响权重", "预估损失率", "预估损失金额", "说明", "处理动作"],
                max_rows=20,
            ),
            "### 全部压力情景",
            "",
            _markdown_table(
                display,
                [
                    "scenario_id",
                    "category",
                    "status",
                    "shock_value",
                    "affected_weight",
                    "estimated_loss_pct",
                    "warning_loss",
                    "block_loss",
                    "affected_symbols",
                ],
                ["情景", "类别", "状态", "冲击", "受影响权重", "预估损失率", "提醒阈值", "阻断阈值", "标的"],
                max_rows=30,
            ),
        ]
    )
    return lines


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
    factor_monitor = result.get("factor_decay_monitor")
    style_exposure = result.get("style_exposure")
    factor_health_report = result.get("factor_health_report")
    risk_gate = result.get("risk_gate")
    risk_blacklist = result.get("risk_blacklist")
    risk_limit_checks = result.get("risk_limit_checks")
    stress_tests = result.get("stress_tests")

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

    lines.extend(_factor_health_section(factor_monitor))
    lines.extend(_style_exposure_section(style_exposure))
    lines.extend(_enhanced_factor_health_section(factor_health_report))
    lines.extend(_risk_gate_section(risk_gate))
    lines.extend(_risk_blacklist_section(risk_blacklist))
    lines.extend(_risk_limit_section(risk_limit_checks))
    lines.extend(_stress_test_section(stress_tests))

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
                ["symbol", "side", "delta_shares", "estimated_amount", "check_reason", "blacklist_reason"],
                ["标的", "方向", "股数变化", "预估金额", "阻断原因", "黑名单说明"],
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
