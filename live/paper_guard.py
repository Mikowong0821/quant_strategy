"""
纸面交易运行检查：在日终流程中识别输入、账户和结果异常。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GuardIssue:
    severity: str
    code: str
    message: str


class DailyPaperGuardError(RuntimeError):
    """日终纸面交易检查发现 ERROR 级问题。"""


def _issue(severity: str, code: str, message: str) -> GuardIssue:
    return GuardIssue(severity=severity.upper(), code=code, message=message)


def _is_finite_nonnegative(value: Any) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(x) and x >= 0.0)


def validate_daily_inputs(
    *,
    target_weights: pd.Series,
    latest_prices: pd.Series,
    run_date: Any,
    target_date: Any,
    price_date: Any,
    max_price_age_days: int = 7,
) -> list[GuardIssue]:
    """检查日终纸面交易输入。"""
    issues: list[GuardIssue] = []
    if target_weights.empty:
        issues.append(_issue("ERROR", "empty_target_weights", "目标权重为空"))
    if latest_prices.empty:
        issues.append(_issue("ERROR", "empty_latest_prices", "最新价格为空"))

    weights = pd.to_numeric(target_weights, errors="coerce")
    if weights.isna().any():
        issues.append(_issue("ERROR", "invalid_target_weight", "目标权重包含非数值"))
    if (weights < -1e-12).any():
        issues.append(_issue("ERROR", "negative_target_weight", "目标权重不能为负"))
    weight_sum = float(weights.fillna(0.0).sum())
    if weight_sum <= 1e-12:
        issues.append(_issue("ERROR", "zero_target_weight_sum", "目标权重之和为 0"))
    if weight_sum > 1.0 + 1e-8:
        issues.append(_issue("ERROR", "target_weight_sum_gt_one", "目标权重之和超过 1"))

    prices = pd.to_numeric(latest_prices, errors="coerce")
    if prices.isna().any():
        issues.append(_issue("ERROR", "invalid_latest_price", "最新价格包含非数值"))
    if (prices <= 0.0).any():
        issues.append(_issue("ERROR", "non_positive_latest_price", "最新价格必须为正"))

    missing_prices = sorted(set(weights[weights > 0.0].index.astype(str)) - set(prices.index.astype(str)))
    if missing_prices:
        issues.append(
            _issue(
                "ERROR",
                "missing_price_for_target",
                "目标权重标的缺少价格: %s" % ", ".join(missing_prices[:10]),
            )
        )

    run_dt = pd.Timestamp(run_date)
    price_dt = pd.Timestamp(price_date)
    target_dt = pd.Timestamp(target_date)
    if price_dt > run_dt:
        issues.append(_issue("ERROR", "future_price_date", "价格日期晚于运行日期"))
    if target_dt > run_dt:
        issues.append(_issue("ERROR", "future_target_date", "目标权重日期晚于运行日期"))
    price_age = int((run_dt.normalize() - price_dt.normalize()).days)
    if price_age > int(max_price_age_days):
        issues.append(
            _issue(
                "WARNING",
                "stale_price_date",
                "价格日期距运行日期 %d 天，可能过旧" % price_age,
            )
        )
    return issues


def validate_daily_result(result: dict[str, Any]) -> list[GuardIssue]:
    """检查纸面交易运行结果。"""
    issues: list[GuardIssue] = []
    cash = result.get("cash")
    if not _is_finite_nonnegative(cash):
        issues.append(_issue("ERROR", "invalid_cash", "运行后现金无效或为负"))

    snapshot = result.get("account_snapshot", {}) or {}
    for key in ("cash", "market_value", "total_asset", "n_positions"):
        if key in snapshot and not _is_finite_nonnegative(snapshot.get(key)):
            issues.append(_issue("ERROR", "invalid_snapshot_%s" % key, "账户快照字段 %s 无效或为负" % key))

    positions = result.get("positions")
    if isinstance(positions, pd.DataFrame) and not positions.empty:
        if "shares" not in positions.columns:
            issues.append(_issue("ERROR", "positions_missing_shares", "持仓表缺少 shares 列"))
        else:
            shares = pd.to_numeric(positions["shares"], errors="coerce")
            if shares.isna().any() or (shares < -1e-12).any():
                issues.append(_issue("ERROR", "invalid_position_shares", "持仓股数无效或为负"))
        if {"available_shares", "shares"}.issubset(positions.columns):
            available = pd.to_numeric(positions["available_shares"], errors="coerce")
            shares = pd.to_numeric(positions["shares"], errors="coerce")
            if available.isna().any() or (available < -1e-12).any():
                issues.append(_issue("ERROR", "invalid_available_shares", "可用股数无效或为负"))
            if (available > shares + 1e-8).any():
                issues.append(_issue("WARNING", "available_gt_shares", "存在可用股数大于持仓股数的记录"))

    checks = result.get("order_checks")
    if isinstance(checks, pd.DataFrame) and not checks.empty:
        if "check_status" in checks.columns:
            n_block = int((checks["check_status"] == "BLOCK").sum())
            if n_block == len(checks):
                issues.append(_issue("WARNING", "all_orders_blocked", "所有订单均被预检查阻断"))
        else:
            issues.append(_issue("ERROR", "checks_missing_status", "订单预检查缺少 check_status 列"))

    trades = result.get("paper_trades")
    if isinstance(trades, pd.DataFrame) and not trades.empty:
        if "cash_after" in trades.columns:
            cash_after = pd.to_numeric(trades["cash_after"], errors="coerce")
            if cash_after.isna().any() or (cash_after < -1e-8).any():
                issues.append(_issue("ERROR", "trade_cash_after_negative", "成交日志存在负现金或无效现金"))
        if "fill_status" in trades.columns and (trades["fill_status"] == "SKIPPED").all():
            issues.append(_issue("WARNING", "all_trades_skipped", "成交层全部跳过"))

    return issues


def raise_on_guard_errors(issues: Iterable[GuardIssue]) -> None:
    errors = [x for x in issues if x.severity == "ERROR"]
    if not errors:
        return
    detail = "\n".join("- %s: %s" % (x.code, x.message) for x in errors)
    raise DailyPaperGuardError("日终纸面交易检查失败:\n%s" % detail)


def format_guard_issues(issues: Iterable[GuardIssue]) -> str:
    rows = list(issues)
    if not rows:
        return "guard=OK"
    return "\n".join("[%s] %s: %s" % (x.severity, x.code, x.message) for x in rows)
