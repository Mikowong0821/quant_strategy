"""组合层统一风险限额表与检查器。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


RISK_LIMIT_COLUMNS = [
    "limit_id",
    "category",
    "metric",
    "direction",
    "warning_threshold",
    "block_threshold",
    "unit",
    "enabled",
    "description",
    "action",
]

RISK_LIMIT_CHECK_COLUMNS = [
    "trade_date",
    "limit_id",
    "category",
    "metric",
    "status",
    "observed_value",
    "warning_threshold",
    "block_threshold",
    "direction",
    "unit",
    "description",
    "action",
    "details",
]

_STATUS_RANK = {"BLOCK": 0, "WATCH": 1, "PASS": 2, "NA": 3}


@dataclass(frozen=True)
class RiskLimit:
    """一条可审计的风控限额。"""

    limit_id: str
    category: str
    metric: str
    direction: str
    warning_threshold: float
    block_threshold: float
    unit: str
    enabled: bool
    description: str
    action: str


def default_risk_limits() -> pd.DataFrame:
    """返回 MVP 默认风险限额表。

    阈值是保守的工程默认值，不是投资建议。实盘前应结合账户规模、标的流动性、
    调仓频率和风险偏好重新校准。
    """
    rows = [
        RiskLimit(
            "max_single_position_weight",
            "portfolio",
            "max_single_position_weight",
            "max",
            0.30,
            0.40,
            "weight",
            True,
            "单只股票目标权重过高时，组合容易被个股事件主导。",
            "降低单票目标权重或增加持仓数量。",
        ),
        RiskLimit(
            "max_top3_weight",
            "portfolio",
            "max_top3_weight",
            "max",
            0.75,
            0.90,
            "weight",
            True,
            "前三大持仓占比过高时，名义分散会变成少数股票押注。",
            "降低前三大持仓集中度，必要时提高现金或扩展股票池。",
        ),
        RiskLimit(
            "min_effective_n",
            "portfolio",
            "effective_n",
            "min",
            5.0,
            3.0,
            "count",
            True,
            "effective_n 衡量真实分散度；越低说明权重越集中。",
            "提高有效持仓数量，或降低组合股票总仓位。",
        ),
        RiskLimit(
            "min_position_count",
            "portfolio",
            "position_count",
            "min",
            5.0,
            3.0,
            "count",
            True,
            "持仓只数太少时，单票波动和事件风险会明显放大。",
            "扩大可投股票池，或触发最小持仓数量降仓规则。",
        ),
        RiskLimit(
            "min_cash_weight",
            "account",
            "cash_weight",
            "min",
            0.03,
            0.0,
            "weight",
            True,
            "现金缓冲太低时，手续费、价格跳动或人工调整会让订单更容易失败。",
            "保留现金缓冲，减少买入金额。",
        ),
        RiskLimit(
            "max_industry_weight",
            "portfolio",
            "max_industry_weight",
            "max",
            0.35,
            0.50,
            "weight",
            True,
            "单行业权重过高时，组合收益可能主要来自行业暴露，而不是选股能力。",
            "降低超限行业权重，补充其他行业标的。",
        ),
        RiskLimit(
            "max_rebalance_turnover",
            "trading",
            "rebalance_turnover",
            "max",
            0.60,
            1.00,
            "weight",
            True,
            "单次调仓变化过大时，交易成本、冲击成本和执行偏差都会上升。",
            "启用调仓节流，或把调仓拆成多次执行。",
        ),
        RiskLimit(
            "risk_gate_block_count",
            "event_risk",
            "risk_gate_block_count",
            "max",
            0.0,
            0.0,
            "count",
            True,
            "目标组合里出现 BLOCK 级公告、舆情或人工黑名单风险。",
            "阻断相关标的买入/加仓，并重新生成订单计划。",
        ),
        RiskLimit(
            "risk_gate_watch_count",
            "event_risk",
            "risk_gate_watch_count",
            "max",
            0.0,
            999.0,
            "count",
            True,
            "目标组合里出现 WATCH 级风险，需要人工复核。",
            "保留观察，人工确认是否降权或暂停买入。",
        ),
        RiskLimit(
            "order_block_count",
            "trading",
            "order_block_count",
            "max",
            0.0,
            0.0,
            "count",
            True,
            "订单预检查出现被阻断订单，说明目标组合和真实可执行状态不一致。",
            "修正目标权重、交易状态、现金或风险黑名单后再运行。",
        ),
        RiskLimit(
            "max_order_block_ratio",
            "trading",
            "order_block_ratio",
            "max",
            0.10,
            0.30,
            "ratio",
            True,
            "被阻断订单占比过高时，当天调仓计划整体可靠性不足。",
            "暂停自动执行，改为人工确认。",
        ),
    ]
    return pd.DataFrame([r.__dict__ for r in rows], columns=RISK_LIMIT_COLUMNS)


def load_risk_limits(path: str | None = None) -> pd.DataFrame:
    """读取风险限额表；未提供路径时返回默认限额。"""
    if path is None or str(path).strip() == "":
        return default_risk_limits()
    frame = pd.read_csv(path)
    missing = set(RISK_LIMIT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError("风险限额表缺少必要列: %s" % ", ".join(sorted(missing)))
    out = frame.loc[:, RISK_LIMIT_COLUMNS].copy()
    out["warning_threshold"] = pd.to_numeric(out["warning_threshold"], errors="coerce")
    out["block_threshold"] = pd.to_numeric(out["block_threshold"], errors="coerce")
    out["enabled"] = out["enabled"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "on"})
    out["direction"] = out["direction"].astype(str).str.strip().str.lower()
    return out


def _weights_to_series(weights: pd.DataFrame | pd.Series | dict[str, float] | None) -> pd.Series:
    if weights is None:
        return pd.Series(dtype=float)
    if isinstance(weights, pd.Series):
        out = weights.astype(float).copy()
        out.index = out.index.astype(str)
    elif isinstance(weights, pd.DataFrame):
        symbol_col = "symbol" if "symbol" in weights.columns else "ts_code"
        weight_col = ""
        for candidate in ("weight", "target_weight", "current_weight"):
            if candidate in weights.columns:
                weight_col = candidate
                break
        if symbol_col not in weights.columns or not weight_col:
            raise ValueError("权重表须包含 symbol/ts_code 与 weight/target_weight/current_weight 列")
        out = pd.Series(
            pd.to_numeric(weights[weight_col], errors="coerce").fillna(0.0).to_numpy(),
            index=weights[symbol_col].astype(str),
            dtype=float,
        )
    else:
        out = pd.Series({str(k): float(v) for k, v in weights.items()}, dtype=float)
    out = out.groupby(level=0).sum().sort_index()
    return out[out.abs() > 1e-12]


def _industry_map(industry: pd.DataFrame | None) -> dict[str, str]:
    if industry is None or industry.empty:
        return {}
    symbol_col = ""
    for candidate in ("symbol", "ts_code", "股票代码", "code"):
        if candidate in industry.columns:
            symbol_col = candidate
            break
    industry_col = "industry" if "industry" in industry.columns else ""
    if not industry_col:
        for candidate in ("分类", "申万行业", "行业", "sw_industry", "sector"):
            if candidate in industry.columns:
                industry_col = candidate
                break
    if not symbol_col or not industry_col:
        return {}
    return {
        str(rec[symbol_col]): str(rec[industry_col])
        for rec in industry[[symbol_col, industry_col]].dropna().to_dict("records")
    }


def _status_for_value(observed: float | None, *, direction: str, warning: float, block: float) -> str:
    if observed is None or pd.isna(observed):
        return "NA"
    value = float(observed)
    direction = str(direction).lower()
    if direction == "max":
        if value > block:
            return "BLOCK"
        if value > warning:
            return "WATCH"
        return "PASS"
    if direction == "min":
        if value < block:
            return "BLOCK"
        if value < warning:
            return "WATCH"
        return "PASS"
    raise ValueError("未知风险限额方向: %s" % direction)


def calculate_portfolio_risk_metrics(
    target_weights: pd.DataFrame | pd.Series | dict[str, float],
    *,
    current_weights: pd.DataFrame | pd.Series | dict[str, float] | None = None,
    industry: pd.DataFrame | None = None,
    risk_gate: pd.DataFrame | None = None,
    order_checks: pd.DataFrame | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """计算限额表可消费的组合、交易和事件风险指标。"""
    target = _weights_to_series(target_weights).clip(lower=0.0)
    current = _weights_to_series(current_weights).clip(lower=0.0)
    metrics: dict[str, float] = {}
    details: dict[str, str] = {}

    total_weight = float(target.sum())
    positive = target[target > 1e-12]
    metrics["gross_target_weight"] = total_weight
    metrics["cash_weight"] = 1.0 - total_weight
    metrics["position_count"] = float(len(positive))
    metrics["max_single_position_weight"] = float(positive.max()) if not positive.empty else 0.0
    metrics["max_top3_weight"] = float(positive.sort_values(ascending=False).head(3).sum()) if not positive.empty else 0.0
    hhi = float((positive**2).sum()) if not positive.empty else 0.0
    metrics["hhi"] = hhi
    metrics["effective_n"] = (1.0 / hhi) if hhi > 1e-12 else 0.0
    details["max_single_position_weight"] = (
        "top=%s %.2f%%" % (positive.idxmax(), float(positive.max()) * 100.0) if not positive.empty else "empty_target"
    )
    details["max_top3_weight"] = ",".join(positive.sort_values(ascending=False).head(3).index.astype(str))
    details["effective_n"] = "hhi=%.6f" % hhi
    details["position_count"] = "weights_above_zero=%d" % len(positive)
    details["cash_weight"] = "target_weight_sum=%.4f" % total_weight

    industries = _industry_map(industry)
    if industries and not positive.empty:
        industry_frame = pd.DataFrame({"symbol": positive.index, "weight": positive.values})
        industry_frame["industry"] = industry_frame["symbol"].map(industries)
        known = industry_frame[industry_frame["industry"].notna()].copy()
        coverage = float(known["weight"].sum() / positive.sum()) if float(positive.sum()) > 1e-12 else 0.0
        if coverage < 0.8 or known.empty:
            metrics["max_industry_weight"] = float("nan")
            details["max_industry_weight"] = "industry_coverage=%.2f%% below 80%%" % (coverage * 100.0)
        else:
            industry_weights = known.groupby("industry")["weight"].sum().sort_values(ascending=False)
            metrics["max_industry_weight"] = float(industry_weights.iloc[0])
            details["max_industry_weight"] = "%s %.2f%%; coverage=%.2f%%" % (
                industry_weights.index[0],
                float(industry_weights.iloc[0]) * 100.0,
                coverage * 100.0,
            )
    else:
        metrics["max_industry_weight"] = float("nan")
        details["max_industry_weight"] = "no_industry_input"

    if not current.empty:
        aligned = sorted(set(target.index) | set(current.index))
        turnover = 0.5 * float((target.reindex(aligned).fillna(0.0) - current.reindex(aligned).fillna(0.0)).abs().sum())
        metrics["rebalance_turnover"] = turnover
        details["rebalance_turnover"] = "symbols=%d" % len(aligned)
    else:
        metrics["rebalance_turnover"] = float("nan")
        details["rebalance_turnover"] = "no_current_weights_input"

    if risk_gate is not None and not risk_gate.empty and "gate_status" in risk_gate.columns:
        gate = risk_gate.copy()
        if "symbol" in gate.columns:
            gate = gate[gate["symbol"].astype(str).isin(set(positive.index.astype(str)))]
        statuses = gate["gate_status"].astype(str).str.upper()
        metrics["risk_gate_block_count"] = float((statuses == "BLOCK").sum())
        metrics["risk_gate_watch_count"] = float((statuses == "WATCH").sum())
        details["risk_gate_block_count"] = "target_symbols_only=%s" % ("symbol" in risk_gate.columns)
        details["risk_gate_watch_count"] = "target_symbols_only=%s" % ("symbol" in risk_gate.columns)
    else:
        metrics["risk_gate_block_count"] = 0.0
        metrics["risk_gate_watch_count"] = 0.0
        details["risk_gate_block_count"] = "no_risk_gate_input"
        details["risk_gate_watch_count"] = "no_risk_gate_input"

    if order_checks is not None and not order_checks.empty and "check_status" in order_checks.columns:
        status = order_checks["check_status"].astype(str).str.upper()
        block_count = float((status == "BLOCK").sum())
        metrics["order_block_count"] = block_count
        metrics["order_block_ratio"] = block_count / float(len(order_checks)) if len(order_checks) else 0.0
        details["order_block_count"] = "orders=%d" % len(order_checks)
        details["order_block_ratio"] = "orders=%d" % len(order_checks)
    else:
        metrics["order_block_count"] = 0.0
        metrics["order_block_ratio"] = 0.0
        details["order_block_count"] = "no_order_checks_input"
        details["order_block_ratio"] = "no_order_checks_input"

    return metrics, details


def check_risk_limits(
    limits: pd.DataFrame,
    target_weights: pd.DataFrame | pd.Series | dict[str, float],
    *,
    trade_date: Any = None,
    current_weights: pd.DataFrame | pd.Series | dict[str, float] | None = None,
    industry: pd.DataFrame | None = None,
    risk_gate: pd.DataFrame | None = None,
    order_checks: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """按限额表检查目标组合，输出 PASS / WATCH / BLOCK 明细。"""
    missing = set(RISK_LIMIT_COLUMNS) - set(limits.columns)
    if missing:
        raise ValueError("风险限额表缺少必要列: %s" % ", ".join(sorted(missing)))
    metrics, details = calculate_portfolio_risk_metrics(
        target_weights,
        current_weights=current_weights,
        industry=industry,
        risk_gate=risk_gate,
        order_checks=order_checks,
    )
    dt = pd.Timestamp(trade_date).strftime("%Y-%m-%d") if trade_date is not None else ""
    rows: list[dict[str, Any]] = []
    enabled = limits.copy()
    enabled["enabled"] = enabled["enabled"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "on"})
    enabled = enabled[enabled["enabled"]].copy()
    for rec in enabled.to_dict("records"):
        metric = str(rec["metric"])
        observed = metrics.get(metric, float("nan"))
        warning = float(rec["warning_threshold"])
        block = float(rec["block_threshold"])
        direction = str(rec["direction"]).lower()
        status = _status_for_value(observed, direction=direction, warning=warning, block=block)
        rows.append(
            {
                "trade_date": dt,
                "limit_id": str(rec["limit_id"]),
                "category": str(rec["category"]),
                "metric": metric,
                "status": status,
                "observed_value": observed,
                "warning_threshold": warning,
                "block_threshold": block,
                "direction": direction,
                "unit": str(rec["unit"]),
                "description": str(rec["description"]),
                "action": str(rec["action"]),
                "details": details.get(metric, ""),
            }
        )
    out = pd.DataFrame(rows, columns=RISK_LIMIT_CHECK_COLUMNS)
    if out.empty:
        return out
    out["_rank"] = out["status"].map(_STATUS_RANK).fillna(9).astype(int)
    return out.sort_values(["_rank", "category", "limit_id"]).drop(columns="_rank").reset_index(drop=True)


def summarize_risk_limit_checks(checks: pd.DataFrame | None) -> tuple[str, str]:
    """给日报 / 命令行使用的风险限额摘要。"""
    if checks is None or checks.empty or "status" not in checks.columns:
        return "PASS", "未找到风险限额超限记录"
    status = checks["status"].astype(str).str.upper()
    block = int((status == "BLOCK").sum())
    watch = int((status == "WATCH").sum())
    passed = int((status == "PASS").sum())
    na = int((status == "NA").sum())
    if block:
        overall = "BLOCK"
    elif watch:
        overall = "WATCH"
    else:
        overall = "PASS"
    return overall, "BLOCK=%d，WATCH=%d，PASS=%d，NA=%d" % (block, watch, passed, na)
