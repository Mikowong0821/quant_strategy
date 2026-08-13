"""容量、参与率与冲击成本估算。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


CAPACITY_RULE_COLUMNS = [
    "rule_id",
    "metric",
    "direction",
    "warning_threshold",
    "block_threshold",
    "unit",
    "enabled",
    "description",
    "action",
]

CAPACITY_IMPACT_COLUMNS = [
    "trade_date",
    "symbol",
    "side",
    "estimated_amount",
    "avg_amount",
    "participation_rate",
    "impact_cost_bps",
    "impact_cost_amount",
    "max_order_amount_at_warning",
    "capacity_multiplier_at_warning",
    "status",
    "details",
]

CAPACITY_SUMMARY_COLUMNS = [
    "trade_date",
    "status",
    "n_orders",
    "n_with_liquidity",
    "n_missing_liquidity",
    "max_participation_rate",
    "total_order_amount",
    "estimated_impact_cost_amount",
    "estimated_impact_cost_bps",
    "portfolio_capacity_multiplier_at_warning",
    "details",
]

_STATUS_RANK = {"BLOCK": 0, "WATCH": 1, "NA": 2, "PASS": 3}


@dataclass(frozen=True)
class CapacityRule:
    """一条容量与冲击成本阈值规则。"""

    rule_id: str
    metric: str
    direction: str
    warning_threshold: float
    block_threshold: float
    unit: str
    enabled: bool
    description: str
    action: str


def default_capacity_rules() -> pd.DataFrame:
    """返回 MVP 默认容量与冲击成本规则。

    默认按订单金额占过去 20 日平均成交额的比例判断。真实交易前应结合标的
    成交结构、账户规模、交易时段、是否拆单和实际滑点记录重新校准。
    """
    rows = [
        CapacityRule(
            "max_order_participation_rate",
            "participation_rate",
            "max",
            0.05,
            0.10,
            "ratio",
            True,
            "单笔订单金额占过去 N 日平均成交额比例过高时，容易产生冲击成本。",
            "降低下单金额、拆单执行或暂缓交易。",
        ),
        CapacityRule(
            "max_impact_cost_bps",
            "impact_cost_bps",
            "max",
            25.0,
            50.0,
            "bps",
            True,
            "估算冲击成本过高时，回测收益可能无法在真实交易中复现。",
            "降低参与率、延长执行时间或剔除流动性不足标的。",
        ),
    ]
    return pd.DataFrame([r.__dict__ for r in rows], columns=CAPACITY_RULE_COLUMNS)


def load_capacity_rules(path: str | Path | None = None) -> pd.DataFrame:
    """读取容量规则；未提供路径时返回默认规则。"""
    if path is None or str(path).strip() == "":
        return default_capacity_rules()
    frame = pd.read_csv(path)
    missing = set(CAPACITY_RULE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError("容量规则表缺少必要列: %s" % ", ".join(sorted(missing)))
    out = frame.loc[:, CAPACITY_RULE_COLUMNS].copy()
    out["warning_threshold"] = pd.to_numeric(out["warning_threshold"], errors="coerce")
    out["block_threshold"] = pd.to_numeric(out["block_threshold"], errors="coerce")
    out["enabled"] = out["enabled"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "on"})
    out["direction"] = out["direction"].astype(str).str.strip().str.lower()
    return out


def load_liquidity_history(path: str | Path | None) -> pd.DataFrame | None:
    """读取日频流动性历史；不存在或未提供时返回 None。"""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    frame = pd.read_csv(p)
    if frame.empty:
        return frame
    return frame


def _status_for_metric(value: float | None, *, warning: float, block: float, direction: str) -> str:
    if value is None or pd.isna(value):
        return "NA"
    v = float(value)
    if direction == "max":
        if v > float(block):
            return "BLOCK"
        if v > float(warning):
            return "WATCH"
        return "PASS"
    if direction == "min":
        if v < float(block):
            return "BLOCK"
        if v < float(warning):
            return "WATCH"
        return "PASS"
    raise ValueError("未知容量规则方向: %s" % direction)


def _combined_status(statuses: list[str]) -> str:
    if not statuses:
        return "NA"
    return min((str(s).upper() for s in statuses), key=lambda s: _STATUS_RANK.get(s, 9))


def _normalize_liquidity_history(liquidity: pd.DataFrame | None) -> pd.DataFrame:
    if liquidity is None or liquidity.empty:
        return pd.DataFrame(columns=["date", "symbol", "amount"])
    frame = liquidity.copy()
    date_col = ""
    for candidate in ("date", "trade_date"):
        if candidate in frame.columns:
            date_col = candidate
            break
    symbol_col = ""
    for candidate in ("symbol", "ts_code", "股票代码", "code"):
        if candidate in frame.columns:
            symbol_col = candidate
            break
    amount_col = ""
    for candidate in ("amount", "turnover", "成交额", "amt"):
        if candidate in frame.columns:
            amount_col = candidate
            break
    if not date_col or not symbol_col:
        return pd.DataFrame(columns=["date", "symbol", "amount"])
    if not amount_col and {"volume", "close"}.issubset(frame.columns):
        frame["amount"] = pd.to_numeric(frame["volume"], errors="coerce") * pd.to_numeric(frame["close"], errors="coerce")
        amount_col = "amount"
    if not amount_col:
        return pd.DataFrame(columns=["date", "symbol", "amount"])
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_col], errors="coerce"),
            "symbol": frame[symbol_col].astype(str),
            "amount": pd.to_numeric(frame[amount_col], errors="coerce"),
        }
    )
    out = out[out["date"].notna() & out["amount"].notna() & (out["amount"] > 0.0)]
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def average_amount_by_symbol(
    liquidity: pd.DataFrame | None,
    *,
    trade_date: Any,
    lookback_days: int = 20,
) -> pd.Series:
    """计算不晚于 trade_date 的过去 lookback_days 条记录平均成交额。"""
    frame = _normalize_liquidity_history(liquidity)
    if frame.empty:
        return pd.Series(dtype=float)
    dt = pd.Timestamp(trade_date)
    frame = frame[frame["date"] <= dt].copy()
    if frame.empty:
        return pd.Series(dtype=float)
    lookback = max(1, int(lookback_days))
    out = (
        frame.sort_values(["symbol", "date"])
        .groupby("symbol", group_keys=False)
        .tail(lookback)
        .groupby("symbol")["amount"]
        .mean()
        .sort_index()
    )
    out.index = out.index.astype(str)
    return out.astype(float)


def _rule_threshold(rules: pd.DataFrame, metric: str, fallback_warning: float, fallback_block: float) -> tuple[float, float]:
    if rules is None or rules.empty:
        return fallback_warning, fallback_block
    frame = rules.copy()
    if "enabled" in frame.columns:
        frame = frame[frame["enabled"].astype(bool)]
    frame = frame[frame["metric"].astype(str) == metric]
    if frame.empty:
        return fallback_warning, fallback_block
    rec = frame.iloc[0]
    return float(rec["warning_threshold"]), float(rec["block_threshold"])


def evaluate_capacity_impact(
    orders: pd.DataFrame,
    liquidity: pd.DataFrame | None,
    *,
    trade_date: Any,
    rules: pd.DataFrame | None = None,
    lookback_days: int = 20,
    impact_coefficient_bps: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """评估订单容量、参与率和冲击成本。"""
    if rules is None:
        rules = default_capacity_rules()
    if orders is None or orders.empty:
        empty_detail = pd.DataFrame(columns=CAPACITY_IMPACT_COLUMNS)
        summary = pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp(trade_date).strftime("%Y-%m-%d"),
                    "status": "NA",
                    "n_orders": 0,
                    "n_with_liquidity": 0,
                    "n_missing_liquidity": 0,
                    "max_participation_rate": float("nan"),
                    "total_order_amount": 0.0,
                    "estimated_impact_cost_amount": 0.0,
                    "estimated_impact_cost_bps": float("nan"),
                    "portfolio_capacity_multiplier_at_warning": float("nan"),
                    "details": "no_orders",
                }
            ],
            columns=CAPACITY_SUMMARY_COLUMNS,
        )
        return empty_detail, summary

    avg_amount = average_amount_by_symbol(liquidity, trade_date=trade_date, lookback_days=lookback_days)
    part_warning, _ = _rule_threshold(rules, "participation_rate", 0.05, 0.10)
    dt_s = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    rows: list[dict[str, Any]] = []
    for rec in orders.to_dict("records"):
        side = str(rec.get("side", "")).upper()
        if side not in {"BUY", "SELL"}:
            continue
        symbol = str(rec.get("symbol", ""))
        amount = abs(float(rec.get("estimated_amount", 0.0) or 0.0))
        avg = float(avg_amount.get(symbol, float("nan"))) if not avg_amount.empty else float("nan")
        if pd.isna(avg) or avg <= 0.0:
            participation = float("nan")
            impact_bps = float("nan")
            impact_amount = float("nan")
            max_order = float("nan")
            cap_multiplier = float("nan")
            status = "NA"
            details = "missing_avg_amount"
        else:
            participation = amount / avg
            impact_bps = float(impact_coefficient_bps) * (participation**0.5)
            impact_amount = amount * impact_bps / 10000.0
            max_order = avg * float(part_warning)
            cap_multiplier = max_order / amount if amount > 1e-12 else float("inf")
            statuses: list[str] = []
            for rule in rules[rules["enabled"].astype(bool)].to_dict("records"):
                metric = str(rule["metric"])
                value = participation if metric == "participation_rate" else impact_bps if metric == "impact_cost_bps" else None
                statuses.append(
                    _status_for_metric(
                        value,
                        warning=float(rule["warning_threshold"]),
                        block=float(rule["block_threshold"]),
                        direction=str(rule["direction"]),
                    )
                )
            status = _combined_status(statuses)
            details = "amount=%.2f avg_amount=%.2f lookback=%d" % (amount, avg, int(lookback_days))
        rows.append(
            {
                "trade_date": dt_s,
                "symbol": symbol,
                "side": side,
                "estimated_amount": amount,
                "avg_amount": avg,
                "participation_rate": participation,
                "impact_cost_bps": impact_bps,
                "impact_cost_amount": impact_amount,
                "max_order_amount_at_warning": max_order,
                "capacity_multiplier_at_warning": cap_multiplier,
                "status": status,
                "details": details,
            }
        )
    detail = pd.DataFrame(rows, columns=CAPACITY_IMPACT_COLUMNS)
    if detail.empty:
        summary_status = "NA"
        n_with = 0
        n_missing = 0
        max_part = float("nan")
        total_amount = 0.0
        impact_total = 0.0
        impact_bps_total = float("nan")
        capacity_multiplier = float("nan")
        details = "no_buy_sell_orders"
    else:
        summary_status = _combined_status(detail["status"].astype(str).tolist())
        n_with = int(detail["avg_amount"].notna().sum())
        n_missing = int(detail["avg_amount"].isna().sum())
        max_part = float(pd.to_numeric(detail["participation_rate"], errors="coerce").max())
        total_amount = float(pd.to_numeric(detail["estimated_amount"], errors="coerce").fillna(0.0).sum())
        impact_total = float(pd.to_numeric(detail["impact_cost_amount"], errors="coerce").fillna(0.0).sum())
        impact_bps_total = impact_total / total_amount * 10000.0 if total_amount > 1e-12 else float("nan")
        cap_series = pd.to_numeric(detail["capacity_multiplier_at_warning"], errors="coerce")
        capacity_multiplier = float(cap_series[cap_series >= 0.0].min()) if cap_series.notna().any() else float("nan")
        details = "lookback=%d impact_coefficient_bps=%.2f" % (int(lookback_days), float(impact_coefficient_bps))
    summary = pd.DataFrame(
        [
            {
                "trade_date": dt_s,
                "status": summary_status,
                "n_orders": int(len(detail)),
                "n_with_liquidity": n_with,
                "n_missing_liquidity": n_missing,
                "max_participation_rate": max_part,
                "total_order_amount": total_amount,
                "estimated_impact_cost_amount": impact_total,
                "estimated_impact_cost_bps": impact_bps_total,
                "portfolio_capacity_multiplier_at_warning": capacity_multiplier,
                "details": details,
            }
        ],
        columns=CAPACITY_SUMMARY_COLUMNS,
    )
    return detail, summary


def summarize_capacity_impact(summary: pd.DataFrame | None) -> tuple[str, str]:
    """返回命令行和日报可读的容量摘要。"""
    if summary is None or summary.empty:
        return "NA", "未生成容量与冲击成本结果"
    row = summary.iloc[0]
    status = str(row.get("status", "NA")).upper()
    try:
        max_part = float(row.get("max_participation_rate", float("nan")))
        max_part_s = "%.2f%%" % (max_part * 100.0)
    except (TypeError, ValueError):
        max_part_s = "NA"
    try:
        impact_bps = float(row.get("estimated_impact_cost_bps", float("nan")))
        impact_bps_s = "%.2f bps" % impact_bps
    except (TypeError, ValueError):
        impact_bps_s = "NA"
    try:
        capacity = float(row.get("portfolio_capacity_multiplier_at_warning", float("nan")))
        capacity_s = "%.2fx" % capacity
    except (TypeError, ValueError):
        capacity_s = "NA"
    detail = "max_participation=%s impact=%s capacity_at_warning=%s missing_liquidity=%s" % (
        max_part_s,
        impact_bps_s,
        capacity_s,
        str(row.get("n_missing_liquidity", "")),
    )
    return status, detail
