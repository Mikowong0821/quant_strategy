"""账户级回撤止损与降仓控制。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from config import Settings
from live.account_state import SNAPSHOT_COLUMNS, account_state_dir
from live.risk_limits import _weights_to_series


DRAWDOWN_RULE_COLUMNS = [
    "rule_id",
    "drawdown_threshold",
    "target_weight_scale",
    "status",
    "action",
    "enabled",
    "description",
]

DRAWDOWN_CONTROL_COLUMNS = [
    "trade_date",
    "status",
    "action",
    "current_total_asset",
    "peak_total_asset",
    "current_drawdown",
    "drawdown_abs",
    "previous_total_asset",
    "latest_return",
    "current_exposure",
    "target_exposure_before",
    "target_exposure_after",
    "target_weight_scale",
    "triggered_rule_id",
    "description",
    "details",
]

_STATUS_RANK = {"BLOCK": 0, "WATCH": 1, "PASS": 2, "NA": 3}


@dataclass(frozen=True)
class DrawdownRule:
    """一条账户回撤控制规则。"""

    rule_id: str
    drawdown_threshold: float
    target_weight_scale: float
    status: str
    action: str
    enabled: bool
    description: str


def default_drawdown_rules() -> pd.DataFrame:
    """返回 MVP 默认回撤控制规则。

    阈值是账户级工程默认值，不是投资建议。真实资金运行前应按账户规模、
    策略波动、持仓流动性和最大可承受亏损重新校准。
    """
    rows = [
        DrawdownRule(
            "drawdown_watch_5pct",
            0.05,
            0.70,
            "WATCH",
            "REDUCE_EXPOSURE",
            True,
            "账户从历史高点回撤超过 5%，目标股票仓位降到原计划的 70%。",
        ),
        DrawdownRule(
            "drawdown_defensive_10pct",
            0.10,
            0.50,
            "WATCH",
            "DEFENSIVE_DE_RISK",
            True,
            "账户从历史高点回撤超过 10%，进入防守状态，目标股票仓位降到原计划的 50%。",
        ),
        DrawdownRule(
            "drawdown_stop_15pct",
            0.15,
            0.00,
            "BLOCK",
            "STOP_LOSS_TO_CASH",
            True,
            "账户从历史高点回撤超过 15%，触发账户级止损，目标股票仓位降为 0。",
        ),
    ]
    return pd.DataFrame([r.__dict__ for r in rows], columns=DRAWDOWN_RULE_COLUMNS)


def load_drawdown_rules(path: str | None = None) -> pd.DataFrame:
    """读取回撤控制规则；未提供路径时返回默认规则。"""
    if path is None or str(path).strip() == "":
        return default_drawdown_rules()
    frame = pd.read_csv(path)
    missing = set(DRAWDOWN_RULE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError("回撤控制规则表缺少必要列: %s" % ", ".join(sorted(missing)))
    out = frame.loc[:, DRAWDOWN_RULE_COLUMNS].copy()
    out["drawdown_threshold"] = pd.to_numeric(out["drawdown_threshold"], errors="coerce")
    out["target_weight_scale"] = pd.to_numeric(out["target_weight_scale"], errors="coerce").clip(0.0, 1.0)
    out["status"] = out["status"].astype(str).str.strip().str.upper()
    out["action"] = out["action"].astype(str).str.strip().str.upper()
    out["enabled"] = out["enabled"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "on"})
    return out


def load_account_snapshots(settings: Settings, *, strategy: str) -> pd.DataFrame:
    """读取纸面账户历史快照；不存在时返回空表。"""
    path = account_state_dir(settings, strategy) / "snapshots.csv"
    if not path.exists():
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    frame = pd.read_csv(path)
    for col in SNAPSHOT_COLUMNS:
        if col not in frame.columns:
            frame[col] = float("nan") if col != "date" else ""
    return frame.loc[:, SNAPSHOT_COLUMNS].copy()


def build_current_account_snapshot(
    *,
    cash: float,
    positions: pd.DataFrame | None,
    latest_prices: pd.Series | dict[str, float],
) -> dict[str, float]:
    """按最新价格估算当前纸面账户快照。"""
    price_s = _weights_to_series(latest_prices)
    market_value = 0.0
    n_positions = 0
    if positions is not None and not positions.empty and "symbol" in positions.columns and "shares" in positions.columns:
        frame = positions.copy()
        frame["symbol"] = frame["symbol"].astype(str)
        frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce").fillna(0.0)
        frame["price"] = frame["symbol"].map(price_s)
        frame["value"] = frame["shares"] * pd.to_numeric(frame["price"], errors="coerce")
        frame = frame[frame["value"].notna() & (frame["value"] > 0.0)]
        market_value = float(frame["value"].sum())
        n_positions = int((frame["shares"] > 0.0).sum())
    total_asset = float(cash) + market_value
    return {
        "cash": float(cash),
        "market_value": market_value,
        "total_asset": total_asset,
        "n_positions": float(n_positions),
    }


def _latest_previous_asset(snapshots: pd.DataFrame, trade_date: pd.Timestamp | None) -> float | None:
    if snapshots.empty or "total_asset" not in snapshots.columns:
        return None
    frame = snapshots.copy()
    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce")
    frame["total_asset"] = pd.to_numeric(frame["total_asset"], errors="coerce")
    frame = frame[frame["total_asset"].notna() & (frame["total_asset"] > 0.0)]
    if trade_date is not None:
        frame = frame[frame["date"] < trade_date]
    frame = frame.sort_values("date")
    if frame.empty:
        return None
    return float(frame["total_asset"].iloc[-1])


def _peak_asset(snapshots: pd.DataFrame, current_asset: float, trade_date: pd.Timestamp | None) -> float | None:
    assets: list[float] = []
    if snapshots is not None and not snapshots.empty and "total_asset" in snapshots.columns:
        frame = snapshots.copy()
        frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce")
        frame["total_asset"] = pd.to_numeric(frame["total_asset"], errors="coerce")
        frame = frame[frame["total_asset"].notna() & (frame["total_asset"] > 0.0)]
        if trade_date is not None:
            frame = frame[frame["date"] <= trade_date]
        assets.extend(float(x) for x in frame["total_asset"].tolist())
    if current_asset > 0.0:
        assets.append(float(current_asset))
    if not assets:
        return None
    return max(assets)


def _triggered_rule(rules: pd.DataFrame, drawdown_abs: float) -> dict[str, Any] | None:
    if rules is None or rules.empty:
        return None
    frame = rules.copy()
    if "enabled" in frame.columns:
        frame = frame[frame["enabled"].astype(bool)]
    frame["drawdown_threshold"] = pd.to_numeric(frame["drawdown_threshold"], errors="coerce")
    frame = frame[frame["drawdown_threshold"].notna()]
    frame = frame[frame["drawdown_threshold"] <= float(drawdown_abs) + 1e-12]
    if frame.empty:
        return None
    frame = frame.sort_values("drawdown_threshold")
    return frame.iloc[-1].to_dict()


def evaluate_drawdown_control(
    rules: pd.DataFrame,
    snapshots: pd.DataFrame,
    current_snapshot: dict[str, Any],
    target_weights: pd.Series | dict[str, float],
    *,
    trade_date: Any = None,
) -> pd.DataFrame:
    """评估账户回撤，并给出目标权重缩放建议。"""
    dt = pd.Timestamp(trade_date) if trade_date is not None else None
    current_asset = float(current_snapshot.get("total_asset", 0.0) or 0.0)
    market_value = float(current_snapshot.get("market_value", 0.0) or 0.0)
    weights = _weights_to_series(target_weights)
    target_exposure_before = float(weights[weights > 0.0].sum()) if not weights.empty else 0.0

    peak = _peak_asset(snapshots, current_asset, dt)
    previous_asset = _latest_previous_asset(snapshots, dt)
    if peak is None or peak <= 0.0 or current_asset <= 0.0:
        row = {
            "trade_date": dt.strftime("%Y-%m-%d") if dt is not None else "",
            "status": "NA",
            "action": "NO_ACCOUNT_ASSET",
            "current_total_asset": current_asset,
            "peak_total_asset": float("nan"),
            "current_drawdown": float("nan"),
            "drawdown_abs": float("nan"),
            "previous_total_asset": previous_asset if previous_asset is not None else float("nan"),
            "latest_return": float("nan"),
            "current_exposure": float("nan"),
            "target_exposure_before": target_exposure_before,
            "target_exposure_after": target_exposure_before,
            "target_weight_scale": 1.0,
            "triggered_rule_id": "",
            "description": "账户资产不可用，回撤控制暂不缩放目标权重。",
            "details": "missing_or_invalid_account_asset",
        }
        return pd.DataFrame([row], columns=DRAWDOWN_CONTROL_COLUMNS)

    current_drawdown = current_asset / peak - 1.0
    drawdown_abs = max(0.0, -current_drawdown)
    rule = _triggered_rule(rules, drawdown_abs)
    if rule is None:
        status = "PASS"
        action = "KEEP"
        scale = 1.0
        rule_id = ""
        description = "账户回撤未触发默认阈值，维持原目标权重。"
    else:
        status = str(rule.get("status", "WATCH")).upper()
        action = str(rule.get("action", "REDUCE_EXPOSURE")).upper()
        scale = float(rule.get("target_weight_scale", 1.0) or 0.0)
        rule_id = str(rule.get("rule_id", ""))
        description = str(rule.get("description", ""))

    latest_return = (
        current_asset / previous_asset - 1.0
        if previous_asset is not None and abs(previous_asset) > 1e-12
        else float("nan")
    )
    current_exposure = market_value / current_asset if abs(current_asset) > 1e-12 else float("nan")
    target_exposure_after = target_exposure_before * scale
    details = "drawdown=%.2f%% peak=%.2f current=%.2f scale=%.2f" % (
        drawdown_abs * 100.0,
        peak,
        current_asset,
        scale,
    )
    row = {
        "trade_date": dt.strftime("%Y-%m-%d") if dt is not None else "",
        "status": status,
        "action": action,
        "current_total_asset": current_asset,
        "peak_total_asset": peak,
        "current_drawdown": current_drawdown,
        "drawdown_abs": drawdown_abs,
        "previous_total_asset": previous_asset if previous_asset is not None else float("nan"),
        "latest_return": latest_return,
        "current_exposure": current_exposure,
        "target_exposure_before": target_exposure_before,
        "target_exposure_after": target_exposure_after,
        "target_weight_scale": scale,
        "triggered_rule_id": rule_id,
        "description": description,
        "details": details,
    }
    return pd.DataFrame([row], columns=DRAWDOWN_CONTROL_COLUMNS)


def apply_drawdown_control_to_weights(
    target_weights: pd.Series | dict[str, float],
    control: pd.DataFrame | None,
) -> pd.Series:
    """按回撤控制结果缩放目标股票权重。"""
    weights = _weights_to_series(target_weights)
    if weights.empty or control is None or control.empty or "target_weight_scale" not in control.columns:
        return weights
    scale = pd.to_numeric(control["target_weight_scale"], errors="coerce").fillna(1.0).iloc[0]
    scale = min(max(float(scale), 0.0), 1.0)
    out = weights * scale
    out = out[out > 1e-12].sort_index()
    return out


def summarize_drawdown_control(control: pd.DataFrame | None) -> tuple[str, str]:
    """返回日报和命令行可读的回撤控制摘要。"""
    if control is None or control.empty:
        return "NA", "未生成回撤控制结果"
    frame = control.copy()
    if "status" not in frame.columns:
        return "NA", "回撤控制结果缺少 status"
    frame["status"] = frame["status"].astype(str).str.upper()
    frame["_rank"] = frame["status"].map(_STATUS_RANK).fillna(9).astype(int)
    row = frame.sort_values("_rank").iloc[0]
    status = str(row.get("status", "NA"))
    drawdown = row.get("drawdown_abs", float("nan"))
    scale = row.get("target_weight_scale", float("nan"))
    action = str(row.get("action", ""))
    rule_id = str(row.get("triggered_rule_id", ""))
    try:
        drawdown_s = "%.2f%%" % (float(drawdown) * 100.0)
    except (TypeError, ValueError):
        drawdown_s = "NA"
    try:
        scale_s = "%.2f" % float(scale)
    except (TypeError, ValueError):
        scale_s = "NA"
    detail = "drawdown=%s scale=%s action=%s" % (drawdown_s, scale_s, action)
    if rule_id:
        detail += " rule=%s" % rule_id
    return status, detail
