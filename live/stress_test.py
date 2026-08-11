"""组合压力测试与情景分析。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from live.risk_limits import _industry_map, _weights_to_series


STRESS_SCENARIO_COLUMNS = [
    "scenario_id",
    "category",
    "shock_type",
    "shock_value",
    "warning_loss",
    "block_loss",
    "enabled",
    "description",
    "action",
]

STRESS_TEST_COLUMNS = [
    "trade_date",
    "scenario_id",
    "category",
    "shock_type",
    "status",
    "shock_value",
    "affected_weight",
    "estimated_portfolio_return",
    "estimated_loss_pct",
    "estimated_loss_amount",
    "warning_loss",
    "block_loss",
    "affected_symbols",
    "description",
    "action",
    "details",
]

_STATUS_RANK = {"BLOCK": 0, "WATCH": 1, "PASS": 2, "NA": 3}


@dataclass(frozen=True)
class StressScenario:
    """一条可审计的压力测试情景。"""

    scenario_id: str
    category: str
    shock_type: str
    shock_value: float
    warning_loss: float
    block_loss: float
    enabled: bool
    description: str
    action: str


def default_stress_scenarios() -> pd.DataFrame:
    """返回 MVP 默认压力测试情景表。

    阈值只用于工程演示和纸面交易风控提示，不构成投资建议。真实账户应结合
    账户规模、持仓波动、流动性和最大可承受回撤重新校准。
    """
    rows = [
        StressScenario(
            "market_down_3pct",
            "market",
            "market_down",
            -0.03,
            0.03,
            0.06,
            True,
            "全组合股票头寸按市场下跌 3% 估算损失。",
            "观察组合股票仓位和现金缓冲。",
        ),
        StressScenario(
            "market_down_5pct",
            "market",
            "market_down",
            -0.05,
            0.03,
            0.06,
            True,
            "全组合股票头寸按市场下跌 5% 估算损失。",
            "若进入 WATCH/BLOCK，降低股票仓位或提高现金。",
        ),
        StressScenario(
            "market_down_8pct",
            "market",
            "market_down",
            -0.08,
            0.03,
            0.06,
            True,
            "全组合股票头寸按市场下跌 8% 估算极端单日冲击。",
            "进入 BLOCK 时不建议继续自动加仓，应人工复核。",
        ),
        StressScenario(
            "largest_position_down_10pct",
            "single_name",
            "largest_position_down",
            -0.10,
            0.02,
            0.04,
            True,
            "第一大持仓单票下跌 10%，观察个股事件风险冲击。",
            "降低单票权重或扩展持仓分散度。",
        ),
        StressScenario(
            "top3_positions_down_8pct",
            "concentration",
            "top_n_down",
            -0.08,
            0.03,
            0.06,
            True,
            "前三大持仓同时下跌 8%，观察集中度风险。",
            "降低前三大集中度，必要时提高现金。",
        ),
        StressScenario(
            "largest_industry_down_8pct",
            "industry",
            "largest_industry_down",
            -0.08,
            0.03,
            0.06,
            True,
            "第一大行业整体下跌 8%，观察行业暴露冲击。",
            "降低超配行业权重，补充其他行业标的。",
        ),
    ]
    return pd.DataFrame([r.__dict__ for r in rows], columns=STRESS_SCENARIO_COLUMNS)


def load_stress_scenarios(path: str | None = None) -> pd.DataFrame:
    """读取压力测试情景表；未提供路径时返回默认情景。"""
    if path is None or str(path).strip() == "":
        return default_stress_scenarios()
    frame = pd.read_csv(path)
    missing = set(STRESS_SCENARIO_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError("压力测试情景表缺少必要列: %s" % ", ".join(sorted(missing)))
    out = frame.loc[:, STRESS_SCENARIO_COLUMNS].copy()
    out["shock_value"] = pd.to_numeric(out["shock_value"], errors="coerce")
    out["warning_loss"] = pd.to_numeric(out["warning_loss"], errors="coerce")
    out["block_loss"] = pd.to_numeric(out["block_loss"], errors="coerce")
    out["enabled"] = out["enabled"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "on"})
    return out


def _status_for_loss(loss_pct: float | None, *, warning_loss: float, block_loss: float) -> str:
    if loss_pct is None or pd.isna(loss_pct):
        return "NA"
    loss = float(loss_pct)
    if loss > float(block_loss):
        return "BLOCK"
    if loss > float(warning_loss):
        return "WATCH"
    return "PASS"


def _scenario_affected_weights(
    weights: pd.Series,
    scenario: dict[str, Any],
    *,
    industry: pd.DataFrame | None,
) -> tuple[pd.Series, str]:
    shock_type = str(scenario["shock_type"])
    positive = weights[weights > 1e-12].sort_values(ascending=False)
    if positive.empty:
        return pd.Series(dtype=float), "empty_target"

    if shock_type == "market_down":
        return positive, "all_positive_positions"

    if shock_type == "largest_position_down":
        return positive.head(1), "largest=%s" % str(positive.index[0])

    if shock_type == "top_n_down":
        return positive.head(3), "top_n=3"

    if shock_type == "largest_industry_down":
        industries = _industry_map(industry)
        if not industries:
            return pd.Series(dtype=float), "no_industry_input"
        frame = pd.DataFrame({"symbol": positive.index.astype(str), "weight": positive.values})
        frame["industry"] = frame["symbol"].map(industries)
        known = frame[frame["industry"].notna()].copy()
        coverage = float(known["weight"].sum() / positive.sum()) if float(positive.sum()) > 1e-12 else 0.0
        if known.empty or coverage < 0.8:
            return pd.Series(dtype=float), "industry_coverage=%.2f%% below 80%%" % (coverage * 100.0)
        industry_weights = known.groupby("industry")["weight"].sum().sort_values(ascending=False)
        top_industry = str(industry_weights.index[0])
        affected = frame[frame["industry"] == top_industry]
        out = pd.Series(affected["weight"].to_numpy(), index=affected["symbol"].astype(str), dtype=float)
        return out.groupby(level=0).sum().sort_values(ascending=False), "%s %.2f%%" % (
            top_industry,
            float(industry_weights.iloc[0]) * 100.0,
        )

    raise ValueError("未知压力测试 shock_type: %s" % shock_type)


def run_portfolio_stress_tests(
    scenarios: pd.DataFrame,
    target_weights: pd.DataFrame | pd.Series | dict[str, float],
    *,
    trade_date: Any = None,
    total_asset: float | None = None,
    industry: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """按情景表对目标组合做压力测试。"""
    missing = set(STRESS_SCENARIO_COLUMNS) - set(scenarios.columns)
    if missing:
        raise ValueError("压力测试情景表缺少必要列: %s" % ", ".join(sorted(missing)))

    weights = _weights_to_series(target_weights).clip(lower=0.0)
    dt = pd.Timestamp(trade_date).strftime("%Y-%m-%d") if trade_date is not None else ""
    asset = float(total_asset) if total_asset is not None and not pd.isna(total_asset) else float("nan")
    enabled = scenarios.copy()
    enabled["enabled"] = enabled["enabled"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "on"})
    enabled = enabled[enabled["enabled"]].copy()

    rows: list[dict[str, Any]] = []
    for rec in enabled.to_dict("records"):
        affected, details = _scenario_affected_weights(weights, rec, industry=industry)
        affected_weight = float(affected.sum()) if not affected.empty else float("nan")
        shock = float(rec["shock_value"])
        if affected.empty:
            estimated_return = float("nan")
            loss_pct = float("nan")
            loss_amount = float("nan")
            symbols = ""
        else:
            estimated_return = affected_weight * shock
            loss_pct = max(0.0, -estimated_return)
            loss_amount = loss_pct * asset if not pd.isna(asset) else float("nan")
            symbols = ",".join(affected.index.astype(str))
        warning = float(rec["warning_loss"])
        block = float(rec["block_loss"])
        status = _status_for_loss(loss_pct, warning_loss=warning, block_loss=block)
        rows.append(
            {
                "trade_date": dt,
                "scenario_id": str(rec["scenario_id"]),
                "category": str(rec["category"]),
                "shock_type": str(rec["shock_type"]),
                "status": status,
                "shock_value": shock,
                "affected_weight": affected_weight,
                "estimated_portfolio_return": estimated_return,
                "estimated_loss_pct": loss_pct,
                "estimated_loss_amount": loss_amount,
                "warning_loss": warning,
                "block_loss": block,
                "affected_symbols": symbols,
                "description": str(rec["description"]),
                "action": str(rec["action"]),
                "details": details,
            }
        )
    out = pd.DataFrame(rows, columns=STRESS_TEST_COLUMNS)
    if out.empty:
        return out
    out["_rank"] = out["status"].map(_STATUS_RANK).fillna(9).astype(int)
    return out.sort_values(["_rank", "category", "scenario_id"]).drop(columns="_rank").reset_index(drop=True)


def summarize_stress_tests(stress_tests: pd.DataFrame | None) -> tuple[str, str]:
    """给日报 / 命令行使用的压力测试摘要。"""
    if stress_tests is None or stress_tests.empty or "status" not in stress_tests.columns:
        return "PASS", "未找到压力测试风险记录"
    status = stress_tests["status"].astype(str).str.upper()
    block = int((status == "BLOCK").sum())
    watch = int((status == "WATCH").sum())
    passed = int((status == "PASS").sum())
    na = int((status == "NA").sum())
    overall = "BLOCK" if block else "WATCH" if watch else "PASS"
    losses = pd.to_numeric(stress_tests.get("estimated_loss_pct"), errors="coerce")
    if losses.notna().any():
        worst_idx = losses.idxmax()
        worst = stress_tests.loc[worst_idx]
        worst_text = "worst=%s %.2f%%" % (str(worst.get("scenario_id", "")), float(losses.loc[worst_idx]) * 100.0)
    else:
        worst_text = "worst=NA"
    return overall, "BLOCK=%d，WATCH=%d，PASS=%d，NA=%d，%s" % (block, watch, passed, na, worst_text)
