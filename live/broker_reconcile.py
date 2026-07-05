"""
纸面账户与只读券商账户对账。

本模块只比较账户和持仓差异，不下单、不撤单。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings
from live.account_state import account_state_dir, load_account_state
from live.broker import BrokerAdapter


ACCOUNT_RECON_COLUMNS = [
    "trade_date",
    "strategy",
    "paper_cash",
    "broker_cash",
    "cash_diff",
    "paper_market_value",
    "broker_market_value",
    "market_value_diff",
    "paper_total_asset",
    "broker_total_asset",
    "total_asset_diff",
    "cash_status",
    "asset_status",
]

POSITION_RECON_COLUMNS = [
    "symbol",
    "paper_shares",
    "broker_shares",
    "share_diff",
    "paper_available_shares",
    "broker_available_shares",
    "available_share_diff",
    "status",
]


def _date_to_str(value: Any) -> str:
    if value is None or value == "":
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _latest_snapshot(settings: Settings, strategy: str, trade_date: Any = None) -> dict[str, Any]:
    path = account_state_dir(settings, strategy) / "snapshots.csv"
    if not path.exists():
        return {}
    snapshots = pd.read_csv(path)
    if snapshots.empty or "date" not in snapshots.columns:
        return {}
    snapshots = snapshots.copy()
    snapshots["date"] = pd.to_datetime(snapshots["date"], errors="coerce")
    snapshots = snapshots[snapshots["date"].notna()]
    if trade_date is not None:
        snapshots = snapshots[snapshots["date"] <= pd.Timestamp(trade_date)]
    if snapshots.empty:
        return {}
    return snapshots.sort_values("date").iloc[-1].to_dict()


def _positions_to_recon_frame(
    positions: pd.DataFrame,
    *,
    prefix: str,
) -> pd.DataFrame:
    cols = ["symbol", "%s_shares" % prefix, "%s_available_shares" % prefix]
    if positions.empty:
        return pd.DataFrame(columns=cols)
    required = {"symbol", "shares"}
    missing = required - set(positions.columns)
    if missing:
        raise ValueError("持仓表缺少必要列: %s" % ", ".join(sorted(missing)))
    frame = positions.copy()
    if "available_shares" not in frame.columns:
        frame["available_shares"] = frame["shares"]
    out = pd.DataFrame(
        {
            "symbol": frame["symbol"].astype(str),
            "%s_shares" % prefix: pd.to_numeric(frame["shares"], errors="coerce").fillna(0.0),
            "%s_available_shares" % prefix: pd.to_numeric(frame["available_shares"], errors="coerce").fillna(0.0),
        }
    )
    return out.groupby("symbol", as_index=False).sum().sort_values("symbol").reset_index(drop=True)


def build_position_reconciliation(
    paper_positions: pd.DataFrame,
    broker_positions: pd.DataFrame,
    *,
    share_tolerance: float = 0.0,
) -> pd.DataFrame:
    """生成纸面持仓和券商持仓的逐股票差异表。"""
    paper = _positions_to_recon_frame(paper_positions, prefix="paper")
    broker = _positions_to_recon_frame(broker_positions, prefix="broker")
    merged = paper.merge(broker, on="symbol", how="outer").fillna(0.0)
    for col in (
        "paper_shares",
        "broker_shares",
        "paper_available_shares",
        "broker_available_shares",
    ):
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    merged["share_diff"] = merged["broker_shares"] - merged["paper_shares"]
    merged["available_share_diff"] = merged["broker_available_shares"] - merged["paper_available_shares"]

    def _status(row: pd.Series) -> str:
        if abs(float(row["share_diff"])) <= float(share_tolerance) and abs(float(row["available_share_diff"])) <= float(share_tolerance):
            return "OK"
        if float(row["paper_shares"]) <= 0.0 < float(row["broker_shares"]):
            return "BROKER_ONLY"
        if float(row["broker_shares"]) <= 0.0 < float(row["paper_shares"]):
            return "PAPER_ONLY"
        return "MISMATCH"

    merged["status"] = merged.apply(_status, axis=1)
    return merged.loc[:, POSITION_RECON_COLUMNS].sort_values(["status", "symbol"]).reset_index(drop=True)


def reconcile_paper_with_broker(
    settings: Settings,
    *,
    strategy: str,
    broker: BrokerAdapter,
    trade_date: Any = None,
    cash_tolerance: float = 1.0,
    asset_tolerance: float = 1.0,
    share_tolerance: float = 0.0,
) -> dict[str, Any]:
    """
    对账纸面账户和只读券商账户。

    :return: dict，包含 account_summary、position_diff、issues、paper/broker 原始快照。
    """
    broker.sync()
    paper_cash, paper_positions = load_account_state(settings, strategy=strategy, default_cash=0.0)
    paper_snapshot = _latest_snapshot(settings, strategy, trade_date=trade_date)
    broker_account = broker.get_account()
    broker_positions = broker.get_positions()

    paper_market_value = float(paper_snapshot.get("market_value", 0.0) or 0.0)
    paper_total_asset = float(paper_snapshot.get("total_asset", paper_cash + paper_market_value) or 0.0)
    broker_cash = float(broker_account.cash)
    broker_market_value = float(broker_account.market_value)
    broker_total_asset = float(broker_account.total_asset)

    cash_diff = broker_cash - float(paper_cash)
    market_value_diff = broker_market_value - paper_market_value
    total_asset_diff = broker_total_asset - paper_total_asset
    cash_status = "OK" if abs(cash_diff) <= float(cash_tolerance) else "MISMATCH"
    asset_status = "OK" if abs(total_asset_diff) <= float(asset_tolerance) else "MISMATCH"
    trade_date_s = _date_to_str(trade_date or paper_snapshot.get("date", broker_account.updated_at))

    account_summary = pd.DataFrame(
        [
            {
                "trade_date": trade_date_s,
                "strategy": strategy,
                "paper_cash": float(paper_cash),
                "broker_cash": broker_cash,
                "cash_diff": cash_diff,
                "paper_market_value": paper_market_value,
                "broker_market_value": broker_market_value,
                "market_value_diff": market_value_diff,
                "paper_total_asset": paper_total_asset,
                "broker_total_asset": broker_total_asset,
                "total_asset_diff": total_asset_diff,
                "cash_status": cash_status,
                "asset_status": asset_status,
            }
        ],
        columns=ACCOUNT_RECON_COLUMNS,
    )
    position_diff = build_position_reconciliation(
        paper_positions,
        broker_positions,
        share_tolerance=share_tolerance,
    )
    issues: list[str] = []
    if cash_status != "OK":
        issues.append("cash_mismatch")
    if asset_status != "OK":
        issues.append("asset_mismatch")
    bad_positions = position_diff[position_diff["status"] != "OK"]
    if not bad_positions.empty:
        issues.append("position_mismatch")

    return {
        "strategy": strategy,
        "trade_date": trade_date_s,
        "account_summary": account_summary,
        "position_diff": position_diff,
        "issues": issues,
        "paper_positions": paper_positions,
        "broker_positions": broker_positions,
        "paper_snapshot": paper_snapshot,
        "broker_account": broker_account,
    }


def reconciliation_dir(settings: Settings, strategy: str) -> Path:
    safe = str(strategy).replace("/", "_")
    return settings.output_dir / "broker_reconciliation" / safe


def _markdown_table(frame: pd.DataFrame, *, max_rows: int = 30) -> str:
    if frame.empty:
        return "无\n"
    rows = frame.head(max_rows)
    lines = [
        "| " + " | ".join(rows.columns.astype(str)) + " |",
        "| " + " | ".join(["---"] * len(rows.columns)) + " |",
    ]
    for rec in rows.to_dict("records"):
        values: list[str] = []
        for col in rows.columns:
            value = rec.get(col, "")
            if isinstance(value, float):
                values.append("%.4f" % value)
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        lines.append("")
        lines.append("仅展示前 %d 行，共 %d 行。" % (max_rows, len(frame)))
    return "\n".join(lines) + "\n"


def build_reconciliation_report(result: dict[str, Any]) -> str:
    """生成纸面账户与券商只读账户对账 Markdown。"""
    strategy = str(result["strategy"])
    trade_date = str(result.get("trade_date", ""))
    account_summary = result["account_summary"]
    position_diff = result["position_diff"]
    issues = list(result.get("issues", []) or [])
    status = "OK" if not issues else "WARNING"
    bad_positions = position_diff[position_diff["status"] != "OK"] if not position_diff.empty else position_diff

    lines = [
        "# 纸面账户 / 真实账户只读对账 - %s - %s" % (strategy, trade_date),
        "",
        "## 对账摘要",
        "",
        "- 状态：`%s`" % status,
        "- 问题：%s" % (", ".join(issues) if issues else "无"),
        "",
        "## 账户差异",
        "",
        _markdown_table(account_summary),
        "## 持仓差异",
        "",
        _markdown_table(position_diff),
    ]
    if not bad_positions.empty:
        lines.extend(
            [
                "## 需要关注的持仓",
                "",
                _markdown_table(bad_positions),
            ]
        )
    lines.extend(
        [
            "## 说明",
            "",
            "本报告只做只读对账，不提交订单，也不撤单。若差异为 WARNING，应先人工确认真实账户、纸面账户和价格口径，再考虑后续交易。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def save_reconciliation_outputs(settings: Settings, result: dict[str, Any]) -> dict[str, Path]:
    """保存对账 CSV 和 Markdown 报告。"""
    base = reconciliation_dir(settings, str(result["strategy"]))
    base.mkdir(parents=True, exist_ok=True)
    trade_date = str(result.get("trade_date", "") or "latest")
    account_path = base / ("%s_account_summary.csv" % trade_date)
    positions_path = base / ("%s_position_diff.csv" % trade_date)
    report_path = base / ("%s.md" % trade_date)
    result["account_summary"].to_csv(account_path, index=False)
    result["position_diff"].to_csv(positions_path, index=False)
    report_path.write_text(build_reconciliation_report(result), encoding="utf-8")
    return {
        "account_summary": account_path,
        "position_diff": positions_path,
        "report": report_path,
    }
