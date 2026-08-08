"""统一风险门禁：合并公告风险、负面舆情和人工黑名单。"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from live.event_risk_filter import build_event_risk_candidates
from live.negative_sentiment_filter import build_negative_sentiment_candidates
from live.risk_blacklist import RISK_BLACKLIST_COLUMNS, active_risk_blacklist
from live.stock_pool import normalize_ts_code


RISK_GATE_COLUMNS = [
    "trade_date",
    "symbol",
    "name",
    "gate_status",
    "severity",
    "risk_count",
    "block_count",
    "watch_count",
    "sources",
    "reason",
    "latest_triggered_at",
    "expires_at",
    "details",
]

RISK_GATE_DETAIL_COLUMNS = [
    "trade_date",
    "symbol",
    "name",
    "gate_status",
    "source",
    "reason",
    "triggered_at",
    "expires_at",
    "raw_action",
]

_WATCH_SEVERITIES = {"WATCH", "MEDIUM", "WARN", "WARNING", "观察", "中"}
_BLOCK_SEVERITIES = {"HIGH", "BLOCK", "BLACKLIST", "严重", "高"}
_STATUS_RANK = {"BLOCK": 0, "WATCH": 1, "PASS": 2}


def _empty_gate() -> pd.DataFrame:
    return pd.DataFrame(columns=RISK_GATE_COLUMNS)


def _empty_detail() -> pd.DataFrame:
    return pd.DataFrame(columns=RISK_GATE_DETAIL_COLUMNS)


def _name_map(symbols: pd.DataFrame | Iterable[str] | None) -> dict[str, str]:
    if symbols is None:
        return {}
    if isinstance(symbols, pd.DataFrame):
        if symbols.empty:
            return {}
        symbol_col = "symbol" if "symbol" in symbols.columns else "ts_code"
        if symbol_col not in symbols.columns:
            symbol_col = "股票代码" if "股票代码" in symbols.columns else symbol_col
        name_col = "name" if "name" in symbols.columns else "股票简称"
        out: dict[str, str] = {}
        for rec in symbols.to_dict("records"):
            symbol = normalize_ts_code(rec.get(symbol_col, ""))
            if not symbol:
                continue
            out[symbol] = str(rec.get(name_col, "") or "")
        return out
    return {normalize_ts_code(symbol): "" for symbol in symbols if normalize_ts_code(symbol)}


def _is_active_window(
    frame: pd.DataFrame,
    *,
    trade_date: pd.Timestamp,
    start_col: str,
    end_col: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    start = pd.to_datetime(out[start_col], errors="coerce", format="mixed")
    end = pd.to_datetime(out[end_col], errors="coerce", format="mixed")
    active = (start.isna() | (start <= trade_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))) & (
        end.isna() | (end >= trade_date)
    )
    return out[active].copy()


def _gate_status_from_action(action: Any, severity: Any = None) -> str:
    action_text = str(action or "").strip().upper()
    severity_text = str(severity or "").strip().upper()
    if action_text in {"BLACKLIST", "BLOCK"} or severity_text in _BLOCK_SEVERITIES:
        return "BLOCK"
    if action_text in {"WATCH", "WARN"} or severity_text in _WATCH_SEVERITIES:
        return "WATCH"
    return "PASS"


def _manual_detail_rows(
    records: pd.DataFrame | Mapping[str, Any] | Iterable[str] | None,
    *,
    trade_date: pd.Timestamp,
    names: dict[str, str],
) -> pd.DataFrame:
    manual = active_risk_blacklist(records, trade_date=trade_date)
    if manual.empty:
        return _empty_detail()
    rows: list[dict[str, Any]] = []
    for rec in manual.to_dict("records"):
        symbol = normalize_ts_code(rec.get("symbol", ""))
        if not symbol:
            continue
        status = _gate_status_from_action("", rec.get("severity", "HIGH"))
        created_at = rec.get("created_at", "")
        expires_at = rec.get("expires_at", "")
        rows.append(
            {
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "symbol": symbol,
                "name": str(rec.get("name", "") or names.get(symbol, "")),
                "gate_status": status,
                "source": str(rec.get("source", "manual") or "manual"),
                "reason": str(rec.get("reason", "manual_blacklist") or "manual_blacklist"),
                "triggered_at": created_at,
                "expires_at": expires_at,
                "raw_action": str(rec.get("severity", "") or ""),
            }
        )
    return pd.DataFrame(rows, columns=RISK_GATE_DETAIL_COLUMNS)


def _event_detail_rows(
    candidates: pd.DataFrame | None,
    *,
    trade_date: pd.Timestamp,
    names: dict[str, str],
) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return _empty_detail()
    frame = candidates.copy()
    if "risk_action" not in frame.columns:
        frame = build_event_risk_candidates(frame, as_of_date=trade_date)
    frame = frame[frame["risk_action"].astype(str).str.upper().isin({"BLACKLIST", "WATCH"})].copy()
    frame = _is_active_window(frame, trade_date=trade_date, start_col="event_date", end_col="blacklist_until")
    rows: list[dict[str, Any]] = []
    for rec in frame.to_dict("records"):
        symbol = normalize_ts_code(rec.get("symbol", ""))
        if not symbol:
            continue
        rows.append(
            {
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "symbol": symbol,
                "name": names.get(symbol, ""),
                "gate_status": _gate_status_from_action(rec.get("risk_action", "")),
                "source": "announcement_event:%s" % str(rec.get("source", "") or "unknown"),
                "reason": str(rec.get("risk_reason", "") or rec.get("title", "") or "announcement_event_risk"),
                "triggered_at": rec.get("event_date", ""),
                "expires_at": rec.get("blacklist_until", ""),
                "raw_action": str(rec.get("risk_action", "") or ""),
            }
        )
    return pd.DataFrame(rows, columns=RISK_GATE_DETAIL_COLUMNS)


def _sentiment_detail_rows(
    candidates: pd.DataFrame | None,
    *,
    trade_date: pd.Timestamp,
    names: dict[str, str],
) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return _empty_detail()
    frame = candidates.copy()
    if "risk_action" not in frame.columns:
        frame = build_negative_sentiment_candidates(frame, as_of_date=trade_date)
    frame = frame[frame["risk_action"].astype(str).str.upper().isin({"BLACKLIST", "WATCH"})].copy()
    frame = _is_active_window(frame, trade_date=trade_date, start_col="publish_time", end_col="blacklist_until")
    rows: list[dict[str, Any]] = []
    for rec in frame.to_dict("records"):
        symbol = normalize_ts_code(rec.get("symbol", ""))
        if not symbol:
            continue
        keywords = str(rec.get("negative_keywords", "") or "")
        reason = str(rec.get("risk_reason", "") or "negative_sentiment_risk")
        if keywords:
            reason = "%s:%s" % (reason, keywords)
        rows.append(
            {
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "symbol": symbol,
                "name": names.get(symbol, ""),
                "gate_status": _gate_status_from_action(rec.get("risk_action", "")),
                "source": "negative_sentiment:%s" % str(rec.get("source", "") or "unknown"),
                "reason": reason,
                "triggered_at": rec.get("publish_time", ""),
                "expires_at": rec.get("blacklist_until", ""),
                "raw_action": str(rec.get("risk_action", "") or ""),
            }
        )
    return pd.DataFrame(rows, columns=RISK_GATE_DETAIL_COLUMNS)


def build_unified_risk_gate(
    *,
    trade_date: Any,
    symbols: pd.DataFrame | Iterable[str] | None = None,
    manual_blacklist: pd.DataFrame | Mapping[str, Any] | Iterable[str] | None = None,
    event_candidates: pd.DataFrame | None = None,
    sentiment_candidates: pd.DataFrame | None = None,
    include_pass: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    合并多类风险输入，输出统一门禁表和风险明细表。

    `gate_status` 的优先级为 `BLOCK > WATCH > PASS`。只要某只股票被任一来源
    判定为 `BLOCK`，最终门禁就是 `BLOCK`；否则如果存在 `WATCH`，就是观察。
    """
    dt = pd.Timestamp(trade_date).normalize()
    names = _name_map(symbols)
    detail_parts = [
        _manual_detail_rows(manual_blacklist, trade_date=dt, names=names),
        _event_detail_rows(event_candidates, trade_date=dt, names=names),
        _sentiment_detail_rows(sentiment_candidates, trade_date=dt, names=names),
    ]
    details = pd.concat(detail_parts, ignore_index=True)
    if not details.empty:
        details = details.sort_values(["symbol", "gate_status", "triggered_at"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for symbol, group in details.groupby("symbol", sort=True):
        statuses = group["gate_status"].astype(str).str.upper()
        block_count = int((statuses == "BLOCK").sum())
        watch_count = int((statuses == "WATCH").sum())
        status = "BLOCK" if block_count else "WATCH"
        sources = sorted({str(x) for x in group["source"].dropna().tolist() if str(x)})
        reasons = [str(x) for x in group["reason"].dropna().tolist() if str(x)]
        latest = pd.to_datetime(group["triggered_at"], errors="coerce", format="mixed").max()
        expires = pd.to_datetime(group["expires_at"], errors="coerce", format="mixed").max()
        rows.append(
            {
                "trade_date": dt.strftime("%Y-%m-%d"),
                "symbol": symbol,
                "name": names.get(symbol, str(group["name"].dropna().iloc[-1]) if group["name"].notna().any() else ""),
                "gate_status": status,
                "severity": "HIGH" if status == "BLOCK" else "WATCH",
                "risk_count": int(len(group)),
                "block_count": block_count,
                "watch_count": watch_count,
                "sources": ";".join(sources),
                "reason": " | ".join(dict.fromkeys(reasons[:5])),
                "latest_triggered_at": latest.strftime("%Y-%m-%d") if pd.notna(latest) else "",
                "expires_at": expires.strftime("%Y-%m-%d") if pd.notna(expires) else "",
                "details": " || ".join(
                    "%s:%s" % (rec["source"], rec["reason"]) for rec in group.to_dict("records")
                ),
            }
        )

    if include_pass:
        risky = {row["symbol"] for row in rows}
        for symbol, name in sorted(names.items()):
            if symbol in risky:
                continue
            rows.append(
                {
                    "trade_date": dt.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "name": name,
                    "gate_status": "PASS",
                    "severity": "OK",
                    "risk_count": 0,
                    "block_count": 0,
                    "watch_count": 0,
                    "sources": "",
                    "reason": "no_active_risk_signal",
                    "latest_triggered_at": "",
                    "expires_at": "",
                    "details": "",
                }
            )

    gate = pd.DataFrame(rows, columns=RISK_GATE_COLUMNS)
    if gate.empty:
        return _empty_gate(), details
    gate["_rank"] = gate["gate_status"].map(_STATUS_RANK).fillna(9)
    gate = gate.sort_values(["_rank", "symbol"]).drop(columns="_rank").reset_index(drop=True)
    return gate, details


def risk_gate_to_blacklist(gate: pd.DataFrame, *, include_watch: bool = False) -> pd.DataFrame:
    """把统一门禁中的 BLOCK/WATCH 转成下单前预检查可读取的黑名单格式。"""
    if gate is None or gate.empty:
        return pd.DataFrame(columns=RISK_BLACKLIST_COLUMNS)
    keep = {"BLOCK", "WATCH"} if include_watch else {"BLOCK"}
    frame = gate[gate["gate_status"].astype(str).str.upper().isin(keep)].copy()
    if frame.empty:
        return pd.DataFrame(columns=RISK_BLACKLIST_COLUMNS)
    out = pd.DataFrame(
        {
            "symbol": frame["symbol"].map(normalize_ts_code),
            "name": frame["name"].fillna("").astype(str),
            "severity": frame["gate_status"].replace({"BLOCK": "HIGH", "WATCH": "WATCH"}),
            "reason": frame["reason"].fillna("").astype(str),
            "source": "unified_risk_gate",
            "active": True,
            "created_at": frame["trade_date"].fillna("").astype(str),
            "expires_at": frame["expires_at"].fillna("").astype(str),
        }
    )
    return out.reindex(columns=RISK_BLACKLIST_COLUMNS).reset_index(drop=True)


def load_risk_gate(path: Path | None, *, trade_date: Any = None) -> pd.DataFrame:
    """读取统一风险门禁文件；路径为空或文件不存在时返回空表。"""
    if path is None or not Path(path).exists():
        return _empty_gate()
    frame = pd.read_csv(Path(path))
    if frame.empty:
        return _empty_gate()
    out = frame.copy()
    for col in RISK_GATE_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out["symbol"] = out["symbol"].map(normalize_ts_code)
    out["gate_status"] = out["gate_status"].fillna("PASS").astype(str).str.upper()
    out = out[out["symbol"].astype(str).str.strip() != ""]
    if trade_date is not None and "trade_date" in out.columns:
        dt = pd.Timestamp(trade_date).normalize()
        dates = pd.to_datetime(out["trade_date"], errors="coerce")
        out = out[dates.dt.normalize() == dt].copy()
    return out.reindex(columns=RISK_GATE_COLUMNS).reset_index(drop=True)


def summarize_risk_gate_for_report(gate: pd.DataFrame | None) -> tuple[str, str]:
    """生成统一风险门禁的简短日报摘要。"""
    if gate is None or gate.empty:
        return "OK", "无风险门禁记录"
    status = gate["gate_status"].astype(str).str.upper()
    block = int((status == "BLOCK").sum())
    watch = int((status == "WATCH").sum())
    passed = int((status == "PASS").sum())
    if block > 0:
        return "BLOCK", "BLOCK=%d; WATCH=%d; PASS=%d" % (block, watch, passed)
    if watch > 0:
        return "WATCH", "WATCH=%d; PASS=%d" % (watch, passed)
    return "OK", "全部通过；PASS=%d" % passed
