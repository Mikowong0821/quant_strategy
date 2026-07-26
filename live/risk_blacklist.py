"""风险预警与黑名单：把人工或系统风险标记转成下单前约束。"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings
from live.stock_pool import normalize_ts_code


RISK_BLACKLIST_COLUMNS = [
    "symbol",
    "name",
    "severity",
    "reason",
    "source",
    "active",
    "created_at",
    "expires_at",
]


_SYMBOL_COLS = ("symbol", "ts_code", "股票代码", "证券代码", "代码")


def default_risk_blacklist_path(settings: Settings) -> Path:
    """默认黑名单文件位置。文件不存在时视为没有黑名单。"""
    return settings.data_dir / "risk_blacklist.csv"


def _to_bool(value: Any, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isnan(value):
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "nan", "none"}:
            return default
        return text in {"1", "true", "yes", "y", "on", "active", "启用", "是"}
    return bool(value)


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    existing = set(columns)
    for col in candidates:
        if col in existing:
            return col
    return None


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def normalize_risk_blacklist(
    records: pd.DataFrame | Mapping[str, Any] | Iterable[str] | None,
) -> pd.DataFrame:
    """
    标准化风险黑名单。

    支持三种输入：
    - DataFrame：包含 symbol/ts_code/股票代码 等任一代码列；
    - Mapping：`{"000001.SZ": "原因"}` 或 `{"000001.SZ": {"reason": "...", "severity": "HIGH"}}`；
    - Iterable[str]：只传代码，原因默认 `manual_blacklist`。
    """
    if records is None:
        return pd.DataFrame(columns=RISK_BLACKLIST_COLUMNS)
    if isinstance(records, pd.DataFrame):
        if records.empty:
            return pd.DataFrame(columns=RISK_BLACKLIST_COLUMNS)
        frame = records.copy()
        symbol_col = _first_existing(frame.columns, _SYMBOL_COLS)
        if symbol_col is None:
            raise ValueError("risk_blacklist 须包含 symbol/ts_code/股票代码/证券代码/代码 任一列")
        name_col = _first_existing(frame.columns, ("name", "股票名称", "证券名称", "名称"))
        reason_col = _first_existing(frame.columns, ("reason", "原因", "risk_reason", "blacklist_reason"))
        severity_col = _first_existing(frame.columns, ("severity", "risk_level", "level", "风险等级"))
        source_col = _first_existing(frame.columns, ("source", "来源"))
        active_col = _first_existing(frame.columns, ("active", "enabled", "is_active", "是否启用"))
        created_col = _first_existing(frame.columns, ("created_at", "start_date", "生效日期"))
        expires_col = _first_existing(frame.columns, ("expires_at", "end_date", "失效日期", "到期日期"))

        out = pd.DataFrame()
        out["symbol"] = frame[symbol_col].map(normalize_ts_code)
        out["name"] = frame[name_col].fillna("").astype(str) if name_col else ""
        out["severity"] = (
            frame[severity_col].fillna("HIGH").astype(str).str.upper()
            if severity_col
            else "HIGH"
        )
        out["reason"] = (
            frame[reason_col].fillna("manual_blacklist").astype(str)
            if reason_col
            else "manual_blacklist"
        )
        out["source"] = frame[source_col].fillna("manual").astype(str) if source_col else "manual"
        out["active"] = frame[active_col].map(_to_bool) if active_col else True
        out["created_at"] = frame[created_col] if created_col else ""
        out["expires_at"] = frame[expires_col] if expires_col else ""
    elif isinstance(records, Mapping):
        rows: list[dict[str, Any]] = []
        for symbol, value in records.items():
            if isinstance(value, Mapping):
                rows.append(
                    {
                        "symbol": normalize_ts_code(symbol),
                        "name": str(value.get("name", "")),
                        "severity": str(value.get("severity", "HIGH")).upper(),
                        "reason": str(value.get("reason", "manual_blacklist")),
                        "source": str(value.get("source", "manual")),
                        "active": _to_bool(value.get("active", True)),
                        "created_at": value.get("created_at", ""),
                        "expires_at": value.get("expires_at", ""),
                    }
                )
            else:
                rows.append(
                    {
                        "symbol": normalize_ts_code(symbol),
                        "name": "",
                        "severity": "HIGH",
                        "reason": str(value or "manual_blacklist"),
                        "source": "manual",
                        "active": True,
                        "created_at": "",
                        "expires_at": "",
                    }
                )
        out = pd.DataFrame(rows)
    else:
        out = pd.DataFrame(
            [
                {
                    "symbol": normalize_ts_code(symbol),
                    "name": "",
                    "severity": "HIGH",
                    "reason": "manual_blacklist",
                    "source": "manual",
                    "active": True,
                    "created_at": "",
                    "expires_at": "",
                }
                for symbol in records
            ]
        )

    if out.empty:
        return pd.DataFrame(columns=RISK_BLACKLIST_COLUMNS)
    out = out.copy()
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out = out[out["symbol"] != ""]
    out["severity"] = out["severity"].replace({"": "HIGH"}).astype(str).str.upper()
    out["reason"] = out["reason"].replace({"": "manual_blacklist"}).astype(str)
    out["source"] = out["source"].replace({"": "manual"}).astype(str)
    out["active"] = out["active"].map(_to_bool)
    out = out.drop_duplicates(subset=["symbol"], keep="last")
    return out.reindex(columns=RISK_BLACKLIST_COLUMNS).sort_values("symbol").reset_index(drop=True)


def load_risk_blacklist(path: Path | None) -> pd.DataFrame:
    """读取风险黑名单文件；路径为空或文件不存在时返回空表。"""
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=RISK_BLACKLIST_COLUMNS)
    return normalize_risk_blacklist(_read_table(Path(path)))


def active_risk_blacklist(
    records: pd.DataFrame | Mapping[str, Any] | Iterable[str] | None,
    *,
    trade_date: Any = None,
) -> pd.DataFrame:
    """按 active、生效日期和失效日期过滤当前有效黑名单。"""
    frame = normalize_risk_blacklist(records)
    if frame.empty:
        return frame
    out = frame[frame["active"].map(_to_bool)].copy()
    if trade_date is None or out.empty:
        return out.reset_index(drop=True)

    dt = pd.Timestamp(trade_date).normalize()
    created = pd.to_datetime(out["created_at"], errors="coerce")
    expires = pd.to_datetime(out["expires_at"], errors="coerce")
    created_ok = created.isna() | (created <= dt)
    expires_ok = expires.isna() | (expires >= dt)
    return out[created_ok & expires_ok].reset_index(drop=True)


def risk_blacklist_map(
    records: pd.DataFrame | Mapping[str, Any] | Iterable[str] | None,
    *,
    trade_date: Any = None,
) -> dict[str, dict[str, str]]:
    """返回按代码索引的有效黑名单明细，供订单预检查快速查询。"""
    frame = active_risk_blacklist(records, trade_date=trade_date)
    out: dict[str, dict[str, str]] = {}
    for rec in frame.to_dict("records"):
        symbol = str(rec.get("symbol", "")).strip()
        if not symbol:
            continue
        out[symbol] = {
            "severity": str(rec.get("severity", "HIGH") or "HIGH").upper(),
            "reason": str(rec.get("reason", "manual_blacklist") or "manual_blacklist"),
            "source": str(rec.get("source", "manual") or "manual"),
            "name": str(rec.get("name", "") or ""),
        }
    return out


def summarize_risk_blacklist_for_report(records: pd.DataFrame | None) -> tuple[str, str]:
    """把黑名单压缩成日报摘要。"""
    if records is None or records.empty:
        return "OK", "无有效黑名单"
    frame = normalize_risk_blacklist(records)
    active_count = int(frame["active"].map(_to_bool).sum()) if "active" in frame.columns else int(len(frame))
    if active_count <= 0:
        return "OK", "无有效黑名单"
    severity_counts = frame[frame["active"].map(_to_bool)]["severity"].astype(str).str.upper().value_counts().to_dict()
    parts = ["%s=%d" % (key, int(value)) for key, value in sorted(severity_counts.items())]
    return "WATCH", "有效黑名单 %d 只；%s" % (active_count, "，".join(parts))
