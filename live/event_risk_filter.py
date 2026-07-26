"""公告事件风险过滤：把公告事件转成观察名单或黑名单候选。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from factors.factor_events import load_announcement_events, normalize_announcement_events


EVENT_RISK_COLUMNS = [
    "event_date",
    "symbol",
    "event_type",
    "title",
    "event_score",
    "risk_level",
    "source",
    "risk_action",
    "risk_reason",
    "blacklist_until",
]

_BLOCK_KEYWORDS = {
    "处罚",
    "立案",
    "调查",
    "重大诉讼",
    "退市",
    "风险警示",
    "暂停上市",
    "penalty",
    "investigation",
    "delisting",
}

_WATCH_KEYWORDS = {
    "问询",
    "诉讼",
    "减持",
    "预减",
    "亏损",
    "商誉减值",
    "质押",
    "监管函",
    "reduction",
    "lawsuit",
    "loss",
}

_HIGH_RISK_LEVELS = {"HIGH", "BLOCK", "BLACKLIST", "严重", "高"}
_WATCH_RISK_LEVELS = {"WATCH", "MEDIUM", "WARN", "观察", "中"}


def _contains_any(text: str, keywords: set[str]) -> list[str]:
    lower = text.lower()
    return [kw for kw in sorted(keywords) if kw.lower() in lower]


def _risk_action_and_reason(
    *,
    event_type: Any,
    title: Any,
    event_score: Any,
    risk_level: Any,
    block_score_threshold: float,
    watch_score_threshold: float,
) -> tuple[str, str]:
    text = "%s %s" % (event_type or "", title or "")
    risk_text = str(risk_level or "").strip().upper()
    block_hits = _contains_any(text, _BLOCK_KEYWORDS)
    watch_hits = _contains_any(text, _WATCH_KEYWORDS)
    score = pd.to_numeric(pd.Series([event_score]), errors="coerce").iloc[0]
    score_f = float(score) if pd.notna(score) else 0.0

    reasons: list[str] = []
    if risk_text in _HIGH_RISK_LEVELS:
        reasons.append("risk_level_high")
    if block_hits:
        reasons.append("block_keyword:%s" % ",".join(block_hits))
    if score_f <= block_score_threshold:
        reasons.append("event_score_below_block_threshold")
    if reasons:
        return "BLACKLIST", "|".join(reasons)

    if risk_text in _WATCH_RISK_LEVELS:
        reasons.append("risk_level_watch")
    if watch_hits:
        reasons.append("watch_keyword:%s" % ",".join(watch_hits))
    if score_f <= watch_score_threshold:
        reasons.append("event_score_below_watch_threshold")
    if reasons:
        return "WATCH", "|".join(reasons)
    return "NONE", "no_risk_signal"


def build_event_risk_candidates(
    events: pd.DataFrame,
    *,
    as_of_date: Any | None = None,
    lookback_days: int = 60,
    block_days: int = 20,
    watch_days: int = 10,
    block_score_threshold: float = -0.8,
    watch_score_threshold: float = -0.3,
) -> pd.DataFrame:
    """从公告事件表生成风险候选。"""
    if lookback_days < 0:
        raise ValueError("lookback_days 不能为负")
    if block_days <= 0 or watch_days <= 0:
        raise ValueError("block_days/watch_days 必须为正整数")

    frame = normalize_announcement_events(events)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_RISK_COLUMNS)
    out = frame.copy()
    if as_of_date is not None:
        dt = pd.Timestamp(as_of_date).normalize()
        start = dt - pd.Timedelta(days=int(lookback_days))
        out = out[(out["event_date"] <= dt) & (out["event_date"] >= start)].copy()
    if out.empty:
        return pd.DataFrame(columns=EVENT_RISK_COLUMNS)

    actions: list[str] = []
    reasons: list[str] = []
    untils: list[pd.Timestamp | pd.NaT] = []
    for rec in out.to_dict("records"):
        action, reason = _risk_action_and_reason(
            event_type=rec.get("event_type", ""),
            title=rec.get("title", ""),
            event_score=rec.get("event_score", 0.0),
            risk_level=rec.get("risk_level", ""),
            block_score_threshold=block_score_threshold,
            watch_score_threshold=watch_score_threshold,
        )
        actions.append(action)
        reasons.append(reason)
        if action == "BLACKLIST":
            untils.append(pd.Timestamp(rec["event_date"]) + pd.Timedelta(days=int(block_days)))
        elif action == "WATCH":
            untils.append(pd.Timestamp(rec["event_date"]) + pd.Timedelta(days=int(watch_days)))
        else:
            untils.append(pd.NaT)

    out["risk_action"] = actions
    out["risk_reason"] = reasons
    out["blacklist_until"] = untils
    out = out[out["risk_action"].isin({"BLACKLIST", "WATCH"})].copy()
    return out.reindex(columns=EVENT_RISK_COLUMNS).sort_values(["event_date", "symbol"]).reset_index(drop=True)


def event_risk_candidates_to_blacklist(
    candidates: pd.DataFrame,
    *,
    include_watch: bool = False,
) -> pd.DataFrame:
    """把风险候选转成 `live.risk_blacklist` 可读取的黑名单格式。"""
    if candidates is None or candidates.empty:
        return pd.DataFrame(columns=["symbol", "name", "severity", "reason", "source", "active", "created_at", "expires_at"])
    frame = candidates.copy()
    keep = {"BLACKLIST", "WATCH"} if include_watch else {"BLACKLIST"}
    frame = frame[frame["risk_action"].isin(keep)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "name", "severity", "reason", "source", "active", "created_at", "expires_at"])
    out = pd.DataFrame(
        {
            "symbol": frame["symbol"].astype(str),
            "name": "",
            "severity": frame["risk_action"].replace({"BLACKLIST": "HIGH", "WATCH": "WATCH"}),
            "reason": frame["risk_reason"].astype(str),
            "source": frame["source"].astype(str).replace({"": "announcement_event"}),
            "active": True,
            "created_at": pd.to_datetime(frame["event_date"], errors="coerce").dt.strftime("%Y-%m-%d"),
            "expires_at": pd.to_datetime(frame["blacklist_until"], errors="coerce").dt.strftime("%Y-%m-%d"),
        }
    )
    out = out.sort_values(["symbol", "severity", "created_at"]).drop_duplicates(subset=["symbol"], keep="last")
    return out.reset_index(drop=True)


def load_event_risk_candidates(
    event_path: Path | None,
    *,
    as_of_date: Any | None = None,
    lookback_days: int = 60,
    block_days: int = 20,
    watch_days: int = 10,
    block_score_threshold: float = -0.8,
    watch_score_threshold: float = -0.3,
) -> pd.DataFrame:
    """从公告事件文件直接构造风险候选。"""
    events = load_announcement_events(event_path)
    return build_event_risk_candidates(
        events,
        as_of_date=as_of_date,
        lookback_days=lookback_days,
        block_days=block_days,
        watch_days=watch_days,
        block_score_threshold=block_score_threshold,
        watch_score_threshold=watch_score_threshold,
    )
