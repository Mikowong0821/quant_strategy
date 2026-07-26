"""公告事件因子：把结构化公告事件表转换成日频事件分数。"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from live.stock_pool import normalize_ts_code


ANNOUNCEMENT_EVENT_SCORE = "ANNOUNCEMENT_EVENT_SCORE"

EVENT_COLUMNS = [
    "event_date",
    "symbol",
    "event_type",
    "title",
    "event_score",
    "risk_level",
    "source",
]

_SYMBOL_COLS = ("symbol", "ts_code", "股票代码", "证券代码", "代码")
_DATE_COLS = ("event_date", "ann_date", "publish_time", "datetime", "date", "公告日期", "披露日期")
_TYPE_COLS = ("event_type", "type", "category", "事件类型", "公告类型", "分类")
_TITLE_COLS = ("title", "event_title", "公告标题", "标题")
_SCORE_COLS = ("event_score", "score", "sentiment_score", "情绪分", "事件分")
_RISK_COLS = ("risk_level", "severity", "风险等级")
_SOURCE_COLS = ("source", "来源")


_POSITIVE_KEYWORDS = {
    "回购": 1.0,
    "增持": 0.8,
    "预增": 0.8,
    "扭亏": 0.8,
    "中标": 0.7,
    "合同": 0.5,
    "分红": 0.4,
    "派息": 0.4,
    "业绩快报": 0.4,
    "buyback": 1.0,
    "repurchase": 1.0,
    "increase": 0.6,
    "dividend": 0.4,
}

_NEGATIVE_KEYWORDS = {
    "减持": -0.8,
    "问询": -0.7,
    "处罚": -1.0,
    "立案": -1.0,
    "调查": -0.9,
    "诉讼": -0.8,
    "亏损": -0.8,
    "预减": -0.7,
    "商誉减值": -0.8,
    "质押": -0.5,
    "退市": -1.0,
    "风险警示": -1.0,
    "penalty": -1.0,
    "investigation": -1.0,
    "lawsuit": -0.8,
    "loss": -0.8,
    "reduction": -0.8,
}


def _first_existing(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    existing = set(columns)
    for col in candidates:
        if col in existing:
            return col
    return None


def _read_event_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _score_from_text(event_type: Any, title: Any) -> float:
    text = ("%s %s" % (event_type or "", title or "")).lower()
    score = 0.0
    for keyword, value in _POSITIVE_KEYWORDS.items():
        if keyword.lower() in text:
            score += value
    for keyword, value in _NEGATIVE_KEYWORDS.items():
        if keyword.lower() in text:
            score += value
    return max(min(score, 2.0), -2.0)


def normalize_announcement_events(events: pd.DataFrame | None) -> pd.DataFrame:
    """把不同来源的公告事件表标准化成统一字段。"""
    if events is None or events.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    frame = events.copy()
    symbol_col = _first_existing(frame.columns, _SYMBOL_COLS)
    date_col = _first_existing(frame.columns, _DATE_COLS)
    if symbol_col is None or date_col is None:
        raise ValueError("公告事件表须包含 symbol/ts_code/股票代码 和 event_date/ann_date/公告日期")

    type_col = _first_existing(frame.columns, _TYPE_COLS)
    title_col = _first_existing(frame.columns, _TITLE_COLS)
    score_col = _first_existing(frame.columns, _SCORE_COLS)
    risk_col = _first_existing(frame.columns, _RISK_COLS)
    source_col = _first_existing(frame.columns, _SOURCE_COLS)

    out = pd.DataFrame()
    out["event_date"] = pd.to_datetime(frame[date_col], errors="coerce")
    out["symbol"] = frame[symbol_col].map(normalize_ts_code)
    out["event_type"] = frame[type_col].fillna("").astype(str) if type_col else ""
    out["title"] = frame[title_col].fillna("").astype(str) if title_col else ""
    if score_col:
        out["event_score"] = pd.to_numeric(frame[score_col], errors="coerce")
    else:
        out["event_score"] = [
            _score_from_text(event_type, title)
            for event_type, title in zip(out["event_type"], out["title"], strict=True)
        ]
    out["risk_level"] = frame[risk_col].fillna("").astype(str) if risk_col else ""
    out["source"] = frame[source_col].fillna("announcement").astype(str) if source_col else "announcement"
    out = out.dropna(subset=["event_date"])
    out = out[out["symbol"].astype(str).str.strip() != ""]
    out["event_score"] = out["event_score"].replace([float("inf"), float("-inf")], pd.NA)
    out = out.dropna(subset=["event_score"])
    out["event_score"] = out["event_score"].astype(float).clip(-2.0, 2.0)
    return out.reindex(columns=EVENT_COLUMNS).sort_values(["event_date", "symbol"]).reset_index(drop=True)


def load_announcement_events(path: Path | None) -> pd.DataFrame:
    """读取公告事件文件；文件不存在时返回空表。"""
    if path is None or not Path(path).expanduser().exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return normalize_announcement_events(_read_event_table(Path(path).expanduser()))


def calc_announcement_event_score(
    events: pd.DataFrame | None,
    prices_long: pd.DataFrame,
    *,
    effective_days: int = 20,
    date_col: str = "trade_date",
    symbol_col: str = "ts_code",
) -> pd.Series:
    """
    将公告事件转成日频事件得分。

    每条事件只在公告日之后生效，向后 `effective_days` 个交易日线性衰减。
    正事件加分，负事件扣分；同一股票同一日多条事件会累加。
    """
    if effective_days <= 0:
        raise ValueError("effective_days 必须为正整数")
    need = {date_col, symbol_col}
    missing = need - set(prices_long.columns)
    if missing:
        raise ValueError("prices_long 缺少列: %s" % ", ".join(sorted(missing)))

    px = prices_long[[date_col, symbol_col]].copy()
    px[date_col] = pd.to_datetime(px[date_col], errors="coerce")
    px[symbol_col] = px[symbol_col].astype(str)
    px = px.dropna(subset=[date_col]).drop_duplicates().sort_values([symbol_col, date_col])
    index = pd.MultiIndex.from_frame(
        px.rename(columns={date_col: "date", symbol_col: "symbol"})[["date", "symbol"]]
    )
    out = pd.Series(0.0, index=index, name=ANNOUNCEMENT_EVENT_SCORE)
    norm_events = normalize_announcement_events(events)
    if norm_events.empty or out.empty:
        return out.astype(float).sort_index()

    date_by_symbol = {
        symbol: group[date_col].sort_values().reset_index(drop=True)
        for symbol, group in px.groupby(symbol_col, sort=False)
    }
    updates: dict[tuple[pd.Timestamp, str], float] = {}
    for rec in norm_events.to_dict("records"):
        symbol = str(rec.get("symbol", ""))
        if symbol not in date_by_symbol:
            continue
        event_date = pd.Timestamp(rec["event_date"])
        score = float(rec.get("event_score", 0.0))
        if not math.isfinite(score) or abs(score) <= 1e-12:
            continue
        dates = date_by_symbol[symbol]
        future_dates = dates[dates >= event_date].head(effective_days)
        if future_dates.empty:
            continue
        n = len(future_dates)
        for i, dt in enumerate(future_dates):
            decay = 1.0 - (i / max(n, 1))
            key = (pd.Timestamp(dt), symbol)
            updates[key] = updates.get(key, 0.0) + score * decay

    if updates:
        upd = pd.Series(updates, dtype=float)
        upd.index = pd.MultiIndex.from_tuples(upd.index, names=["date", "symbol"])
        out = out.add(upd.reindex(out.index).fillna(0.0), fill_value=0.0)
    out = out.astype(float).sort_index()
    out.name = ANNOUNCEMENT_EVENT_SCORE
    return out
