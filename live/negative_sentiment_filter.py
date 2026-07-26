"""负面舆情过滤：把新闻/舆情文本转成风险候选或黑名单候选。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from live.stock_pool import normalize_ts_code


NEGATIVE_SENTIMENT_COLUMNS = [
    "publish_time",
    "symbol",
    "title",
    "content",
    "sentiment_score",
    "negative_keywords",
    "source",
    "risk_action",
    "risk_reason",
    "blacklist_until",
]

_SYMBOL_COLS = ("symbol", "ts_code", "股票代码", "证券代码", "代码")
_TIME_COLS = ("publish_time", "datetime", "time", "date", "发布时间", "日期")
_TITLE_COLS = ("title", "标题", "news_title")
_CONTENT_COLS = ("content", "正文", "summary", "摘要", "text")
_SCORE_COLS = ("sentiment_score", "score", "情绪分", "舆情分")
_SOURCE_COLS = ("source", "src", "来源")

_HIGH_NEGATIVE_KEYWORDS = {
    "立案",
    "处罚",
    "退市",
    "暴雷",
    "造假",
    "违规",
    "调查",
    "重大诉讼",
    "债务违约",
    "风险警示",
    "penalty",
    "fraud",
    "investigation",
    "default",
    "delisting",
}

_WATCH_NEGATIVE_KEYWORDS = {
    "问询",
    "诉讼",
    "减持",
    "亏损",
    "预亏",
    "预减",
    "质押",
    "商誉减值",
    "下调",
    "监管函",
    "lawsuit",
    "loss",
    "reduction",
    "downgrade",
}


def _first_existing(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
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


def _keyword_hits(text: str, keywords: set[str]) -> list[str]:
    lower = text.lower()
    return [kw for kw in sorted(keywords) if kw.lower() in lower]


def _keyword_sentiment_score(text: str) -> tuple[float, list[str]]:
    high = _keyword_hits(text, _HIGH_NEGATIVE_KEYWORDS)
    watch = _keyword_hits(text, _WATCH_NEGATIVE_KEYWORDS)
    score = -1.0 * len(high) - 0.4 * len(watch)
    return max(score, -3.0), high + watch


def normalize_sentiment_items(items: pd.DataFrame | None) -> pd.DataFrame:
    """标准化新闻/舆情表。"""
    if items is None or items.empty:
        return pd.DataFrame(columns=NEGATIVE_SENTIMENT_COLUMNS)
    frame = items.copy()
    symbol_col = _first_existing(frame.columns, _SYMBOL_COLS)
    time_col = _first_existing(frame.columns, _TIME_COLS)
    if symbol_col is None or time_col is None:
        raise ValueError("舆情表须包含 symbol/ts_code/股票代码 和 publish_time/date/发布时间")
    title_col = _first_existing(frame.columns, _TITLE_COLS)
    content_col = _first_existing(frame.columns, _CONTENT_COLS)
    score_col = _first_existing(frame.columns, _SCORE_COLS)
    source_col = _first_existing(frame.columns, _SOURCE_COLS)

    out = pd.DataFrame()
    out["publish_time"] = pd.to_datetime(frame[time_col], errors="coerce")
    out["symbol"] = frame[symbol_col].map(normalize_ts_code)
    out["title"] = frame[title_col].fillna("").astype(str) if title_col else ""
    out["content"] = frame[content_col].fillna("").astype(str) if content_col else ""
    if score_col:
        out["sentiment_score"] = pd.to_numeric(frame[score_col], errors="coerce")
        keyword_pairs = [
            _keyword_sentiment_score("%s %s" % (title, content))[1]
            for title, content in zip(out["title"], out["content"], strict=True)
        ]
    else:
        pairs = [
            _keyword_sentiment_score("%s %s" % (title, content))
            for title, content in zip(out["title"], out["content"], strict=True)
        ]
        out["sentiment_score"] = [score for score, _ in pairs]
        keyword_pairs = [hits for _, hits in pairs]
    out["negative_keywords"] = [";".join(hits) for hits in keyword_pairs]
    out["source"] = frame[source_col].fillna("sentiment").astype(str) if source_col else "sentiment"
    out = out.dropna(subset=["publish_time"])
    out = out[out["symbol"].astype(str).str.strip() != ""]
    out["sentiment_score"] = pd.to_numeric(out["sentiment_score"], errors="coerce")
    out = out.dropna(subset=["sentiment_score"])
    out["risk_action"] = "NONE"
    out["risk_reason"] = "not_evaluated"
    out["blacklist_until"] = pd.NaT
    return out.reindex(columns=NEGATIVE_SENTIMENT_COLUMNS).sort_values(["publish_time", "symbol"]).reset_index(drop=True)


def load_sentiment_items(path: Path | None) -> pd.DataFrame:
    """读取新闻/舆情 CSV/XLSX。"""
    if path is None or not Path(path).expanduser().exists():
        return pd.DataFrame(columns=NEGATIVE_SENTIMENT_COLUMNS)
    return normalize_sentiment_items(_read_table(Path(path).expanduser()))


def build_negative_sentiment_candidates(
    items: pd.DataFrame,
    *,
    as_of_date: Any | None = None,
    lookback_days: int = 7,
    block_days: int = 10,
    watch_days: int = 5,
    block_score_threshold: float = -1.0,
    watch_score_threshold: float = -0.4,
) -> pd.DataFrame:
    """把负面新闻/舆情转成 WATCH / BLACKLIST 候选。"""
    if lookback_days < 0:
        raise ValueError("lookback_days 不能为负")
    if block_days <= 0 or watch_days <= 0:
        raise ValueError("block_days/watch_days 必须为正整数")
    frame = normalize_sentiment_items(items)
    if frame.empty:
        return frame
    if as_of_date is not None:
        dt = pd.Timestamp(as_of_date).normalize()
        start = dt - pd.Timedelta(days=int(lookback_days))
        pub = pd.to_datetime(frame["publish_time"], errors="coerce")
        frame = frame[(pub <= dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)) & (pub >= start)].copy()
    if frame.empty:
        return frame

    actions: list[str] = []
    reasons: list[str] = []
    untils: list[pd.Timestamp | pd.NaT] = []
    for rec in frame.to_dict("records"):
        score = float(rec.get("sentiment_score", 0.0))
        keywords = str(rec.get("negative_keywords", "") or "")
        has_high = any(kw in keywords for kw in _HIGH_NEGATIVE_KEYWORDS)
        if score <= block_score_threshold or has_high:
            actions.append("BLACKLIST")
            reasons.append("negative_sentiment_block")
            untils.append(pd.Timestamp(rec["publish_time"]) + pd.Timedelta(days=int(block_days)))
        elif score <= watch_score_threshold or keywords:
            actions.append("WATCH")
            reasons.append("negative_sentiment_watch")
            untils.append(pd.Timestamp(rec["publish_time"]) + pd.Timedelta(days=int(watch_days)))
        else:
            actions.append("NONE")
            reasons.append("no_negative_signal")
            untils.append(pd.NaT)

    frame["risk_action"] = actions
    frame["risk_reason"] = reasons
    frame["blacklist_until"] = untils
    out = frame[frame["risk_action"].isin({"BLACKLIST", "WATCH"})].copy()
    return out.reindex(columns=NEGATIVE_SENTIMENT_COLUMNS).sort_values(["publish_time", "symbol"]).reset_index(drop=True)


def negative_sentiment_candidates_to_blacklist(
    candidates: pd.DataFrame,
    *,
    include_watch: bool = False,
) -> pd.DataFrame:
    """把负面舆情候选转成 risk_blacklist 格式。"""
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
            "reason": frame["risk_reason"].astype(str) + ":" + frame["negative_keywords"].astype(str),
            "source": frame["source"].astype(str).replace({"": "negative_sentiment"}),
            "active": True,
            "created_at": pd.to_datetime(frame["publish_time"], errors="coerce").dt.strftime("%Y-%m-%d"),
            "expires_at": pd.to_datetime(frame["blacklist_until"], errors="coerce").dt.strftime("%Y-%m-%d"),
        }
    )
    out = out.sort_values(["symbol", "severity", "created_at"]).drop_duplicates(subset=["symbol"], keep="last")
    return out.reset_index(drop=True)
