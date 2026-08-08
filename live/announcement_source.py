"""真实公告数据源接入：把上游公告接口统一成 announcement_events 表。"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from config import get_tushare_token
from factors.factor_events import normalize_announcement_events


TUSHARE_ANNOUNCEMENT_COLUMNS = [
    "event_date",
    "symbol",
    "event_type",
    "title",
    "event_score",
    "risk_level",
    "source",
]


def _norm_date(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        if text.isdigit() and len(text) == 8:
            return text
    return pd.Timestamp(value).strftime("%Y%m%d")


def normalize_tushare_announcements(frame: pd.DataFrame | None) -> pd.DataFrame:
    """把 Tushare 公告接口返回值转成统一公告事件表。"""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=TUSHARE_ANNOUNCEMENT_COLUMNS)
    raw = frame.copy()
    rename: dict[str, str] = {}
    if "ts_code" in raw.columns:
        rename["ts_code"] = "symbol"
    if "ann_date" in raw.columns:
        rename["ann_date"] = "event_date"
    elif "pub_time" in raw.columns:
        rename["pub_time"] = "event_date"
    elif "datetime" in raw.columns:
        rename["datetime"] = "event_date"
    if "ann_type" in raw.columns:
        rename["ann_type"] = "event_type"
    elif "type" in raw.columns:
        rename["type"] = "event_type"
    raw = raw.rename(columns=rename)
    if "event_type" not in raw.columns:
        raw["event_type"] = "announcement"
    if "source" not in raw.columns:
        raw["source"] = "tushare"
    return normalize_announcement_events(raw).reindex(columns=TUSHARE_ANNOUNCEMENT_COLUMNS)


def normalize_akshare_cninfo_announcements(frame: pd.DataFrame | None) -> pd.DataFrame:
    """把 AkShare 巨潮公告返回值转成统一公告事件表。"""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=TUSHARE_ANNOUNCEMENT_COLUMNS)
    raw = frame.copy()
    rename: dict[str, str] = {}
    if "代码" in raw.columns:
        rename["代码"] = "symbol"
    elif "证券代码" in raw.columns:
        rename["证券代码"] = "symbol"
    if "公告时间" in raw.columns:
        rename["公告时间"] = "event_date"
    elif "公告日期" in raw.columns:
        rename["公告日期"] = "event_date"
    if "公告标题" in raw.columns:
        rename["公告标题"] = "title"
    elif "标题" in raw.columns:
        rename["标题"] = "title"
    if "公告类型" in raw.columns:
        rename["公告类型"] = "event_type"
    elif "分类" in raw.columns:
        rename["分类"] = "event_type"
    raw = raw.rename(columns=rename)
    if "event_type" not in raw.columns:
        raw["event_type"] = "announcement"
    if "source" not in raw.columns:
        raw["source"] = "akshare_cninfo"
    return normalize_announcement_events(raw).reindex(columns=TUSHARE_ANNOUNCEMENT_COLUMNS)


def _call_anns_endpoint(pro: Any, *, symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    调用 Tushare 公告接口。

    Tushare 不同版本/权限下公告接口参数可能略有差异，因此这里做几种参数组合兜底。
    """
    endpoint = None
    for name in ("anns_d", "anns"):
        if hasattr(pro, name):
            endpoint = getattr(pro, name)
            break
    if endpoint is None:
        raise AttributeError("当前 Tushare pro 对象没有 anns_d/anns 公告接口")
    attempts = [
        {"ts_code": symbol, "start_date": start, "end_date": end},
        {"ts_code": symbol, "ann_date": ""},
        {"start_date": start, "end_date": end},
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            df = endpoint(**kwargs)
        except Exception as exc:
            last_error = exc
            continue
        if df is None or df.empty:
            continue
        out = df.copy()
        if "ts_code" in out.columns and symbol:
            out = out[out["ts_code"].astype(str) == str(symbol)]
        if "ann_date" in out.columns:
            ann_date = pd.to_datetime(out["ann_date"], format="%Y%m%d", errors="coerce")
            start_dt = pd.to_datetime(start, format="%Y%m%d", errors="coerce")
            end_dt = pd.to_datetime(end, format="%Y%m%d", errors="coerce")
            out = out[(ann_date >= start_dt) & (ann_date <= end_dt)]
        return out
    if last_error is not None:
        raise last_error
    return pd.DataFrame()


def fetch_tushare_announcement_events(
    symbols: Iterable[str],
    start: Any,
    end: Any,
    *,
    token: str | None = None,
    pro: Any | None = None,
) -> pd.DataFrame:
    """
    从 Tushare 拉取公告并标准化成 announcement_events 表。

    该函数只负责公告源接入，不做风险过滤；风险识别交给 `live.event_risk_filter`。
    """
    start_s = _norm_date(start)
    end_s = _norm_date(end)
    if not start_s or not end_s:
        raise ValueError("start/end 不能为空")
    if pro is None:
        try:
            import tushare as ts
        except ImportError as exc:
            raise ImportError("需要安装 tushare: pip install tushare") from exc
        tok = (token or "").strip() or get_tushare_token()
        pro = ts.pro_api(tok)

    frames: list[pd.DataFrame] = []
    seen: set[str] = set()
    for symbol in symbols:
        sym = str(symbol).strip()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        df = _call_anns_endpoint(pro, symbol=sym, start=start_s, end=end_s)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=TUSHARE_ANNOUNCEMENT_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return normalize_tushare_announcements(out)


def fetch_akshare_cninfo_announcement_events(
    symbols: Iterable[str],
    start: Any,
    end: Any,
    *,
    category: str = "",
    keyword: str = "",
    market: str = "沪深京",
    fetcher: Any | None = None,
) -> pd.DataFrame:
    """
    从 AkShare 巨潮资讯公告接口拉取公告并标准化。

    AkShare 的 `stock_zh_a_disclosure_report_cninfo` 使用不带交易所后缀的 6 位股票代码。
    """
    start_s = _norm_date(start)
    end_s = _norm_date(end)
    if not start_s or not end_s:
        raise ValueError("start/end 不能为空")
    if fetcher is None:
        try:
            import akshare as ak
        except ImportError as exc:
            raise ImportError("需要安装 akshare: pip install akshare") from exc
        fetcher = ak.stock_zh_a_disclosure_report_cninfo

    frames: list[pd.DataFrame] = []
    seen: set[str] = set()
    for symbol in symbols:
        sym = str(symbol).strip()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        code = sym.split(".", 1)[0]
        try:
            df = fetcher(
                symbol=code,
                market=market,
                keyword=keyword,
                category=category,
                start_date=start_s,
                end_date=end_s,
            )
        except Exception:
            continue
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=TUSHARE_ANNOUNCEMENT_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return normalize_akshare_cninfo_announcements(out)


def save_announcement_events(frame: pd.DataFrame, path: Path) -> Path:
    """保存公告事件表。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalize_announcement_events(frame).to_csv(path, index=False)
    return path
