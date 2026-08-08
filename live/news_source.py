"""新闻 / 舆情数据源接入：把不同上游统一成 news_sentiment 表。"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any
import time

import pandas as pd

from live.negative_sentiment_filter import NEGATIVE_SENTIMENT_COLUMNS, normalize_sentiment_items
from live.stock_pool import normalize_ts_code


NEWS_ITEM_COLUMNS = list(NEGATIVE_SENTIMENT_COLUMNS)


def _plain_code(symbol: Any) -> str:
    code = normalize_ts_code(symbol)
    return code.split(".", 1)[0] if "." in code else str(symbol).strip()


def normalize_news_items(frame: pd.DataFrame | None) -> pd.DataFrame:
    """把任意新闻 / 舆情表标准化为工程内部 news_sentiment 表。"""
    return normalize_sentiment_items(frame).reindex(columns=NEWS_ITEM_COLUMNS)


def normalize_akshare_stock_news(frame: pd.DataFrame | None, *, symbol: Any) -> pd.DataFrame:
    """把 AkShare `stock_news_em` 返回值转成统一 news_sentiment 表。"""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=NEWS_ITEM_COLUMNS)
    raw = frame.copy()
    rename: dict[str, str] = {}
    if "关键词" in raw.columns:
        rename["关键词"] = "symbol"
    if "新闻标题" in raw.columns:
        rename["新闻标题"] = "title"
    if "新闻内容" in raw.columns:
        rename["新闻内容"] = "content"
    if "发布时间" in raw.columns:
        rename["发布时间"] = "publish_time"
    if "文章来源" in raw.columns:
        rename["文章来源"] = "source"
    if "新闻链接" in raw.columns:
        rename["新闻链接"] = "url"
    raw = raw.rename(columns=rename)
    raw["symbol"] = normalize_ts_code(symbol)
    if "source" not in raw.columns:
        raw["source"] = "akshare_stock_news_em"
    else:
        raw["source"] = raw["source"].fillna("").astype(str).replace({"": "akshare_stock_news_em"})
    return normalize_news_items(raw)


def normalize_tushare_news(frame: pd.DataFrame | None) -> pd.DataFrame:
    """
    把 Tushare 或类似新闻接口结果转成统一 news_sentiment 表。

    不同权限下新闻字段可能不同，这里只做常见字段兼容；真正拉取函数后续
    可在数据源层补，不影响因子层。
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=NEWS_ITEM_COLUMNS)
    raw = frame.copy()
    rename: dict[str, str] = {}
    if "ts_code" in raw.columns:
        rename["ts_code"] = "symbol"
    elif "stock_code" in raw.columns:
        rename["stock_code"] = "symbol"
    if "datetime" in raw.columns:
        rename["datetime"] = "publish_time"
    elif "publish_time" in raw.columns:
        rename["publish_time"] = "publish_time"
    elif "pub_time" in raw.columns:
        rename["pub_time"] = "publish_time"
    if "content" in raw.columns:
        rename["content"] = "content"
    elif "summary" in raw.columns:
        rename["summary"] = "content"
    if "title" in raw.columns:
        rename["title"] = "title"
    if "src" in raw.columns:
        rename["src"] = "source"
    if "url" in raw.columns:
        rename["url"] = "url"
    raw = raw.rename(columns=rename)
    if "source" not in raw.columns:
        raw["source"] = "tushare_news"
    return normalize_news_items(raw)


def merge_news_items(*frames: pd.DataFrame) -> pd.DataFrame:
    """合并多批新闻并按关键字段去重。"""
    parts = [normalize_news_items(frame) for frame in frames if frame is not None and not frame.empty]
    if not parts:
        return pd.DataFrame(columns=NEWS_ITEM_COLUMNS)
    out = pd.concat(parts, ignore_index=True)
    dedup_cols = ["symbol", "publish_time", "title", "source", "url"]
    out = out.sort_values(["publish_time", "symbol"])
    out = out.drop_duplicates(subset=dedup_cols, keep="last")
    return out.reindex(columns=NEWS_ITEM_COLUMNS).reset_index(drop=True)


def fetch_akshare_stock_news_items(
    symbols: Iterable[str],
    *,
    fetcher: Any | None = None,
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    """从 AkShare 东方财富个股新闻接口拉取最近新闻并标准化。"""
    if fetcher is None:
        try:
            import akshare as ak
        except ImportError as exc:
            raise ImportError("需要安装 akshare: pip install akshare") from exc
        fetcher = ak.stock_news_em

    frames: list[pd.DataFrame] = []
    seen: set[str] = set()
    for symbol in symbols:
        norm = normalize_ts_code(symbol)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        try:
            raw = fetcher(symbol=_plain_code(norm))
        except Exception:
            if sleep_seconds > 0:
                time.sleep(float(sleep_seconds))
            continue
        frame = normalize_akshare_stock_news(raw, symbol=norm)
        if not frame.empty:
            frames.append(frame)
        if sleep_seconds > 0:
            time.sleep(float(sleep_seconds))
    return merge_news_items(*frames)


def save_news_items(frame: pd.DataFrame, path: Path) -> Path:
    """保存统一 news_sentiment 表。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalize_news_items(frame).to_csv(path, index=False)
    return path
