"""新闻 / 舆情日频因子：把统一 news_sentiment 表转成 date-symbol 面板。"""
from __future__ import annotations

from typing import Any

import pandas as pd

from live.negative_sentiment_filter import normalize_sentiment_items


NEWS_SENTIMENT_DECAY = "NEWS_SENTIMENT_DECAY"
NEWS_NEGATIVE_RISK_SCORE = "NEWS_NEGATIVE_RISK_SCORE"
NEWS_NEGATIVE_COUNT_7D = "NEWS_NEGATIVE_COUNT_7D"
NEWS_HEAT_7D = "NEWS_HEAT_7D"

NEWS_FACTOR_NAMES = (
    NEWS_SENTIMENT_DECAY,
    NEWS_NEGATIVE_RISK_SCORE,
    NEWS_NEGATIVE_COUNT_7D,
    NEWS_HEAT_7D,
)


def _price_index(
    prices_long: pd.DataFrame,
    *,
    date_col: str,
    symbol_col: str,
) -> tuple[pd.MultiIndex, dict[str, pd.Series]]:
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
    dates_by_symbol = {
        symbol: group[date_col].sort_values().reset_index(drop=True)
        for symbol, group in px.groupby(symbol_col, sort=False)
    }
    return index.sort_values(), dates_by_symbol


def _empty_panel(index: pd.MultiIndex) -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(0.0, index=index, dtype=float) for name in NEWS_FACTOR_NAMES})


def calc_news_sentiment_factors(
    items: pd.DataFrame | None,
    prices_long: pd.DataFrame,
    *,
    effective_days: int = 7,
    lookback_days: int = 7,
    date_col: str = "trade_date",
    symbol_col: str = "ts_code",
) -> pd.DataFrame:
    """
    把新闻 / 舆情表转成日频因子。

    - `NEWS_SENTIMENT_DECAY`：情绪分按未来交易日衰减累加，高值偏正、低值偏负。
    - `NEWS_NEGATIVE_RISK_SCORE`：负面风险强度，数值越高表示近期负面越强。
    - `NEWS_NEGATIVE_COUNT_7D`：近窗口内负面新闻条数。
    - `NEWS_HEAT_7D`：近窗口内新闻总条数，表示关注度 / 热度。
    """
    if effective_days <= 0 or lookback_days <= 0:
        raise ValueError("effective_days/lookback_days 必须为正整数")
    index, dates_by_symbol = _price_index(prices_long, date_col=date_col, symbol_col=symbol_col)
    out = _empty_panel(index)
    norm = normalize_sentiment_items(items)
    if norm.empty or out.empty:
        return out.sort_index()

    updates: dict[str, dict[tuple[pd.Timestamp, str], float]] = {name: {} for name in NEWS_FACTOR_NAMES}
    for rec in norm.to_dict("records"):
        symbol = str(rec.get("symbol", ""))
        if symbol not in dates_by_symbol:
            continue
        publish_time = pd.Timestamp(rec["publish_time"])
        score = pd.to_numeric(pd.Series([rec.get("sentiment_score", 0.0)]), errors="coerce").iloc[0]
        score_f = float(score) if pd.notna(score) else 0.0
        neg_hits = str(rec.get("negative_keywords", "") or "").strip()
        is_negative = score_f < 0.0 or bool(neg_hits)
        dates = dates_by_symbol[symbol]

        future_for_decay = dates[dates >= publish_time.normalize()].head(int(effective_days))
        n_decay = len(future_for_decay)
        for i, dt in enumerate(future_for_decay):
            decay = 1.0 - (i / max(n_decay, 1))
            key = (pd.Timestamp(dt), symbol)
            updates[NEWS_SENTIMENT_DECAY][key] = updates[NEWS_SENTIMENT_DECAY].get(key, 0.0) + score_f * decay
            if is_negative:
                risk = max(-score_f, 0.1) * decay
                updates[NEWS_NEGATIVE_RISK_SCORE][key] = updates[NEWS_NEGATIVE_RISK_SCORE].get(key, 0.0) + risk

        future_for_count = dates[dates >= publish_time.normalize()].head(int(lookback_days))
        for dt in future_for_count:
            key = (pd.Timestamp(dt), symbol)
            updates[NEWS_HEAT_7D][key] = updates[NEWS_HEAT_7D].get(key, 0.0) + 1.0
            if is_negative:
                updates[NEWS_NEGATIVE_COUNT_7D][key] = updates[NEWS_NEGATIVE_COUNT_7D].get(key, 0.0) + 1.0

    for name, values in updates.items():
        if not values:
            continue
        ser = pd.Series(values, dtype=float)
        ser.index = pd.MultiIndex.from_tuples(ser.index, names=["date", "symbol"])
        out[name] = out[name].add(ser.reindex(out.index).fillna(0.0), fill_value=0.0)
    return out.astype(float).sort_index()
