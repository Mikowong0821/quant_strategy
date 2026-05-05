"""
市盈率因子。输出契约：PanelLong，索引 (date, symbol)。

使用财报公告日 ann_date 做 merge_asof（backward）：每个交易日使用「已公告」的最新一期 EPS。
返回值为 -PE（PE=股价/EPS）：因子越大表示 PE 越低（越便宜），与「高因子优先」一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calc_pe(
    finance_df: pd.DataFrame,
    prices_long: pd.DataFrame,
    *,
    ts_code_col: str = "ts_code",
    date_col: str = "trade_date",
) -> pd.Series:
    """
    :param finance_df: Tushare fina_indicator 等，须含 ts_code, ann_date, eps
    :param prices_long: 长表行情，须含 trade_date, ts_code, close
    """
    need_f = {ts_code_col, "ann_date", "eps"}
    missing_f = need_f - set(finance_df.columns)
    if missing_f:
        raise ValueError(f"finance_df 缺少列: {missing_f}")

    px = prices_long[[date_col, ts_code_col, "close"]].copy()
    px[date_col] = pd.to_datetime(px[date_col])
    fin = finance_df.copy()
    fin["ann_date"] = pd.to_datetime(fin["ann_date"])
    fin = fin.sort_values([ts_code_col, "ann_date"])

    parts: list[pd.Series] = []
    for sym in px[ts_code_col].unique():
        pxs = px[px[ts_code_col] == sym].sort_values(date_col)
        fs = fin[fin[ts_code_col] == sym].dropna(subset=["ann_date", "eps"])
        fs = fs.drop_duplicates(subset=["ann_date"], keep="last")
        if fs.empty or pxs.empty:
            continue
        merged = pd.merge_asof(
            pxs,
            fs,
            left_on=date_col,
            right_on="ann_date",
            direction="backward",
        )
        eps = pd.to_numeric(merged["eps"], errors="coerce")
        close = pd.to_numeric(merged["close"], errors="coerce")
        pe = close / eps
        pe = pe.where(eps > 0)
        score = -pe
        midx = pd.MultiIndex.from_arrays(
            [merged[date_col].values, np.full(len(merged), sym, dtype=object)],
            names=["date", "symbol"],
        )
        parts.append(pd.Series(score.values, index=midx))

    if not parts:
        return pd.Series(dtype=float)
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out
