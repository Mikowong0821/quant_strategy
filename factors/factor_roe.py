"""
ROE 因子。输出契约：PanelLong，索引 (date, symbol)。

与 PE 相同：按 ann_date merge_asof 到每个交易日，使用已公告最新一期 roe（净资产收益率，Tushare 多为百分数）。
因子值越大表示 ROE 越高。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calc_roe(
    finance_df: pd.DataFrame,
    prices_long: pd.DataFrame,
    *,
    ts_code_col: str = "ts_code",
    date_col: str = "trade_date",
) -> pd.Series:
    """
    :param finance_df: 须含 ts_code, ann_date, roe
    :param prices_long: 须含 trade_date, ts_code, close（close 仅用于对齐交易日网格）
    """
    need_f = {ts_code_col, "ann_date", "roe"}
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
        fs = fin[fin[ts_code_col] == sym].dropna(subset=["ann_date", "roe"])
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
        score = pd.to_numeric(merged["roe"], errors="coerce")
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
