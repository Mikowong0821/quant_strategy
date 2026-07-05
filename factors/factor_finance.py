"""
财务衍生因子。输出契约：PanelLong，索引 (date, symbol)。

所有财务字段都按公告日 ann_date 做 backward merge_asof：每个交易日只使用
当时已经公告的最新财务指标，避免用到未来信息。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def calc_finance_metric(
    finance_df: pd.DataFrame,
    prices_long: pd.DataFrame,
    *,
    metric_candidates: tuple[str, ...],
    higher_is_better: bool = True,
    ts_code_col: str = "ts_code",
    date_col: str = "trade_date",
) -> pd.Series:
    """
    将单个财务指标对齐到日频交易网格。

    :param finance_df: 财务指标表，至少含 ts_code、ann_date 和目标指标列之一
    :param prices_long: 长表行情，至少含 trade_date、ts_code、close
    :param metric_candidates: 目标指标候选列名，按优先级选择第一个存在列
    :param higher_is_better: False 时取反，适合负债率等越低越好的指标
    """
    need_px = {date_col, ts_code_col, "close"}
    missing_px = need_px - set(prices_long.columns)
    if missing_px:
        raise ValueError(f"prices_long 缺少列: {missing_px}")
    need_f = {ts_code_col, "ann_date"}
    missing_f = need_f - set(finance_df.columns)
    if missing_f:
        raise ValueError(f"finance_df 缺少列: {missing_f}")

    metric_col = _pick_column(finance_df, metric_candidates)
    if metric_col is None:
        return pd.Series(dtype=float)

    px = prices_long[[date_col, ts_code_col, "close"]].copy()
    px[date_col] = pd.to_datetime(px[date_col])
    fin = finance_df[[ts_code_col, "ann_date", metric_col]].copy()
    fin["ann_date"] = pd.to_datetime(fin["ann_date"], errors="coerce")
    fin[metric_col] = pd.to_numeric(fin[metric_col], errors="coerce")
    fin = fin.dropna(subset=["ann_date", metric_col]).sort_values([ts_code_col, "ann_date"])

    parts: list[pd.Series] = []
    for sym in px[ts_code_col].unique():
        pxs = px[px[ts_code_col] == sym].sort_values(date_col)
        fs = fin[fin[ts_code_col] == sym].drop_duplicates(subset=["ann_date"], keep="last")
        if fs.empty or pxs.empty:
            continue
        merged = pd.merge_asof(
            pxs,
            fs,
            left_on=date_col,
            right_on="ann_date",
            direction="backward",
        )
        score = pd.to_numeric(merged[metric_col], errors="coerce")
        if not higher_is_better:
            score = -score
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


def calc_gross_margin(finance_df: pd.DataFrame, prices_long: pd.DataFrame) -> pd.Series:
    """毛利率因子，越高越好。"""
    return calc_finance_metric(
        finance_df,
        prices_long,
        metric_candidates=("grossprofit_margin", "gross_margin"),
    )


def calc_net_margin(finance_df: pd.DataFrame, prices_long: pd.DataFrame) -> pd.Series:
    """净利率因子，越高越好。"""
    return calc_finance_metric(
        finance_df,
        prices_long,
        metric_candidates=("netprofit_margin", "net_margin"),
    )


def calc_low_debt_to_assets(finance_df: pd.DataFrame, prices_long: pd.DataFrame) -> pd.Series:
    """低资产负债率因子，取负号后越高表示负债率越低。"""
    return calc_finance_metric(
        finance_df,
        prices_long,
        metric_candidates=("debt_to_assets", "debt_assets"),
        higher_is_better=False,
    )


def calc_revenue_growth(finance_df: pd.DataFrame, prices_long: pd.DataFrame) -> pd.Series:
    """营收同比增长因子，越高越好。"""
    return calc_finance_metric(
        finance_df,
        prices_long,
        metric_candidates=("or_yoy", "tr_yoy", "revenue_yoy", "q_sales_yoy"),
    )


def calc_profit_growth(finance_df: pd.DataFrame, prices_long: pd.DataFrame) -> pd.Series:
    """利润同比增长因子，越高越好。"""
    return calc_finance_metric(
        finance_df,
        prices_long,
        metric_candidates=("netprofit_yoy", "profit_yoy", "q_profit_yoy", "dt_netprofit_yoy"),
    )
