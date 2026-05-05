"""
回测公共工具：收益、分组、面板对齐（契约见 docs/INTERFACE_AND_CONTRACTS.md）。
"""
from __future__ import annotations

import pandas as pd


def to_returns(
    prices: pd.DataFrame,
    *,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    宽表价格 -> 日简单收益率。索引为 date，列为 symbol。
    要求 `prices` 列为标的代码，值为价格（不含 MultiIndex columns）。
    """
    if isinstance(prices.columns, pd.MultiIndex):
        raise TypeError("to_returns 暂不支持 MultiIndex columns，请先转为单列宽表")
    out = prices.sort_index().pct_change()
    return out


def align_panel(
    factor: pd.Series,
    prices_long: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    symbol_col: str = "ts_code",
) -> tuple[pd.Series, pd.DataFrame]:
    """
    将因子与行情对齐到同一 (date, symbol) 网格（左表为 prices_long 中出现的键）。
    因子索引名统一为 date、symbol。
    """
    if not isinstance(factor.index, pd.MultiIndex) or factor.index.nlevels != 2:
        raise TypeError("factor 须为二级 MultiIndex (date, symbol)")
    pl = prices_long[[date_col, symbol_col]].drop_duplicates()
    pl = pl.rename(columns={date_col: "date", symbol_col: "symbol"})
    idx = pd.MultiIndex.from_frame(pl)
    fac = factor.copy()
    fac.index = fac.index.set_names(["date", "symbol"])
    factor_aligned = fac.reindex(idx)
    prices_indexed = prices_long.set_index([date_col, symbol_col]).sort_index()
    return factor_aligned, prices_indexed


def long_to_wide(
    df_long: pd.DataFrame,
    value_col: str,
    *,
    date_col: str = "trade_date",
    symbol_col: str = "ts_code",
) -> pd.DataFrame:
    """长表 -> 宽表（pivot），索引为交易日，列为 ts_code。"""
    need = {date_col, symbol_col, value_col}
    missing = need - set(df_long.columns)
    if missing:
        raise ValueError(f"long_to_wide 缺少列: {missing}")
    out = df_long.pivot(index=date_col, columns=symbol_col, values=value_col)
    out = out.sort_index().sort_index(axis=1)
    return out


def wide_to_long(
    df_wide: pd.DataFrame,
    value_name: str = "close",
) -> pd.DataFrame:
    """宽表 -> 长表；列名为 trade_date, ts_code, value_name。"""
    s = df_wide.stack()
    s.index.names = ["trade_date", "ts_code"]
    out = s.reset_index(name=value_name)
    return out


def prices_to_wide_close(
    prices: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    symbol_col: str = "ts_code",
    close_col: str = "close",
) -> pd.DataFrame:
    """
    接受宽表收盘价，或含 trade_date/ts_code/close 的长表，统一为收盘价宽表。
    """
    if {close_col, date_col, symbol_col}.issubset(prices.columns):
        return long_to_wide(prices, close_col, date_col=date_col, symbol_col=symbol_col)
    if isinstance(prices.index, pd.DatetimeIndex):
        return prices.sort_index().sort_index(axis=1)
    raise ValueError(
        "prices 须为 DatetimeIndex 的收盘价宽表，或长表且含 "
        f"{date_col!r}, {symbol_col!r}, {close_col!r}"
    )
