"""
一次性计算多列因子面板，供单因子回测（避免重复算子）与横截面 z-score 融合使用。
与 `backtest_single` 中各因子分支保持同一口径。
"""
from __future__ import annotations

import pandas as pd

from backtest.backtest_utils import prices_to_wide_close, to_returns, wide_to_long
from config import Settings
from factors.factor_momentum import calc_momentum
from factors.factor_pe import calc_pe
from factors.factor_roe import calc_roe
from factors.factor_volatility import calc_volatility
from live.data_feed import fetch_fina_indicator_panel

DEFAULT_FACTOR_ORDER = ["MOMENTUM", "VOLATILITY", "PE", "ROE"]


def build_four_factor_panel(
    prices: pd.DataFrame,
    long_df: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """
    :param prices: 收盘价宽表或契约长表
    :param long_df: 含 trade_date, ts_code, close 的长表（PE/ROE 用）
    :return: DataFrame，索引 MultiIndex(date, symbol)，列为四因子原始值
    """
    prices_wide = prices_to_wide_close(
        prices,
        date_col="trade_date",
        symbol_col="ts_code",
        close_col=settings.price_col,
    )
    prices_wide = prices_wide.sort_index().sort_index(axis=1)

    mom = calc_momentum(prices_wide, lookback=settings.momentum_lookback)
    vol = calc_volatility(
        to_returns(prices_wide),
        window=settings.vol_window,
        annualize=True,
        trading_days=settings.trading_days_per_year,
    )

    long_px = long_df
    if long_px is None or long_px.empty:
        long_px = wide_to_long(prices_wide, settings.price_col)

    fina = fetch_fina_indicator_panel(
        list(prices_wide.columns),
        settings.backtest_start,
        settings.backtest_end,
        history_years=settings.fina_history_years,
    )
    if fina.empty:
        idx = mom.index
        pe = pd.Series(index=idx, dtype=float)
        roe = pd.Series(index=idx, dtype=float)
    else:
        pe = calc_pe(fina, long_px)
        roe = calc_roe(fina, long_px)

    panel = pd.concat(
        {"MOMENTUM": mom, "VOLATILITY": vol, "PE": pe, "ROE": roe},
        axis=1,
    )
    return panel.sort_index()
