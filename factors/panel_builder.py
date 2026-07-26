"""
一次性计算多列因子面板，供单因子回测（避免重复算子）与横截面 z-score 融合使用。
与 `backtest_single` 中各因子分支保持同一口径。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.backtest_utils import long_to_wide, prices_to_wide_close, to_returns, wide_to_long
from config import Settings
from factors.factor_events import (
    ANNOUNCEMENT_EVENT_SCORE,
    calc_announcement_event_score,
    load_announcement_events,
)
from factors.factor_finance import (
    calc_cash_profit_quality,
    calc_free_cash_flow_yield,
    calc_gross_margin,
    calc_low_debt_to_assets,
    calc_net_margin,
    calc_profit_growth,
    calc_revenue_growth,
)
from factors.factor_momentum import calc_momentum
from factors.factor_pe import calc_pe
from factors.factor_reversal import calc_reversal
from factors.factor_roe import calc_roe
from factors.factor_volume import calc_volume_ratio
from factors.factor_volatility import calc_volatility
from live.data_feed import fetch_fina_indicator_panel, load_fina_indicator_from_csv

DEFAULT_FACTOR_ORDER = [
    "MOMENTUM",
    "MOMENTUM_60D",
    "REVERSAL_5D",
    "VOLATILITY",
    "VOLUME_RATIO_20D",
    "PE",
    "ROE",
    "GROSS_MARGIN",
    "NET_MARGIN",
    "LOW_DEBT_TO_ASSETS",
    "REVENUE_GROWTH",
    "PROFIT_GROWTH",
    "FREE_CASH_FLOW_YIELD",
    "CASH_PROFIT_QUALITY",
    ANNOUNCEMENT_EVENT_SCORE,
]


def _empty_like(index: pd.MultiIndex) -> pd.Series:
    return pd.Series(index=index, dtype=float)


def _volume_wide_from_long(long_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if long_df is None or long_df.empty or "volume" not in long_df.columns:
        return None
    try:
        return long_to_wide(long_df, "volume")
    except Exception:
        return None


def _safe_finance_factor(func, fina: pd.DataFrame, long_px: pd.DataFrame, index: pd.MultiIndex) -> pd.Series:
    try:
        out = func(fina, long_px)
    except Exception:
        return pd.Series(index=index, dtype=float)
    if out.empty:
        return pd.Series(index=index, dtype=float)
    return out.reindex(index)


def _align_factor(series: pd.Series, index: pd.MultiIndex) -> pd.Series:
    if series.empty:
        return pd.Series(index=index, dtype=float)
    out = series.copy()
    if isinstance(out.index, pd.MultiIndex):
        out.index = out.index.set_names(["date", "symbol"])
    return out.reindex(index)


def build_four_factor_panel(
    prices: pd.DataFrame,
    long_df: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """
    :param prices: 收盘价宽表或契约长表
    :param long_df: 含 trade_date, ts_code, close/volume 的长表（PE/ROE、成交量因子用）
    :return: DataFrame，索引 MultiIndex(date, symbol)，列为多因子原始值
    """
    prices_wide = prices_to_wide_close(
        prices,
        date_col="trade_date",
        symbol_col="ts_code",
        close_col=settings.price_col,
    )
    prices_wide = prices_wide.sort_index().sort_index(axis=1)

    mom = calc_momentum(prices_wide, lookback=settings.momentum_lookback)
    mom_long = calc_momentum(
        prices_wide,
        lookback=getattr(settings, "momentum_long_lookback", 60),
    )
    reversal = calc_reversal(
        prices_wide,
        lookback=getattr(settings, "reversal_lookback", 5),
    )
    vol = calc_volatility(
        to_returns(prices_wide),
        window=settings.vol_window,
        annualize=True,
        trading_days=settings.trading_days_per_year,
    )

    long_px = long_df
    if long_px is None or long_px.empty:
        long_px = wide_to_long(prices_wide, settings.price_col)

    volume_wide = _volume_wide_from_long(long_df)
    if volume_wide is None:
        volume_ratio = _empty_like(mom.index)
    else:
        volume_ratio = calc_volume_ratio(
            volume_wide,
            window=getattr(settings, "volume_ratio_window", 20),
        )
    idx = mom.index
    mom = _align_factor(mom, idx)
    mom_long = _align_factor(mom_long, idx)
    reversal = _align_factor(reversal, idx)
    vol = _align_factor(vol, idx)
    volume_ratio = _align_factor(volume_ratio, idx)

    try:
        fina_cache = getattr(settings, "fina_indicator_cache_path", None)
        if fina_cache is not None and Path(fina_cache).expanduser().is_file():
            fina = load_fina_indicator_from_csv(fina_cache)
        else:
            fina = fetch_fina_indicator_panel(
                list(prices_wide.columns),
                settings.backtest_start,
                settings.backtest_end,
                history_years=settings.fina_history_years,
            )
    except Exception as exc:
        print("财务指标加载失败，财务因子将为空:", exc)
        fina = pd.DataFrame()
    if fina.empty:
        pe = pd.Series(index=idx, dtype=float)
        roe = pd.Series(index=idx, dtype=float)
        gross_margin = pd.Series(index=idx, dtype=float)
        net_margin = pd.Series(index=idx, dtype=float)
        low_debt = pd.Series(index=idx, dtype=float)
        revenue_growth = pd.Series(index=idx, dtype=float)
        profit_growth = pd.Series(index=idx, dtype=float)
        free_cash_flow_yield = pd.Series(index=idx, dtype=float)
        cash_profit_quality = pd.Series(index=idx, dtype=float)
    else:
        pe = _safe_finance_factor(calc_pe, fina, long_px, idx)
        roe = _safe_finance_factor(calc_roe, fina, long_px, idx)
        gross_margin = _safe_finance_factor(calc_gross_margin, fina, long_px, idx)
        net_margin = _safe_finance_factor(calc_net_margin, fina, long_px, idx)
        low_debt = _safe_finance_factor(calc_low_debt_to_assets, fina, long_px, idx)
        revenue_growth = _safe_finance_factor(calc_revenue_growth, fina, long_px, idx)
        profit_growth = _safe_finance_factor(calc_profit_growth, fina, long_px, idx)
        free_cash_flow_yield = _safe_finance_factor(calc_free_cash_flow_yield, fina, long_px, idx)
        cash_profit_quality = _safe_finance_factor(calc_cash_profit_quality, fina, long_px, idx)

    try:
        event_path = getattr(settings, "announcement_event_path", None)
        events = load_announcement_events(event_path)
        event_score = calc_announcement_event_score(
            events,
            long_px,
            effective_days=int(getattr(settings, "announcement_event_effective_days", 20)),
        ).reindex(idx)
    except Exception as exc:
        print("公告事件因子加载失败，将为空:", exc)
        event_score = pd.Series(index=idx, dtype=float)

    panel = pd.concat(
        {
            "MOMENTUM": mom,
            "MOMENTUM_60D": mom_long,
            "REVERSAL_5D": reversal,
            "VOLATILITY": vol,
            "VOLUME_RATIO_20D": volume_ratio,
            "PE": pe,
            "ROE": roe,
            "GROSS_MARGIN": gross_margin,
            "NET_MARGIN": net_margin,
            "LOW_DEBT_TO_ASSETS": low_debt,
            "REVENUE_GROWTH": revenue_growth,
            "PROFIT_GROWTH": profit_growth,
            "FREE_CASH_FLOW_YIELD": free_cash_flow_yield,
            "CASH_PROFIT_QUALITY": cash_profit_quality,
            ANNOUNCEMENT_EVENT_SCORE: event_score,
        },
        axis=1,
    )
    return panel.sort_index()
