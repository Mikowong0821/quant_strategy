"""
基准与超额收益分析。

MVP 版本先使用「股票池等权基准」：每天持有价格宽表中的所有可用股票，
按当日简单收益做等权平均，得到一条不依赖外部指数数据的对照净值。
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from analysis.performance import summarize
from backtest.backtest_utils import prices_to_wide_close


def equal_weight_benchmark_nav(
    prices: pd.DataFrame,
    *,
    dates: pd.Index | None = None,
    price_col: str = "close",
    name: str = "BENCH_EQUAL_WEIGHT",
) -> pd.Series:
    """
    用当前股票池构造等权基准净值。

    :param prices: 收盘价宽表，或契约长表。
    :param dates: 可选目标日期索引；传入策略净值日期时，基准会对齐到同一组日期。
    :param price_col: 若 prices 为长表，使用该列作为收盘价。
    :param name: 返回序列名。
    """
    wide = prices_to_wide_close(prices, close_col=price_col)
    wide = wide.sort_index().sort_index(axis=1).astype(float)
    if wide.empty:
        return pd.Series(dtype=float, name=name)

    ret = wide.pct_change(fill_method=None)
    ret = ret.replace([np.inf, -np.inf], np.nan)
    bench_ret = ret.mean(axis=1, skipna=True).fillna(0.0)
    nav = (1.0 + bench_ret).cumprod()
    nav.name = name

    if dates is not None:
        idx = pd.DatetimeIndex(dates, name="date")
        nav = nav.reindex(idx).ffill().dropna()
        if not nav.empty:
            nav = nav / float(nav.iloc[0])
            nav.name = name
    return nav


def align_nav_pair(strategy_nav: pd.Series, benchmark_nav: pd.Series) -> pd.DataFrame:
    """对齐策略净值和基准净值，并各自归一到首个共同日期为 1。"""
    df = pd.concat(
        [
            strategy_nav.astype(float).rename("strategy"),
            benchmark_nav.astype(float).rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if df.empty:
        return df
    first = df.iloc[0].replace(0.0, np.nan)
    df = df.div(first, axis=1).dropna()
    return df


def excess_return_series(strategy_nav: pd.Series, benchmark_nav: pd.Series) -> pd.Series:
    """策略日收益减基准日收益，得到主动收益序列。"""
    aligned = align_nav_pair(strategy_nav, benchmark_nav)
    if aligned.empty:
        return pd.Series(dtype=float, name="excess_return")
    ret = aligned.pct_change(fill_method=None).fillna(0.0)
    out = ret["strategy"] - ret["benchmark"]
    out.name = "excess_return"
    return out


def excess_nav(strategy_nav: pd.Series, benchmark_nav: pd.Series) -> pd.Series:
    """由主动收益累乘得到超额净值，起点为 1。"""
    active_ret = excess_return_series(strategy_nav, benchmark_nav)
    if active_ret.empty:
        return pd.Series(dtype=float, name="excess_nav")
    out = (1.0 + active_ret).cumprod()
    out.name = "excess_nav"
    return out


def summarize_excess(
    strategy_nav: pd.Series,
    benchmark_nav: pd.Series,
    *,
    periods: int = 252,
) -> dict[str, Any]:
    """
    汇总相对基准的指标。

    - benchmark_*：基准自身的年化收益、波动等。
    - excess_ann_return：超额净值的年化收益。
    - tracking_error：主动收益的年化波动。
    - information_ratio：主动收益年化均值 / tracking_error。
    """
    aligned = align_nav_pair(strategy_nav, benchmark_nav)
    if aligned.empty:
        return {
            "benchmark_ann_return": np.nan,
            "benchmark_ann_vol": np.nan,
            "excess_ann_return": np.nan,
            "tracking_error": np.nan,
            "information_ratio": np.nan,
        }

    bench_stats = summarize(aligned["benchmark"], periods=periods)
    active_ret = excess_return_series(aligned["strategy"], aligned["benchmark"])
    xnav = (1.0 + active_ret).cumprod()
    xstats = summarize(xnav, periods=periods)

    te = float(active_ret.std() * np.sqrt(periods))
    active_ann_mean = float(active_ret.mean() * periods)
    ir = float(active_ann_mean / te) if te > 1e-12 else float("nan")
    return {
        "benchmark_ann_return": bench_stats["ann_return"],
        "benchmark_ann_vol": bench_stats["ann_vol"],
        "excess_ann_return": xstats["ann_return"],
        "tracking_error": te,
        "information_ratio": ir,
    }


def excess_nav_frame(
    nav_by_name: Mapping[str, pd.Series],
    benchmark_nav: pd.Series,
) -> pd.DataFrame:
    """为多条策略生成超额净值宽表，列名与策略名一致。"""
    rows: dict[str, pd.Series] = {}
    for name, nav in nav_by_name.items():
        if name == benchmark_nav.name:
            continue
        xnav = excess_nav(nav, benchmark_nav)
        if not xnav.empty:
            rows[str(name)] = xnav
    return pd.DataFrame(rows)
