"""
纸面交易日终运行控制：交易日日历与重复运行保护。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings
from live.account_state import account_state_dir


class DailyPaperRunControlError(RuntimeError):
    """日终纸面交易运行控制失败。"""


def load_trading_calendar_from_prices(path: Path) -> pd.DatetimeIndex:
    """从价格宽表缓存中读取交易日日历。"""
    if not path.exists():
        raise FileNotFoundError("未找到价格缓存: %s" % path)
    frame = pd.read_csv(path, usecols=[0])
    if frame.empty:
        raise ValueError("价格缓存为空: %s" % path)
    date_col = frame.columns[0]
    dates = pd.to_datetime(frame[date_col], errors="coerce").dropna()
    if dates.empty:
        raise ValueError("价格缓存没有有效日期: %s" % path)
    return pd.DatetimeIndex(sorted(dates.dt.normalize().unique()))


def is_trading_day(trade_date: Any, calendar: pd.DatetimeIndex) -> bool:
    """判断 trade_date 是否在交易日日历中。"""
    dt = pd.Timestamp(trade_date).normalize()
    days = pd.DatetimeIndex(calendar).normalize()
    return bool(dt in days)


def previous_trading_day(trade_date: Any, calendar: pd.DatetimeIndex) -> pd.Timestamp | None:
    """返回不晚于 trade_date 的最近交易日。"""
    dt = pd.Timestamp(trade_date).normalize()
    days = pd.DatetimeIndex(calendar).normalize()
    prev = days[days <= dt]
    if len(prev) == 0:
        return None
    return pd.Timestamp(prev[-1])


def has_paper_snapshot(settings: Settings, *, strategy: str, trade_date: Any) -> bool:
    """检查指定策略和日期是否已存在纸面账户快照。"""
    path = account_state_dir(settings, strategy) / "snapshots.csv"
    if not path.exists():
        return False
    snapshots = pd.read_csv(path, usecols=["date"])
    if snapshots.empty or "date" not in snapshots.columns:
        return False
    dt = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    return bool((snapshots["date"].astype(str) == dt).any())


def validate_daily_run_control(
    settings: Settings,
    *,
    strategy: str,
    trade_date: Any,
    trading_calendar: pd.DatetimeIndex,
    persist_outputs: bool = True,
    allow_non_trading_day: bool = False,
    allow_rerun: bool = False,
) -> None:
    """
    检查日终纸面交易是否允许运行。

    - 非交易日默认阻断，防止周末/节假日误用上一交易日价格写入新快照。
    - 已存在同日快照时默认阻断，防止重复运行覆盖状态。
    - `persist_outputs=False` 时视为只读检查，不做重复运行阻断。
    """
    dt = pd.Timestamp(trade_date).normalize()
    if not is_trading_day(dt, trading_calendar):
        prev = previous_trading_day(dt, trading_calendar)
        hint = "" if prev is None else "；最近交易日为 %s" % prev.strftime("%Y-%m-%d")
        if not allow_non_trading_day:
            raise DailyPaperRunControlError(
                "运行日期 %s 不在交易日日历中%s；如需强制运行，请使用 --allow-non-trading-day"
                % (dt.strftime("%Y-%m-%d"), hint)
            )

    if persist_outputs and not allow_rerun and has_paper_snapshot(settings, strategy=strategy, trade_date=dt):
        raise DailyPaperRunControlError(
            "策略 %s 在 %s 已存在纸面账户快照；如需覆盖，请使用 --allow-rerun"
            % (strategy, dt.strftime("%Y-%m-%d"))
        )
