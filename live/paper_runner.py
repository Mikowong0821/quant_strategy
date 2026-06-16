"""
每日纸面交易运行器：串联账户状态、订单计划、预检查、纸面成交与状态保存。
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings
from live.account_state import (
    load_account_state,
    positions_from_trades,
    save_account_state,
)
from live.cache_io import save_order_checks, save_order_plans, save_paper_trades
from live.order_builder import build_order_plan
from live.order_precheck import precheck_order_plan
from live.paper_trading import paper_account_snapshot, run_paper_trading


def run_daily_paper_trade(
    settings: Settings,
    *,
    strategy: str,
    target_weights: Mapping[str, float] | pd.Series | pd.DataFrame,
    latest_prices: Mapping[str, float] | pd.Series | pd.DataFrame,
    trade_date: Any,
    trade_status: pd.DataFrame | Mapping[str, Mapping[str, Any]] | None = None,
    default_cash: float | None = None,
    total_asset: float | None = None,
    persist_outputs: bool = True,
) -> dict[str, Any]:
    """
    运行单日纸面交易流程。

    流程：
    1. 读取纸面账户状态
    2. 生成订单计划
    3. 做订单预检查
    4. 执行纸面交易
    5. 生成账户快照并保存状态

    :return: dict，含 `orders/order_checks/paper_trades/account_snapshot/positions/paths`
    """
    start_cash, current_positions = load_account_state(
        settings,
        strategy=strategy,
        default_cash=(
            float(settings.paper_initial_cash)
            if default_cash is None
            else float(default_cash)
        ),
    )
    orders = build_order_plan(
        target_weights,
        current_positions,
        latest_prices,
        cash=start_cash,
        total_asset=total_asset,
        trade_date=trade_date,
        lot_size=int(settings.order_lot_size),
        min_order_amount=float(settings.min_order_amount),
        include_holds=False,
    )
    checks = precheck_order_plan(
        orders,
        cash=start_cash,
        current_positions=current_positions,
        trade_status=trade_status,
        lot_size=int(settings.order_lot_size),
        min_order_amount=float(settings.min_order_amount),
        cash_buffer=float(settings.order_cash_buffer),
    )
    trades = run_paper_trading(
        initial_cash=start_cash,
        orders=orders,
        order_checks=checks,
        current_positions=current_positions,
        commission_rate=float(settings.commission_rate),
    )
    latest_cash = float(trades["cash_after"].iloc[-1]) if not trades.empty else float(start_cash)
    latest_positions = positions_from_trades(
        trades,
        current_positions,
        updated_at=trade_date,
    )
    snapshot = paper_account_snapshot(
        trades,
        latest_prices,
        current_positions=current_positions,
        cash=start_cash,
    )
    if trades.empty:
        snapshot["cash"] = latest_cash
        snapshot["total_asset"] = latest_cash + float(snapshot.get("market_value", 0.0))

    paths: dict[str, Path | dict[str, Path]] = {}
    if persist_outputs:
        paths["order_plans"] = save_order_plans(settings, {strategy: orders}).get(strategy)
        paths["order_checks"] = save_order_checks(settings, {strategy: checks}).get(strategy)
        paths["paper_trades"] = save_paper_trades(settings, {strategy: trades}).get(strategy)
        paths["account_state"] = save_account_state(
            settings,
            strategy=strategy,
            cash=latest_cash,
            positions=latest_positions,
            snapshot=snapshot,
            trade_date=trade_date,
        )

    return {
        "strategy": strategy,
        "trade_date": trade_date,
        "starting_cash": start_cash,
        "starting_positions": current_positions,
        "orders": orders,
        "order_checks": checks,
        "paper_trades": trades,
        "cash": latest_cash,
        "positions": latest_positions,
        "account_snapshot": snapshot,
        "paths": paths,
    }
