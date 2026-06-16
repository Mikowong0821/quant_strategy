"""
纸面交易：用虚拟账户执行通过预检查的订单，更新现金和持仓。

本模块只做模拟成交，不连接券商，也不处理真实成交回报。
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


PAPER_TRADE_COLUMNS = [
    "date",
    "symbol",
    "side",
    "qty",
    "price",
    "gross_amount",
    "commission",
    "net_cash_flow",
    "cash_before",
    "cash_after",
    "position_before",
    "position_after",
    "fill_status",
    "fill_reason",
]


def _positions_to_series(current_positions: pd.DataFrame | Mapping[str, float] | pd.Series | None) -> pd.Series:
    if current_positions is None:
        return pd.Series(dtype=float)
    if isinstance(current_positions, pd.DataFrame):
        symbol_col = "symbol" if "symbol" in current_positions.columns else "ts_code"
        if symbol_col not in current_positions.columns or "shares" not in current_positions.columns:
            raise ValueError("current_positions DataFrame 须包含 symbol/ts_code 与 shares 列")
        out = pd.Series(
            current_positions["shares"].astype(float).to_numpy(),
            index=current_positions[symbol_col].astype(str),
            dtype=float,
        )
    elif isinstance(current_positions, pd.Series):
        out = current_positions.astype(float).copy()
        out.index = out.index.astype(str)
    else:
        out = pd.Series({str(k): float(v) for k, v in current_positions.items()}, dtype=float)
    return out.groupby(level=0).sum().sort_index()


def _passed_order_keys(order_checks: pd.DataFrame | None) -> set[tuple[str, str, int]]:
    if order_checks is None or order_checks.empty:
        return set()
    required = {"symbol", "side", "delta_shares", "check_status"}
    missing = required - set(order_checks.columns)
    if missing:
        raise ValueError("order_checks 缺少必要列: %s" % ", ".join(sorted(missing)))
    keys: set[tuple[str, str, int]] = set()
    for rec in order_checks.to_dict("records"):
        if str(rec.get("check_status", "")).upper() == "PASS":
            keys.add(
                (
                    str(rec.get("symbol", "")),
                    str(rec.get("side", "")).upper(),
                    int(round(float(rec.get("delta_shares", 0)))),
                )
            )
    return keys


def _order_rank(side: str) -> int:
    if side == "SELL":
        return 0
    if side == "BUY":
        return 1
    return 2


def run_paper_trading(
    symbols: Sequence[str] | None = None,
    *,
    initial_cash: float = 1_000_000.0,
    orders: pd.DataFrame | None = None,
    order_checks: pd.DataFrame | None = None,
    current_positions: pd.DataFrame | Mapping[str, float] | pd.Series | None = None,
    commission_rate: float = 0.0003,
    allow_unchecked: bool = False,
    signals: pd.Series | None = None,
    prices: pd.DataFrame | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    执行纸面交易并返回成交日志。

    :param symbols: 保留兼容旧接口；当前订单驱动模式不依赖该参数
    :param initial_cash: 纸面账户初始 / 当前可用现金
    :param orders: `live.order_builder.build_order_plan` 输出
    :param order_checks: `live.order_precheck.precheck_order_plan` 输出；默认只执行 PASS 订单
    :param current_positions: 当前持仓，含 `symbol,shares` 或映射
    :param commission_rate: 单边手续费率
    :param allow_unchecked: True 时允许没有 `order_checks` 也执行订单；默认 False
    :return: 成交与跳过日志，列见 `PAPER_TRADE_COLUMNS`
    """
    del symbols, signals, prices, kwargs
    if initial_cash < 0:
        raise ValueError("initial_cash 不能为负")
    if commission_rate < 0:
        raise ValueError("commission_rate 不能为负")
    if orders is None:
        raise NotImplementedError("纸面交易当前只支持订单计划驱动；请传入 orders")
    if orders.empty:
        return pd.DataFrame(columns=PAPER_TRADE_COLUMNS)

    required = {"symbol", "side", "delta_shares", "price", "estimated_amount"}
    missing = required - set(orders.columns)
    if missing:
        raise ValueError("orders 缺少必要列: %s" % ", ".join(sorted(missing)))
    if order_checks is None and not allow_unchecked:
        raise ValueError("缺少 order_checks；若确认跳过预检查，请设置 allow_unchecked=True")

    passed_keys = _passed_order_keys(order_checks)
    positions = _positions_to_series(current_positions)
    cash_now = float(initial_cash)

    frame = orders.copy()
    frame["side"] = frame["side"].astype(str).str.upper()
    frame["_order_rank"] = frame["side"].map(_order_rank)
    frame = frame.sort_values(["_order_rank", "symbol"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for rec in frame.to_dict("records"):
        symbol = str(rec.get("symbol", ""))
        side = str(rec.get("side", "")).upper()
        delta = int(round(float(rec.get("delta_shares", 0))))
        price = float(rec.get("price", float("nan")))
        qty = abs(delta)
        gross = float(rec.get("estimated_amount", qty * price))
        date = rec.get("date", "")
        cash_before = cash_now
        pos_before = int(round(float(positions.get(symbol, 0.0))))

        key = (symbol, side, delta)
        reason = "filled"
        status = "FILLED"
        commission = 0.0
        net_cash_flow = 0.0
        pos_after = pos_before

        if side not in {"BUY", "SELL"}:
            status = "SKIPPED"
            reason = "not_trade_order"
        elif not math.isfinite(price) or price <= 0.0 or qty <= 0 or gross <= 0.0:
            status = "SKIPPED"
            reason = "invalid_order"
        elif order_checks is not None and key not in passed_keys:
            status = "SKIPPED"
            reason = "blocked_by_precheck"
        elif side == "SELL" and qty > pos_before:
            status = "SKIPPED"
            reason = "insufficient_position"
        else:
            commission = gross * float(commission_rate)
            if side == "BUY":
                required_cash = gross + commission
                if required_cash > cash_now + 1e-8:
                    status = "SKIPPED"
                    reason = "insufficient_cash"
                    commission = 0.0
                else:
                    net_cash_flow = -required_cash
                    cash_now += net_cash_flow
                    pos_after = pos_before + qty
                    positions.loc[symbol] = float(pos_after)
            elif side == "SELL":
                net_cash_flow = gross - commission
                cash_now += net_cash_flow
                pos_after = pos_before - qty
                if pos_after == 0 and symbol in positions.index:
                    positions = positions.drop(symbol)
                else:
                    positions.loc[symbol] = float(pos_after)

        rows.append(
            {
                "date": date,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "gross_amount": gross,
                "commission": commission,
                "net_cash_flow": net_cash_flow,
                "cash_before": cash_before,
                "cash_after": cash_now,
                "position_before": pos_before,
                "position_after": pos_after,
                "fill_status": status,
                "fill_reason": reason,
            }
        )

    return pd.DataFrame(rows, columns=PAPER_TRADE_COLUMNS)


def paper_account_snapshot(
    trades: pd.DataFrame,
    latest_prices: Mapping[str, float] | pd.Series | pd.DataFrame,
    current_positions: pd.DataFrame | Mapping[str, float] | pd.Series | None = None,
    cash: float | None = None,
) -> dict[str, float]:
    """
    根据纸面成交日志和最新价格估算账户快照。

    :return: `cash`、`market_value`、`total_asset`、`n_positions`
    """
    if isinstance(latest_prices, pd.DataFrame):
        symbol_col = "symbol" if "symbol" in latest_prices.columns else "ts_code"
        price_col = "price" if "price" in latest_prices.columns else "close"
        prices = pd.Series(
            latest_prices[price_col].astype(float).to_numpy(),
            index=latest_prices[symbol_col].astype(str),
            dtype=float,
        )
    elif isinstance(latest_prices, pd.Series):
        prices = latest_prices.astype(float).copy()
        prices.index = prices.index.astype(str)
    else:
        prices = pd.Series({str(k): float(v) for k, v in latest_prices.items()}, dtype=float)

    if trades.empty:
        positions_s = _positions_to_series(current_positions)
        market_value = 0.0
        n_positions = 0
        for symbol, shares_f in positions_s.items():
            shares = int(round(float(shares_f)))
            if shares <= 0 or symbol not in prices.index:
                continue
            market_value += shares * float(prices.loc[symbol])
            n_positions += 1
        return {
            "cash": float(cash or 0.0),
            "market_value": market_value,
            "total_asset": float(cash or 0.0) + market_value,
            "n_positions": float(n_positions),
        }

    latest_cash = float(trades["cash_after"].iloc[-1])
    filled = trades[trades["fill_status"] == "FILLED"]
    positions = {str(k): int(round(float(v))) for k, v in _positions_to_series(current_positions).items()}
    for rec in filled.to_dict("records"):
        symbol = str(rec["symbol"])
        positions[symbol] = int(rec["position_after"])

    market_value = 0.0
    n_positions = 0
    for symbol, shares in positions.items():
        if shares <= 0:
            continue
        if symbol not in prices.index:
            continue
        market_value += shares * float(prices.loc[symbol])
        n_positions += 1
    return {
        "cash": latest_cash,
        "market_value": market_value,
        "total_asset": latest_cash + market_value,
        "n_positions": float(n_positions),
    }
