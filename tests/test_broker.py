"""统一券商接口与模拟券商适配器自检。"""
from __future__ import annotations

import unittest

import pandas as pd

from live.broker import (
    BROKER_MODE_REAL_READONLY,
    BrokerAccount,
    BrokerReadOnlyError,
    ORDER_STATUS_FILLED,
    ORDER_STATUS_REJECTED,
    RealBrokerConfig,
    RealBrokerReadOnlyAdapter,
    SimulatedBroker,
)
from live.order_builder import build_order_plan
from live.order_precheck import precheck_order_plan


class TestSimulatedBroker(unittest.TestCase):
    def test_submit_buy_and_sell_updates_cash_positions(self) -> None:
        broker = SimulatedBroker(
            cash=10_000.0,
            positions={"AAA": 200},
            latest_prices={"AAA": 10.0, "BBB": 20.0},
            commission_rate=0.001,
        )

        buy = broker.submit_order(
            symbol="BBB",
            side="BUY",
            qty=100,
            price=20.0,
            date="2024-01-31",
        )
        sell = broker.submit_order(
            symbol="AAA",
            side="SELL",
            qty=100,
            price=10.0,
            date="2024-01-31",
        )

        self.assertEqual(buy.status, ORDER_STATUS_FILLED)
        self.assertEqual(sell.status, ORDER_STATUS_FILLED)
        self.assertAlmostEqual(broker.get_cash(), 10_000.0 - 2_000.0 - 2.0 + 1_000.0 - 1.0)
        positions = broker.get_positions().set_index("symbol")
        self.assertEqual(int(positions.loc["AAA", "shares"]), 100)
        self.assertEqual(int(positions.loc["BBB", "shares"]), 100)
        self.assertAlmostEqual(broker.get_account().market_value, 3_000.0)

    def test_rejects_insufficient_cash_or_position(self) -> None:
        broker = SimulatedBroker(cash=1_000.0, positions={"AAA": 100}, latest_prices={"AAA": 10.0})

        buy = broker.submit_order(symbol="BBB", side="BUY", qty=100, price=20.0)
        sell = broker.submit_order(symbol="AAA", side="SELL", qty=200, price=10.0)

        self.assertEqual(buy.status, ORDER_STATUS_REJECTED)
        self.assertEqual(buy.reason, "insufficient_cash")
        self.assertEqual(sell.status, ORDER_STATUS_REJECTED)
        self.assertEqual(sell.reason, "insufficient_position")
        self.assertAlmostEqual(broker.get_cash(), 1_000.0)
        self.assertEqual(int(broker.get_positions().set_index("symbol").loc["AAA", "shares"]), 100)

    def test_submit_order_plan_respects_precheck(self) -> None:
        orders = build_order_plan(
            {"AAA": 0.0, "BBB": 1.0},
            {"AAA": 100},
            {"AAA": 10.0, "BBB": 10.0},
            cash=0.0,
            lot_size=100,
        )
        checks = precheck_order_plan(
            orders,
            cash=0.0,
            current_positions={"AAA": 100},
            lot_size=100,
        )
        broker = SimulatedBroker(
            cash=0.0,
            positions={"AAA": 100},
            latest_prices={"AAA": 10.0, "BBB": 10.0},
            commission_rate=0.0,
        )

        result = broker.submit_order_plan(orders, order_checks=checks)

        self.assertEqual(list(result["status"]), [ORDER_STATUS_FILLED, ORDER_STATUS_FILLED])
        self.assertAlmostEqual(broker.get_cash(), 0.0)
        positions = broker.get_positions().set_index("symbol")
        self.assertNotIn("AAA", positions.index)
        self.assertEqual(int(positions.loc["BBB", "shares"]), 100)

    def test_submit_order_plan_records_blocked_orders(self) -> None:
        orders = pd.DataFrame(
            {
                "date": ["2024-01-31"],
                "symbol": ["AAA"],
                "side": ["BUY"],
                "delta_shares": [100],
                "price": [10.0],
                "estimated_amount": [1_000.0],
            }
        )
        checks = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "side": ["BUY"],
                "delta_shares": [100],
                "check_status": ["BLOCK"],
            }
        )
        broker = SimulatedBroker(cash=10_000.0)

        result = broker.submit_order_plan(orders, order_checks=checks)

        self.assertEqual(str(result.loc[0, "status"]), ORDER_STATUS_REJECTED)
        self.assertEqual(str(result.loc[0, "reason"]), "blocked_by_precheck")
        self.assertAlmostEqual(broker.get_cash(), 10_000.0)


class TestRealBrokerReadOnlyAdapter(unittest.TestCase):
    def test_readonly_adapter_returns_snapshots(self) -> None:
        config = RealBrokerConfig(provider="qmt", account_id="demo", mode="REAL_READONLY")
        broker = RealBrokerReadOnlyAdapter(
            config,
            account=BrokerAccount(cash=8_000.0, market_value=2_000.0, total_asset=10_000.0),
            positions={"AAA": 100},
            latest_prices={"AAA": 20.0},
        )

        self.assertEqual(config.mode, BROKER_MODE_REAL_READONLY)
        self.assertAlmostEqual(broker.get_cash(), 8_000.0)
        self.assertAlmostEqual(broker.get_account().total_asset, 10_000.0)
        positions = broker.get_positions().set_index("symbol")
        self.assertEqual(int(positions.loc["AAA", "shares"]), 100)
        self.assertAlmostEqual(float(positions.loc["AAA", "market_value"]), 2_000.0)

    def test_readonly_adapter_blocks_trading_methods(self) -> None:
        config = RealBrokerConfig(provider="ptrade")
        broker = RealBrokerReadOnlyAdapter(config)

        with self.assertRaises(BrokerReadOnlyError):
            broker.submit_order(symbol="AAA", side="BUY", qty=100, price=10.0)
        with self.assertRaises(BrokerReadOnlyError):
            broker.cancel_order("order-1")

    def test_readonly_adapter_rejects_trading_mode(self) -> None:
        config = RealBrokerConfig(provider="qmt", mode="real_trading")

        with self.assertRaises(ValueError):
            RealBrokerReadOnlyAdapter(config)


if __name__ == "__main__":
    unittest.main()
