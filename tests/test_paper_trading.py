"""纸面交易：用通过预检查的订单更新虚拟账户。"""
from __future__ import annotations

import unittest

import pandas as pd

from live.order_builder import build_order_plan
from live.order_precheck import precheck_order_plan
from live.paper_trading import paper_account_snapshot, run_paper_trading


class TestPaperTrading(unittest.TestCase):
    def test_executes_passed_orders_and_applies_commission(self) -> None:
        orders = build_order_plan(
            {"AAA": 0.0, "BBB": 1.0},
            {"AAA": 1_000},
            {"AAA": 10.0, "BBB": 10.0},
            cash=0.0,
            lot_size=100,
        )
        checks = precheck_order_plan(
            orders,
            cash=0.0,
            current_positions={"AAA": 1_000},
            lot_size=100,
        )

        trades = run_paper_trading(
            initial_cash=0.0,
            orders=orders,
            order_checks=checks,
            current_positions={"AAA": 1_000},
            commission_rate=0.001,
        )

        self.assertEqual(list(trades["fill_status"]), ["FILLED", "SKIPPED"])
        self.assertEqual(str(trades.loc[0, "side"]), "SELL")
        self.assertAlmostEqual(float(trades.loc[0, "commission"]), 10.0)
        self.assertAlmostEqual(float(trades.loc[0, "cash_after"]), 9_990.0)
        self.assertEqual(str(trades.loc[1, "fill_reason"]), "insufficient_cash")

    def test_skips_orders_blocked_by_precheck(self) -> None:
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

        trades = run_paper_trading(
            initial_cash=10_000.0,
            orders=orders,
            order_checks=checks,
        )

        self.assertEqual(str(trades.loc[0, "fill_status"]), "SKIPPED")
        self.assertEqual(str(trades.loc[0, "fill_reason"]), "blocked_by_precheck")
        self.assertAlmostEqual(float(trades.loc[0, "cash_after"]), 10_000.0)

    def test_requires_precheck_unless_explicitly_allowed(self) -> None:
        orders = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "side": ["BUY"],
                "delta_shares": [100],
                "price": [10.0],
                "estimated_amount": [1_000.0],
            }
        )

        with self.assertRaises(ValueError):
            run_paper_trading(initial_cash=10_000.0, orders=orders)

        trades = run_paper_trading(
            initial_cash=10_000.0,
            orders=orders,
            allow_unchecked=True,
        )
        self.assertEqual(str(trades.loc[0, "fill_status"]), "FILLED")

    def test_account_snapshot_includes_remaining_positions(self) -> None:
        orders = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "side": ["SELL"],
                "delta_shares": [-100],
                "price": [10.0],
                "estimated_amount": [1_000.0],
            }
        )
        checks = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "side": ["SELL"],
                "delta_shares": [-100],
                "check_status": ["PASS"],
            }
        )
        trades = run_paper_trading(
            initial_cash=0.0,
            orders=orders,
            order_checks=checks,
            current_positions={"AAA": 200, "BBB": 100},
            commission_rate=0.0,
        )
        snapshot = paper_account_snapshot(
            trades,
            {"AAA": 10.0, "BBB": 20.0},
            current_positions={"AAA": 200, "BBB": 100},
        )

        self.assertAlmostEqual(snapshot["cash"], 1_000.0)
        self.assertAlmostEqual(snapshot["market_value"], 3_000.0)
        self.assertAlmostEqual(snapshot["total_asset"], 4_000.0)
        self.assertEqual(snapshot["n_positions"], 2.0)


if __name__ == "__main__":
    unittest.main()
