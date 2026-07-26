"""订单预检查：现金、持仓与交易状态约束。"""
from __future__ import annotations

import unittest

import pandas as pd

from live.order_builder import build_order_plan
from live.order_precheck import precheck_order_plan


class TestOrderPrecheck(unittest.TestCase):
    def test_sell_first_releases_cash_for_buy(self) -> None:
        orders = build_order_plan(
            {"AAA": 0.0, "BBB": 1.0},
            {"AAA": 1_000},
            {"AAA": 10.0, "BBB": 10.0},
            cash=0.0,
            lot_size=100,
        )

        checked = precheck_order_plan(
            orders,
            cash=0.0,
            current_positions={"AAA": 1_000},
            lot_size=100,
        )

        self.assertEqual(list(checked["side"]), ["SELL", "BUY"])
        self.assertEqual(list(checked["check_status"]), ["PASS", "PASS"])
        self.assertAlmostEqual(float(checked.iloc[-1]["cash_after"]), 0.0)

    def test_buy_blocks_when_cash_is_insufficient(self) -> None:
        orders = pd.DataFrame(
            {
                "date": ["2024-01-31"],
                "symbol": ["AAA"],
                "side": ["BUY"],
                "delta_shares": [200],
                "price": [10.0],
                "estimated_amount": [2_000.0],
            }
        )

        checked = precheck_order_plan(orders, cash=1_000.0, lot_size=100)

        self.assertEqual(str(checked.loc[0, "check_status"]), "BLOCK")
        self.assertIn("insufficient_cash", str(checked.loc[0, "check_reason"]))

    def test_trade_status_blocks_suspended_and_limit_orders(self) -> None:
        orders = pd.DataFrame(
            {
                "date": ["2024-01-31", "2024-01-31", "2024-01-31"],
                "symbol": ["AAA", "BBB", "CCC"],
                "side": ["BUY", "SELL", "BUY"],
                "delta_shares": [100, -100, 100],
                "price": [10.0, 10.0, 10.0],
                "estimated_amount": [1_000.0, 1_000.0, 1_000.0],
            }
        )
        trade_status = pd.DataFrame(
            {
                "symbol": ["AAA", "BBB", "CCC"],
                "is_limit_up": [True, False, False],
                "is_limit_down": [False, True, False],
                "is_suspended": [False, False, True],
            }
        )

        checked = precheck_order_plan(
            orders,
            cash=10_000.0,
            current_positions={"BBB": 100},
            trade_status=trade_status,
            lot_size=100,
        )
        reasons = dict(zip(checked["symbol"], checked["check_reason"], strict=True))

        self.assertIn("limit_up_blocks_buy", str(reasons["AAA"]))
        self.assertIn("limit_down_blocks_sell", str(reasons["BBB"]))
        self.assertIn("suspended", str(reasons["CCC"]))
        self.assertTrue((checked["check_status"] == "BLOCK").all())

    def test_sell_blocks_when_available_shares_are_insufficient(self) -> None:
        orders = pd.DataFrame(
            {
                "date": ["2024-01-31"],
                "symbol": ["AAA"],
                "side": ["SELL"],
                "delta_shares": [-200],
                "price": [10.0],
                "estimated_amount": [2_000.0],
            }
        )
        positions = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "shares": [200],
                "available_shares": [100],
            }
        )

        checked = precheck_order_plan(orders, cash=0.0, current_positions=positions)

        self.assertEqual(str(checked.loc[0, "check_status"]), "BLOCK")
        self.assertIn("insufficient_available_shares", str(checked.loc[0, "check_reason"]))

    def test_buy_must_match_lot_size(self) -> None:
        orders = pd.DataFrame(
            {
                "date": ["2024-01-31"],
                "symbol": ["AAA"],
                "side": ["BUY"],
                "delta_shares": [150],
                "price": [10.0],
                "estimated_amount": [1_500.0],
            }
        )

        checked = precheck_order_plan(orders, cash=10_000.0, lot_size=100)

        self.assertEqual(str(checked.loc[0, "check_status"]), "BLOCK")
        self.assertIn("not_lot_size", str(checked.loc[0, "check_reason"]))

    def test_risk_blacklist_blocks_order(self) -> None:
        orders = pd.DataFrame(
            {
                "date": ["2024-01-31", "2024-01-31"],
                "symbol": ["AAA", "BBB"],
                "side": ["BUY", "BUY"],
                "delta_shares": [100, 100],
                "price": [10.0, 10.0],
                "estimated_amount": [1_000.0, 1_000.0],
            }
        )
        risk_blacklist = pd.DataFrame(
            [
                {
                    "symbol": "AAA",
                    "severity": "HIGH",
                    "reason": "earnings_uncertainty",
                },
            ]
        )

        checked = precheck_order_plan(
            orders,
            cash=10_000.0,
            risk_blacklist=risk_blacklist,
            lot_size=100,
        )
        by_symbol = checked.set_index("symbol")

        self.assertEqual(str(by_symbol.loc["AAA", "check_status"]), "BLOCK")
        self.assertIn("risk_blacklist", str(by_symbol.loc["AAA", "check_reason"]))
        self.assertTrue(bool(by_symbol.loc["AAA", "is_blacklisted"]))
        self.assertEqual(str(by_symbol.loc["AAA", "blacklist_reason"]), "earnings_uncertainty")
        self.assertEqual(str(by_symbol.loc["BBB", "check_status"]), "PASS")


if __name__ == "__main__":
    unittest.main()
