"""订单生成：目标权重转换为买卖股数。"""
from __future__ import annotations

import unittest

import pandas as pd

from live.order_builder import (
    build_order_plan,
    build_order_plan_from_rebalance_meta,
    latest_target_weights_from_rebalance_meta,
)


class TestOrderBuilder(unittest.TestCase):
    def test_build_order_plan_sells_first_and_rounds_to_lot(self) -> None:
        target = {"AAA": 0.5, "BBB": 0.3}
        positions = pd.DataFrame(
            {
                "symbol": ["AAA", "CCC"],
                "shares": [100, 200],
            }
        )
        prices = pd.Series({"AAA": 10.0, "BBB": 20.0, "CCC": 5.0})

        orders = build_order_plan(
            target,
            positions,
            prices,
            cash=8_000.0,
            trade_date="2024-01-31",
            lot_size=100,
        )

        self.assertEqual(list(orders["symbol"]), ["CCC", "AAA", "BBB"])
        self.assertEqual(list(orders["side"]), ["SELL", "BUY", "BUY"])
        self.assertEqual(int(orders.loc[orders["symbol"] == "CCC", "delta_shares"].iloc[0]), -200)
        self.assertEqual(int(orders.loc[orders["symbol"] == "AAA", "target_shares"].iloc[0]), 500)
        self.assertEqual(int(orders.loc[orders["symbol"] == "BBB", "target_shares"].iloc[0]), 100)

    def test_min_order_amount_filters_tiny_orders(self) -> None:
        orders = build_order_plan(
            {"AAA": 0.61},
            {"AAA": 500},
            {"AAA": 10.0},
            cash=5_000.0,
            lot_size=100,
            min_order_amount=2_000.0,
            include_holds=True,
        )

        self.assertEqual(len(orders), 1)
        self.assertEqual(str(orders.loc[0, "side"]), "HOLD")
        self.assertEqual(str(orders.loc[0, "trade_reason"]), "below_min_order_amount")
        self.assertEqual(int(orders.loc[0, "delta_shares"]), 0)

    def test_target_weight_sum_cannot_exceed_one(self) -> None:
        with self.assertRaises(ValueError):
            build_order_plan(
                {"AAA": 0.7, "BBB": 0.4},
                {},
                {"AAA": 10.0, "BBB": 10.0},
                cash=10_000.0,
            )

    def test_build_from_latest_rebalance_meta(self) -> None:
        meta = {
            "rebalance_log": [
                {
                    "date": pd.Timestamp("2024-01-31"),
                    "picks": ["AAA", "BBB"],
                    "weights": [0.6, 0.4],
                }
            ]
        }
        dt, target = latest_target_weights_from_rebalance_meta(meta)
        self.assertEqual(dt, pd.Timestamp("2024-01-31"))
        self.assertAlmostEqual(float(target["AAA"]), 0.6)

        orders = build_order_plan_from_rebalance_meta(
            meta,
            current_positions=None,
            latest_prices={"AAA": 10.0, "BBB": 20.0},
            cash=10_000.0,
            lot_size=100,
        )

        self.assertEqual(list(orders["symbol"]), ["AAA", "BBB"])
        self.assertEqual(list(orders["side"]), ["BUY", "BUY"])
        self.assertEqual(int(orders.loc[orders["symbol"] == "AAA", "target_shares"].iloc[0]), 600)
        self.assertEqual(int(orders.loc[orders["symbol"] == "BBB", "target_shares"].iloc[0]), 200)


if __name__ == "__main__":
    unittest.main()
