"""容量与冲击成本估算。"""
from __future__ import annotations

import unittest

import pandas as pd

from live.capacity_impact import (
    average_amount_by_symbol,
    default_capacity_rules,
    evaluate_capacity_impact,
    summarize_capacity_impact,
)


class TestCapacityImpact(unittest.TestCase):
    def test_average_amount_by_symbol_uses_lookback_records(self) -> None:
        liquidity = pd.DataFrame(
            [
                {"trade_date": "2024-01-01", "ts_code": "AAA", "amount": 1000.0},
                {"trade_date": "2024-01-02", "ts_code": "AAA", "amount": 2000.0},
                {"trade_date": "2024-01-03", "ts_code": "AAA", "amount": 4000.0},
                {"trade_date": "2024-01-01", "ts_code": "BBB", "amount": 8000.0},
            ]
        )

        avg = average_amount_by_symbol(liquidity, trade_date="2024-01-03", lookback_days=2)

        self.assertAlmostEqual(float(avg["AAA"]), 3000.0)
        self.assertAlmostEqual(float(avg["BBB"]), 8000.0)

    def test_evaluate_capacity_impact_passes_small_orders(self) -> None:
        orders = pd.DataFrame(
            [
                {"symbol": "AAA", "side": "BUY", "estimated_amount": 10_000.0},
            ]
        )
        liquidity = pd.DataFrame(
            [
                {"date": "2024-01-01", "symbol": "AAA", "amount": 1_000_000.0},
                {"date": "2024-01-02", "symbol": "AAA", "amount": 1_000_000.0},
            ]
        )

        detail, summary = evaluate_capacity_impact(
            orders,
            liquidity,
            trade_date="2024-01-02",
            lookback_days=2,
            impact_coefficient_bps=100.0,
        )

        self.assertEqual(str(detail.loc[0, "status"]), "PASS")
        self.assertAlmostEqual(float(detail.loc[0, "participation_rate"]), 0.01)
        self.assertAlmostEqual(float(detail.loc[0, "impact_cost_bps"]), 10.0)
        self.assertEqual(str(summary.loc[0, "status"]), "PASS")
        self.assertIn("max_participation=1.00%", summarize_capacity_impact(summary)[1])

    def test_evaluate_capacity_impact_blocks_large_participation(self) -> None:
        orders = pd.DataFrame(
            [
                {"symbol": "AAA", "side": "BUY", "estimated_amount": 150_000.0},
            ]
        )
        liquidity = pd.DataFrame(
            [
                {"date": "2024-01-02", "symbol": "AAA", "amount": 1_000_000.0},
            ]
        )

        detail, summary = evaluate_capacity_impact(
            orders,
            liquidity,
            trade_date="2024-01-02",
            rules=default_capacity_rules(),
        )

        self.assertEqual(str(detail.loc[0, "status"]), "BLOCK")
        self.assertAlmostEqual(float(detail.loc[0, "participation_rate"]), 0.15)
        self.assertEqual(str(summary.loc[0, "status"]), "BLOCK")

    def test_evaluate_capacity_impact_marks_missing_liquidity_na(self) -> None:
        orders = pd.DataFrame(
            [
                {"symbol": "AAA", "side": "BUY", "estimated_amount": 10_000.0},
            ]
        )

        detail, summary = evaluate_capacity_impact(
            orders,
            pd.DataFrame(),
            trade_date="2024-01-02",
        )

        self.assertEqual(str(detail.loc[0, "status"]), "NA")
        self.assertEqual(str(summary.loc[0, "status"]), "NA")
        self.assertEqual(int(summary.loc[0, "n_missing_liquidity"]), 1)


if __name__ == "__main__":
    unittest.main()
