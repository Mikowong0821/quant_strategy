import unittest

import pandas as pd

from live.stress_test import (
    default_stress_scenarios,
    run_portfolio_stress_tests,
    summarize_stress_tests,
)


class StressTestTest(unittest.TestCase):
    def test_market_and_single_name_stress_tests(self) -> None:
        checks = run_portfolio_stress_tests(
            default_stress_scenarios(),
            {"AAA": 0.5, "BBB": 0.3, "CCC": 0.1},
            trade_date="2024-01-31",
            total_asset=100_000.0,
        )

        by_id = checks.set_index("scenario_id")
        self.assertEqual(str(by_id.loc["market_down_8pct", "status"]), "BLOCK")
        self.assertAlmostEqual(float(by_id.loc["market_down_8pct", "estimated_loss_pct"]), 0.072)
        self.assertAlmostEqual(float(by_id.loc["market_down_8pct", "estimated_loss_amount"]), 7200.0)
        self.assertEqual(str(by_id.loc["largest_position_down_10pct", "affected_symbols"]), "AAA")
        self.assertEqual(str(by_id.loc["largest_position_down_10pct", "status"]), "BLOCK")

    def test_largest_industry_stress_test_uses_industry_mapping(self) -> None:
        industry = pd.DataFrame(
            [
                {"symbol": "AAA", "industry": "食品"},
                {"symbol": "BBB", "industry": "食品"},
                {"symbol": "CCC", "industry": "化工"},
            ]
        )
        checks = run_portfolio_stress_tests(
            default_stress_scenarios(),
            {"AAA": 0.35, "BBB": 0.25, "CCC": 0.2},
            industry=industry,
        )

        row = checks.set_index("scenario_id").loc["largest_industry_down_8pct"]
        self.assertEqual(str(row["status"]), "WATCH")
        self.assertAlmostEqual(float(row["affected_weight"]), 0.6)
        self.assertIn("食品", str(row["details"]))

    def test_summarize_stress_tests(self) -> None:
        checks = pd.DataFrame(
            [
                {"status": "PASS", "scenario_id": "a", "estimated_loss_pct": 0.01},
                {"status": "WATCH", "scenario_id": "b", "estimated_loss_pct": 0.04},
            ]
        )
        status, detail = summarize_stress_tests(checks)

        self.assertEqual(status, "WATCH")
        self.assertIn("WATCH=1", detail)
        self.assertIn("worst=b 4.00%", detail)


if __name__ == "__main__":
    unittest.main()
