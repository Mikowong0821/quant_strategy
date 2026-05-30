"""turnover：调仓日志换手率与成本估算自检。"""
from __future__ import annotations

import unittest

import pandas as pd

from analysis.turnover import summarize_turnover, turnover_frame, turnover_wide


class TestTurnover(unittest.TestCase):
    def test_turnover_frame_and_summary(self) -> None:
        log = [
            {
                "date": pd.Timestamp("2024-01-31"),
                "picks": ["AAA", "BBB"],
                "weights": [0.6, 0.4],
                "weighting": "equal",
            },
            {
                "date": pd.Timestamp("2024-02-29"),
                "picks": ["BBB", "CCC"],
                "weights": [0.5, 0.5],
                "weighting": "max_sharpe",
            },
        ]
        frame = turnover_frame(log, commission_rate=0.001)
        self.assertEqual(list(frame.columns), ["date", "turnover", "estimated_cost", "n_positions", "weighting"])
        self.assertAlmostEqual(float(frame.loc[0, "turnover"]), 1.0)
        self.assertAlmostEqual(float(frame.loc[1, "turnover"]), 1.2)
        self.assertAlmostEqual(float(frame.loc[1, "estimated_cost"]), 0.0012)
        self.assertEqual(int(frame.loc[1, "n_positions"]), 2)

        summary = summarize_turnover(log, commission_rate=0.001)
        self.assertAlmostEqual(float(summary["avg_turnover"]), 1.1)
        self.assertAlmostEqual(float(summary["total_turnover"]), 2.2)
        self.assertAlmostEqual(float(summary["estimated_total_cost"]), 0.0022)
        self.assertEqual(int(summary["n_turnover_periods"]), 2)

    def test_turnover_wide(self) -> None:
        f1 = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
                "turnover": [1.0, 0.5],
            }
        )
        f2 = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-31")],
                "turnover": [1.0],
            }
        )
        wide = turnover_wide({"A": f1, "B": f2})
        self.assertEqual(list(wide.columns), ["A", "B"])
        self.assertAlmostEqual(float(wide.loc[pd.Timestamp("2024-02-29"), "A"]), 0.5)
        self.assertTrue(pd.isna(wide.loc[pd.Timestamp("2024-02-29"), "B"]))


if __name__ == "__main__":
    unittest.main()
