"""risk_exposure：持仓集中度指标自检。"""
from __future__ import annotations

import unittest

import pandas as pd

from analysis.risk_exposure import (
    concentration_frame,
    effective_n_wide,
    summarize_concentration,
)


class TestRiskExposure(unittest.TestCase):
    def test_concentration_frame_and_summary(self) -> None:
        log = [
            {
                "date": pd.Timestamp("2024-01-31"),
                "picks": ["AAA", "BBB"],
                "weights": [0.6, 0.4],
                "weighting": "max_sharpe",
            },
            {
                "date": pd.Timestamp("2024-02-29"),
                "picks": ["AAA", "BBB", "CCC"],
                "weights": [0.5, 0.3, 0.2],
                "weighting": "max_sharpe",
            },
        ]

        frame = concentration_frame(log)
        self.assertEqual(
            list(frame.columns),
            [
                "date",
                "hhi",
                "effective_n",
                "top1_weight",
                "top3_weight",
                "max_weight",
                "n_positions",
                "weighting",
            ],
        )
        self.assertAlmostEqual(float(frame.loc[0, "hhi"]), 0.52)
        self.assertAlmostEqual(float(frame.loc[0, "effective_n"]), 1.0 / 0.52)
        self.assertAlmostEqual(float(frame.loc[0, "top1_weight"]), 0.6)
        self.assertAlmostEqual(float(frame.loc[0, "top3_weight"]), 1.0)
        self.assertEqual(int(frame.loc[0, "n_positions"]), 2)

        self.assertAlmostEqual(float(frame.loc[1, "hhi"]), 0.38)
        self.assertAlmostEqual(float(frame.loc[1, "effective_n"]), 1.0 / 0.38)
        self.assertEqual(int(frame.loc[1, "n_positions"]), 3)

        summary = summarize_concentration(log)
        self.assertEqual(summary["n_concentration_periods"], 2)
        self.assertAlmostEqual(summary["avg_hhi"], (0.52 + 0.38) / 2.0)
        self.assertAlmostEqual(summary["max_top1_weight"], 0.6)
        self.assertEqual(summary["min_n_positions"], 2)

    def test_effective_n_wide(self) -> None:
        frame = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
                "effective_n": [2.0, 3.0],
            }
        )
        wide = effective_n_wide({"MOMENTUM": frame})
        self.assertEqual(list(wide.columns), ["MOMENTUM"])
        self.assertAlmostEqual(float(wide.iloc[1, 0]), 3.0)


if __name__ == "__main__":
    unittest.main()
