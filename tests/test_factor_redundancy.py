import unittest

import pandas as pd

from analysis.factor_redundancy import (
    build_factor_redundancy_report,
    factor_cross_sectional_correlation,
    prune_redundant_factors,
)


class FactorRedundancyTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2026-01-01", periods=30)
        symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        base = [1, 2, 3, 4, 5] * len(dates)
        same = [x * 2 for x in base]
        opposite = [-x for x in base]
        noisy = ([5, 1, 4, 2, 3] * len(dates))
        self.panel = pd.DataFrame(
            {
                "GOOD": base,
                "DUPLICATE": same,
                "OPPOSITE": opposite,
                "OTHER": noisy,
            },
            index=idx,
            dtype=float,
        )

    def test_factor_cross_sectional_correlation(self):
        corr, days = factor_cross_sectional_correlation(
            self.panel,
            factors=["GOOD", "DUPLICATE", "OPPOSITE", "OTHER"],
            min_symbols=5,
        )
        self.assertAlmostEqual(float(corr.loc["GOOD", "DUPLICATE"]), 1.0)
        self.assertAlmostEqual(float(corr.loc["GOOD", "OPPOSITE"]), -1.0)
        self.assertEqual(int(days.loc["GOOD", "DUPLICATE"]), 30)

    def test_redundancy_report_and_pruning(self):
        corr, days = factor_cross_sectional_correlation(self.panel, min_symbols=5)
        selection = pd.DataFrame(
            [
                {"factor": "GOOD", "decision": "PASS", "factor_score": 0.8},
                {"factor": "DUPLICATE", "decision": "PASS", "factor_score": 0.4},
                {"factor": "OPPOSITE", "decision": "WATCH", "factor_score": 0.9},
                {"factor": "OTHER", "decision": "PASS", "factor_score": 0.2},
            ]
        )
        report = build_factor_redundancy_report(
            corr,
            days,
            selection=selection,
            threshold=0.9,
            min_days=20,
        )
        pair = report[
            (report["factor_a"] == "GOOD") & (report["factor_b"] == "DUPLICATE")
        ].iloc[0]
        self.assertEqual(str(pair["recommended_keep"]), "GOOD")
        self.assertEqual(str(pair["recommended_drop"]), "DUPLICATE")

        pruned = prune_redundant_factors(["GOOD", "DUPLICATE", "OTHER"], report)
        self.assertEqual(pruned, ["GOOD", "OTHER"])


if __name__ == "__main__":
    unittest.main()
