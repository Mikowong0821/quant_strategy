import unittest

import pandas as pd

from analysis.factor_weight_stability import (
    factor_weight_drift_events,
    factor_weight_portfolio_drift,
    factor_weight_stability_summary,
)


class FactorWeightStabilityTests(unittest.TestCase):
    def setUp(self):
        self.log = pd.DataFrame(
            [
                {"date": "2026-01-31", "factor": "VALUE_STYLE", "final_weight": 0.50, "reason": "computed"},
                {"date": "2026-01-31", "factor": "QUALITY_STYLE", "final_weight": 0.50, "reason": "computed"},
                {"date": "2026-02-28", "factor": "VALUE_STYLE", "final_weight": 0.30, "reason": "computed_smoothed"},
                {"date": "2026-02-28", "factor": "QUALITY_STYLE", "final_weight": 0.70, "reason": "computed_smoothed"},
                {"date": "2026-03-31", "factor": "VALUE_STYLE", "final_weight": 0.25, "reason": "computed_smoothed"},
                {"date": "2026-03-31", "factor": "QUALITY_STYLE", "final_weight": 0.75, "reason": "computed_smoothed"},
            ]
        )

    def test_factor_weight_stability_summary(self):
        summary = factor_weight_stability_summary(
            self.log,
            watch_avg_change=0.15,
            watch_max_change=0.18,
        )
        value = summary[summary["factor"] == "VALUE_STYLE"].iloc[0]
        self.assertEqual(int(value["n_periods"]), 3)
        self.assertAlmostEqual(float(value["latest_weight"]), 0.25)
        self.assertAlmostEqual(float(value["max_abs_change"]), 0.20)
        self.assertEqual(str(value["status"]), "WATCH")

    def test_factor_weight_drift_events(self):
        events = factor_weight_drift_events(
            self.log,
            change_threshold=0.10,
            high_change_threshold=0.18,
        )
        self.assertEqual(len(events), 2)
        self.assertEqual(set(events["event_type"]), {"weight_drop", "weight_jump"})
        self.assertIn("HIGH", set(events["severity"]))

    def test_factor_weight_portfolio_drift(self):
        drift = factor_weight_portfolio_drift(self.log)
        latest = drift.sort_values("date").iloc[-1]
        self.assertEqual(str(latest["dominant_factor"]), "QUALITY_STYLE")
        self.assertAlmostEqual(float(latest["dominant_weight"]), 0.75)
        self.assertGreater(float(latest["effective_factor_n"]), 1.0)
        self.assertAlmostEqual(float(drift.iloc[1]["weight_turnover"]), 0.20)


if __name__ == "__main__":
    unittest.main()
