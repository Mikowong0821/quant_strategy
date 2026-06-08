import unittest

import numpy as np
import pandas as pd

from analysis.ic import ic_distribution_summary, ic_rolling_stability, summarize_ic


class TestIcDiagnostics(unittest.TestCase):
    def test_distribution_summary_includes_quantiles_and_sign_rates(self):
        ser = pd.Series(
            [-0.2, -0.1, 0.0, 0.1, 0.3],
            index=pd.bdate_range("2024-01-01", periods=5),
            name="ic",
        )
        out = ic_distribution_summary({"FACTOR_A": ser})
        self.assertEqual(list(out["factor"]), ["FACTOR_A"])
        row = out.iloc[0]
        self.assertAlmostEqual(float(row["median"]), 0.0)
        self.assertAlmostEqual(float(row["positive_rate"]), 2 / 5)
        self.assertAlmostEqual(float(row["negative_rate"]), 2 / 5)
        self.assertIn("p05", out.columns)
        self.assertIn("p95", out.columns)

    def test_rolling_stability_outputs_each_window(self):
        ser = pd.Series(
            np.linspace(-0.1, 0.2, 12),
            index=pd.bdate_range("2024-01-01", periods=12),
            name="ic",
        )
        out = ic_rolling_stability({"FACTOR_A": ser}, windows=(3, 6))
        self.assertEqual(list(out["window"]), [3, 6])
        self.assertIn("rolling_mean_last", out.columns)
        self.assertIn("rolling_mean_positive_rate", out.columns)
        self.assertGreater(float(out[out["window"] == 3]["rolling_mean_last"].iloc[0]), 0.0)

    def test_summarize_ic_keeps_existing_contract(self):
        st = summarize_ic(pd.Series([0.1, 0.2, np.nan, -0.1]))
        self.assertEqual(st["n_days"], 3)
        self.assertAlmostEqual(st["hit_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
