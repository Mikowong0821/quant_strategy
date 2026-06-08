import unittest

import pandas as pd

from models.factor_weighting import build_factor_weight_summary


class TestFactorWeighting(unittest.TestCase):
    def test_build_factor_weight_summary_prefers_stronger_factor(self):
        ic_dist = pd.DataFrame(
            [
                {"factor": "GOOD", "mean_ic": 0.08, "ic_ir": 1.2, "positive_rate": 0.7},
                {"factor": "WEAK", "mean_ic": 0.01, "ic_ir": 0.2, "positive_rate": 0.52},
            ]
        )
        rolling = pd.DataFrame(
            [
                {
                    "factor": "GOOD",
                    "window": 60,
                    "rolling_mean_last": 0.06,
                    "rolling_mean_positive_rate": 0.8,
                },
                {
                    "factor": "WEAK",
                    "window": 60,
                    "rolling_mean_last": -0.01,
                    "rolling_mean_positive_rate": 0.45,
                },
            ]
        )
        group = pd.DataFrame(
            [
                {
                    "factor": "GOOD",
                    "group": 5,
                    "top_minus_bottom_ann": 0.12,
                    "monotonicity_score": 1.0,
                },
                {
                    "factor": "WEAK",
                    "group": 5,
                    "top_minus_bottom_ann": -0.02,
                    "monotonicity_score": 0.25,
                },
            ]
        )
        out = build_factor_weight_summary(
            ic_dist,
            rolling,
            group,
            factors=["GOOD", "WEAK"],
            preferred_rolling_window=60,
        )
        self.assertEqual(list(out["factor"]), ["GOOD", "WEAK"])
        self.assertAlmostEqual(float(out["fusion_weight"].sum()), 1.0)
        good = out[out["factor"] == "GOOD"].iloc[0]
        weak = out[out["factor"] == "WEAK"].iloc[0]
        self.assertGreater(float(good["factor_score"]), float(weak["factor_score"]))
        self.assertGreater(float(good["fusion_weight"]), float(weak["fusion_weight"]))

    def test_build_factor_weight_summary_falls_back_to_equal(self):
        ic_dist = pd.DataFrame(
            [
                {"factor": "A", "mean_ic": -0.1, "ic_ir": -1.0, "positive_rate": 0.2},
                {"factor": "B", "mean_ic": -0.2, "ic_ir": -2.0, "positive_rate": 0.1},
            ]
        )
        out = build_factor_weight_summary(ic_dist, pd.DataFrame(), pd.DataFrame())
        self.assertAlmostEqual(float(out["fusion_weight"].iloc[0]), 0.5)
        self.assertAlmostEqual(float(out["fusion_weight"].iloc[1]), 0.5)
        self.assertTrue(bool(out["weighting_fallback"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
