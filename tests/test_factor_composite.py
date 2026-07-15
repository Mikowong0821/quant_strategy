import unittest

import numpy as np
import pandas as pd

from analysis.factor_composite import build_factor_composite_scores


class FactorCompositeTest(unittest.TestCase):
    def test_build_factor_composite_scores_uses_eligible_components(self):
        dates = pd.to_datetime(["2026-01-02", "2026-01-03"])
        symbols = ["000001.SZ", "000002.SZ"]
        idx = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
        panel = pd.DataFrame(
            {
                "MOMENTUM": [1.0, 2.0, 3.0, 4.0],
                "MOMENTUM_60D": [2.0, 4.0, 6.0, 8.0],
                "PE": [-10.0, -8.0, -6.0, -4.0],
                "ROE": [0.1, 0.2, np.nan, 0.4],
            },
            index=idx,
        )

        scores, components = build_factor_composite_scores(
            panel,
            eligible_factors=["MOMENTUM", "MOMENTUM_60D", "ROE"],
        )

        self.assertIn("PRICE_VOLUME_STYLE", scores.columns)
        self.assertIn("QUALITY_STYLE", scores.columns)
        self.assertNotIn("VALUE_STYLE", scores.columns)
        self.assertAlmostEqual(scores.loc[(dates[0], "000001.SZ"), "PRICE_VOLUME_STYLE"], 1.5)
        self.assertAlmostEqual(scores.loc[(dates[0], "000002.SZ"), "PRICE_VOLUME_STYLE"], 3.0)
        self.assertAlmostEqual(scores.loc[(dates[0], "000001.SZ"), "QUALITY_STYLE"], 0.1)
        self.assertTrue(pd.isna(scores.loc[(dates[1], "000001.SZ"), "QUALITY_STYLE"]))

        price_row = components.loc[components["composite_factor"] == "PRICE_VOLUME_STYLE"].iloc[0]
        self.assertEqual(price_row["eligible_components"], "MOMENTUM,MOMENTUM_60D")
        self.assertIn("REVERSAL_5D", price_row["missing_candidates"])
        self.assertIn("VOLATILITY", price_row["missing_candidates"])


if __name__ == "__main__":
    unittest.main()
