"""factors.preprocess：因子清洗与标准化自检。"""
from __future__ import annotations

import unittest

import pandas as pd

from factors.preprocess import (
    cross_sectional_zscore,
    industry_neutral_zscore,
    preprocess_factor_panel,
    winsorize_series,
)


class TestFactorPreprocess(unittest.TestCase):
    def test_winsorize_series_clips_extremes(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 100.0])
        out = winsorize_series(s, lower_q=0.0, upper_q=0.75)
        self.assertLess(float(out.iloc[-1]), 100.0)
        self.assertAlmostEqual(float(out.iloc[-1]), float(s.quantile(0.75)))

    def test_cross_sectional_zscore_by_date(self) -> None:
        days = pd.to_datetime(["2024-01-01", "2024-01-02"])
        idx = pd.MultiIndex.from_product([days, ["A", "B", "C"]], names=["date", "symbol"])
        panel = pd.DataFrame(
            {
                "F1": [1.0, 2.0, 3.0, 10.0, 10.0, 10.0],
                "F2": [1.0, None, 5.0, 2.0, 3.0, 4.0],
            },
            index=idx,
        )
        z = cross_sectional_zscore(panel, winsorize=False)
        first = z.xs(days[0], level=0)["F1"]
        self.assertAlmostEqual(float(first.mean()), 0.0)
        self.assertAlmostEqual(float(first.std(ddof=0)), 1.0)
        second = z.xs(days[1], level=0)["F1"]
        self.assertTrue((second == 0.0).all())

    def test_preprocess_factor_panel_preserves_shape(self) -> None:
        days = pd.to_datetime(["2024-01-01", "2024-01-02"])
        idx = pd.MultiIndex.from_product([days, ["A", "B"]], names=["date", "symbol"])
        panel = pd.DataFrame({"F1": [1.0, 2.0, 3.0, 4.0]}, index=idx)
        z = preprocess_factor_panel(panel)
        self.assertEqual(z.shape, panel.shape)
        self.assertEqual(list(z.columns), ["F1"])
        self.assertEqual(z.index.names, ["date", "symbol"])

    def test_industry_neutral_zscore_standardizes_within_industry(self) -> None:
        day = pd.Timestamp("2024-01-01")
        idx = pd.MultiIndex.from_product([[day], ["A", "B", "C", "D"]], names=["date", "symbol"])
        panel = pd.DataFrame({"VALUE": [1.0, 3.0, 100.0, 300.0]}, index=idx)
        industry = pd.Series(["Tech", "Tech", "Bank", "Bank"], index=idx, name="industry")

        z = industry_neutral_zscore(
            panel,
            industry,
            winsorize=False,
            min_count=2,
            min_industry_count=2,
        )

        self.assertAlmostEqual(float(z.loc[(day, "A"), "VALUE"]), -1.0)
        self.assertAlmostEqual(float(z.loc[(day, "B"), "VALUE"]), 1.0)
        self.assertAlmostEqual(float(z.loc[(day, "C"), "VALUE"]), -1.0)
        self.assertAlmostEqual(float(z.loc[(day, "D"), "VALUE"]), 1.0)

    def test_industry_neutral_zscore_falls_back_for_small_groups(self) -> None:
        day = pd.Timestamp("2024-01-01")
        idx = pd.MultiIndex.from_product([[day], ["A", "B", "C"]], names=["date", "symbol"])
        panel = pd.DataFrame({"VALUE": [1.0, 2.0, 100.0]}, index=idx)
        industry = pd.Series(["Tech", "Tech", "Solo"], index=idx, name="industry")

        z = industry_neutral_zscore(
            panel,
            industry,
            winsorize=False,
            min_count=2,
            min_industry_count=2,
        )
        global_z = cross_sectional_zscore(panel, winsorize=False)

        self.assertAlmostEqual(float(z.loc[(day, "A"), "VALUE"]), -1.0)
        self.assertAlmostEqual(float(z.loc[(day, "B"), "VALUE"]), 1.0)
        self.assertAlmostEqual(
            float(z.loc[(day, "C"), "VALUE"]),
            float(global_z.loc[(day, "C"), "VALUE"]),
        )


if __name__ == "__main__":
    unittest.main()
