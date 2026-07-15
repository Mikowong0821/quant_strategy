import unittest

import pandas as pd

from analysis.style_exposure import (
    batch_style_exposure,
    style_exposure_frame,
    style_exposure_return_link,
    summarize_style_exposure,
)


class StyleExposureTest(unittest.TestCase):
    def setUp(self):
        idx = pd.MultiIndex.from_product(
            [
                pd.to_datetime(["2026-01-31", "2026-02-28"]),
                ["000001.SZ", "000002.SZ"],
            ],
            names=["date", "symbol"],
        )
        self.style_scores = pd.DataFrame(
            {
                "QUALITY_STYLE": [1.0, -1.0, 0.5, -0.5],
                "VALUE_STYLE": [-0.5, 0.5, -1.0, 1.0],
            },
            index=idx,
        )
        self.log = [
            {
                "date": pd.Timestamp("2026-01-31"),
                "picks": ["000001.SZ", "000002.SZ"],
                "weights": [0.7, 0.3],
            },
            {
                "date": pd.Timestamp("2026-02-28"),
                "picks": ["000001.SZ", "000002.SZ"],
                "weights": [0.5, 0.5],
            },
        ]

    def test_style_exposure_frame(self):
        frame = style_exposure_frame(self.log, self.style_scores, strategy="FUSED")
        quality = frame[
            (frame["date"] == pd.Timestamp("2026-01-31"))
            & (frame["style"] == "QUALITY_STYLE")
        ].iloc[0]
        self.assertAlmostEqual(quality["weighted_exposure"], 0.4)
        self.assertAlmostEqual(quality["score_coverage"], 1.0)
        self.assertEqual(quality["n_scored_positions"], 2)

    def test_batch_and_summary(self):
        meta = {"FUSED": {"rebalance_log": self.log}}
        exposure = batch_style_exposure(meta, self.style_scores, strategies=["FUSED"])
        summary = summarize_style_exposure(exposure)
        self.assertEqual(set(summary["style"]), {"QUALITY_STYLE", "VALUE_STYLE"})
        quality = summary[summary["style"] == "QUALITY_STYLE"].iloc[0]
        self.assertEqual(quality["n_periods"], 2)
        self.assertAlmostEqual(quality["positive_rate"], 0.5)

    def test_style_exposure_return_link(self):
        exposure = style_exposure_frame(self.log, self.style_scores, strategy="FUSED")
        nav = pd.Series(
            [1.0, 1.1, 1.2],
            index=pd.to_datetime(["2026-01-31", "2026-02-28", "2026-03-31"]),
            name="FUSED",
        )
        link = style_exposure_return_link(exposure, {"FUSED": nav})
        self.assertEqual(set(link["style"]), {"QUALITY_STYLE", "VALUE_STYLE"})
        self.assertTrue((link["n_periods"] == 2).all())


if __name__ == "__main__":
    unittest.main()
