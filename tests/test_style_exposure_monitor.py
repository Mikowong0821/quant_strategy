import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import get_settings
from live.style_exposure_monitor import (
    latest_style_exposure_for_strategy,
    load_style_exposure,
    summarize_style_exposure_for_report,
)


class StyleExposureMonitorTest(unittest.TestCase):
    def test_load_and_select_latest_style_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(get_settings(), output_dir=Path(td) / "output")
            path = settings.output_dir / "factor_diagnostics" / "style_exposure.csv"
            path.parent.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "date": "2024-01-31",
                        "strategy": "FUSED",
                        "style": "QUALITY_STYLE",
                        "weighted_exposure": 0.5,
                        "abs_weighted_exposure": 0.5,
                        "score_coverage": 1.0,
                        "n_positions": 5,
                        "n_scored_positions": 5,
                    },
                    {
                        "date": "2024-02-29",
                        "strategy": "FUSED",
                        "style": "PRICE_VOLUME_STYLE",
                        "weighted_exposure": 1.2,
                        "abs_weighted_exposure": 1.2,
                        "score_coverage": 0.9,
                        "n_positions": 6,
                        "n_scored_positions": 5,
                    },
                    {
                        "date": "2024-02-29",
                        "strategy": "OTHER",
                        "style": "QUALITY_STYLE",
                        "weighted_exposure": 9.9,
                    },
                ]
            ).to_csv(path, index=False)

            exposure = load_style_exposure(settings)
            latest = latest_style_exposure_for_strategy(
                exposure,
                strategy="FUSED",
                trade_date="2024-03-05",
            )
            status, detail = summarize_style_exposure_for_report(latest)

            self.assertEqual(list(latest["style"]), ["PRICE_VOLUME_STYLE"])
            self.assertEqual(status, "PRICE_VOLUME_STYLE")
            self.assertIn("2024-02-29:PRICE_VOLUME_STYLE:1.2000:positive", detail)

    def test_missing_file_returns_unknown_summary(self) -> None:
        settings = replace(get_settings(), output_dir=Path("/tmp/not-existing-style-exposure"))
        exposure = load_style_exposure(settings)
        status, reason = summarize_style_exposure_for_report(exposure)
        self.assertEqual(status, "UNKNOWN")
        self.assertEqual(reason, "style_exposure_missing")


if __name__ == "__main__":
    unittest.main()
