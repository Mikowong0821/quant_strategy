from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import get_settings
from live.factor_health_report import (
    build_factor_health_report,
    summarize_factor_health_report,
)


class FactorHealthReportTests(unittest.TestCase):
    def test_build_factor_health_report_from_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(get_settings(), output_dir=Path(td) / "output")
            (settings.output_dir / "factor_validation").mkdir(parents=True)
            (settings.output_dir / "factor_diagnostics").mkdir(parents=True)
            (settings.output_dir / "market_regime").mkdir(parents=True)

            pd.DataFrame(
                [
                    {
                        "factor": "ML_SCORE",
                        "status": "FAILED",
                        "reasons": "validation_ic_below_threshold",
                    },
                    {"factor": "ROE", "status": "OK", "reasons": ""},
                ]
            ).to_csv(settings.output_dir / "factor_validation" / "factor_decay_monitor.csv", index=False)
            pd.DataFrame(
                [
                    {"factor": "ML_SCORE", "status": "UNSTABLE", "stable_window_rate": 0.0},
                    {"factor": "ROE", "status": "STABLE", "stable_window_rate": 1.0},
                ]
            ).to_csv(
                settings.output_dir / "factor_validation" / "rolling_out_of_sample_summary.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "factor": "ROE",
                        "decision": "PASS",
                        "selected_for_fusion": True,
                    },
                    {
                        "factor": "ML_SCORE",
                        "decision": "WATCH",
                        "selected_for_fusion": False,
                    },
                ]
            ).to_csv(
                settings.output_dir / "factor_diagnostics" / "factor_selection_summary.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {"factor": "ROE", "status": "PASS"},
                ]
            ).to_csv(
                settings.output_dir / "factor_diagnostics" / "factor_weight_stability_summary.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "date": "2024-01-31",
                        "factor": "ML_SCORE",
                        "event_type": "weight_jump",
                        "severity": "WATCH",
                        "abs_weight_change": 0.12,
                    }
                ]
            ).to_csv(
                settings.output_dir / "factor_diagnostics" / "factor_weight_drift_events.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "factor_a": "ROE",
                        "factor_b": "NET_MARGIN",
                        "recommended_drop": "NET_MARGIN",
                    }
                ]
            ).to_csv(
                settings.output_dir / "factor_diagnostics" / "factor_redundancy_report.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "strategy": "TEST",
                        "status": "UNSTABLE",
                        "bull_excess_ann_return": -0.1,
                        "bear_excess_ann_return": -0.2,
                        "sideways_excess_ann_return": 0.1,
                        "positive_excess_regime_rate": 0.3333,
                    }
                ]
            ).to_csv(settings.output_dir / "market_regime" / "strategy_regime_summary.csv", index=False)

            report = build_factor_health_report(settings, strategy="TEST")
            status, detail = summarize_factor_health_report(report)

            self.assertEqual(status, "FAILED")
            self.assertIn("样本外失效", set(report["category"]))
            self.assertIn("牛熊市分段", set(report["category"]))
            self.assertIn("risky_categories", detail)


if __name__ == "__main__":
    unittest.main()
