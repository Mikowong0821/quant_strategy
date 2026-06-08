"""cache_io：实验记录落盘自检。"""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import get_settings
from live.cache_io import (
    save_data_quality_reports,
    save_performance_summary,
    save_rebalance_logs,
    save_risk_exposure_logs,
    save_risk_exposure_summary,
    save_run_cache,
    save_run_config,
    save_turnover_logs,
)


class TestExperimentOutputs(unittest.TestCase):
    def test_run_config_performance_and_rebalance_logs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            settings = replace(
                get_settings(),
                output_dir=root / "output",
                data_dir=root / "data",
            )

            cfg_path = save_run_config(settings)
            self.assertTrue(cfg_path.is_file())
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.assertEqual(cfg["output_dir"], str(settings.output_dir))
            self.assertEqual(cfg["data_dir"], str(settings.data_dir))
            self.assertIn("written_utc", cfg)

            days = pd.bdate_range("2024-01-01", periods=2)
            long_df = pd.DataFrame(
                {
                    "trade_date": [days[0], days[1]],
                    "ts_code": ["AAA", "AAA"],
                    "close": [1.0, 1.1],
                    "volume": [100.0, 120.0],
                }
            )
            prices = pd.DataFrame({"AAA": [1.0, 1.1]}, index=days)
            idx = pd.MultiIndex.from_product([days, ["AAA"]], names=["date", "symbol"])
            panel = pd.DataFrame({"MOMENTUM": [0.0, 0.1]}, index=idx)
            cache_paths = save_run_cache(settings, long_df, prices, panel, panel_zscore=panel)
            self.assertTrue(cache_paths["factor_panel"].is_file())
            self.assertTrue(cache_paths["factor_panel_zscore"].is_file())

            dq_paths = save_data_quality_reports(
                settings,
                {
                    "factor_coverage": pd.DataFrame(
                        {"factor": ["MOMENTUM"], "coverage": [0.9]}
                    )
                },
            )
            self.assertTrue(dq_paths["factor_coverage"].is_file())

            perf_path = save_performance_summary(
                settings,
                {
                    "MOMENTUM": {
                        "ann_return": 0.12,
                        "ann_vol": 0.20,
                        "sharpe": 0.60,
                        "max_drawdown": -0.08,
                    }
                },
            )
            perf = pd.read_csv(perf_path)
            self.assertEqual(list(perf["strategy"]), ["MOMENTUM"])
            self.assertAlmostEqual(float(perf.loc[0, "sharpe"]), 0.60)

            logs = save_rebalance_logs(
                settings,
                {
                    "MOMENTUM": {
                        "rebalance_log": [
                            {
                                "date": pd.Timestamp("2024-01-31"),
                                "picks": ["AAA", "BBB"],
                                "weights": [0.6, 0.4],
                                "weighting": "max_sharpe",
                            }
                        ]
                    }
                },
            )
            log_path = logs["MOMENTUM"]
            self.assertTrue(log_path.is_file())
            log_df = pd.read_csv(log_path)
            self.assertEqual(
                list(log_df.columns),
                [
                    "date",
                    "symbol",
                    "weight",
                    "weighting",
                    "rank",
                    "selected",
                    "selected_rank",
                    "target_turnover",
                    "turnover_capped",
                    "turnover_scale",
                ],
            )
            self.assertEqual(list(log_df["symbol"]), ["AAA", "BBB"])
            self.assertEqual(list(log_df["rank"]), [1, 2])

            turnover_paths = save_turnover_logs(
                settings,
                {
                    "MOMENTUM": pd.DataFrame(
                        {
                            "date": [pd.Timestamp("2024-01-31")],
                            "turnover": [1.0],
                            "estimated_cost": [0.001],
                        }
                    )
                },
            )
            turnover_path = turnover_paths["MOMENTUM"]
            self.assertTrue(turnover_path.is_file())
            turnover_df = pd.read_csv(turnover_path)
            self.assertAlmostEqual(float(turnover_df.loc[0, "turnover"]), 1.0)

            risk_paths = save_risk_exposure_logs(
                settings,
                {
                    "MOMENTUM": pd.DataFrame(
                        {
                            "date": [pd.Timestamp("2024-01-31")],
                            "hhi": [0.52],
                            "effective_n": [1.0 / 0.52],
                            "top1_weight": [0.6],
                        }
                    )
                },
            )
            risk_path = risk_paths["MOMENTUM"]
            self.assertTrue(risk_path.is_file())
            risk_df = pd.read_csv(risk_path)
            self.assertAlmostEqual(float(risk_df.loc[0, "hhi"]), 0.52)

            risk_summary_path = save_risk_exposure_summary(
                settings,
                {"MOMENTUM": {"avg_effective_n": 1.0 / 0.52, "max_hhi": 0.52}},
            )
            self.assertTrue(risk_summary_path.is_file())
            risk_summary = pd.read_csv(risk_summary_path)
            self.assertEqual(list(risk_summary["strategy"]), ["MOMENTUM"])


if __name__ == "__main__":
    unittest.main()
