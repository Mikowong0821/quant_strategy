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
    save_performance_summary,
    save_rebalance_logs,
    save_run_config,
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
                ["date", "symbol", "weight", "weighting", "rank"],
            )
            self.assertEqual(list(log_df["symbol"]), ["AAA", "BBB"])
            self.assertEqual(list(log_df["rank"]), [1, 2])


if __name__ == "__main__":
    unittest.main()
