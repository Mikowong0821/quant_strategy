import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import get_settings
from live.paper_scheduler import run_scheduled_daily_paper


class PaperSchedulerTest(unittest.TestCase):
    def _settings_with_inputs(self, td: str):
        settings = replace(
            get_settings(),
            output_dir=Path(td) / "output",
            data_dir=Path(td) / "data",
            paper_initial_cash=10_000.0,
            commission_rate=0.0,
        )
        (settings.output_dir / "rebalance_logs").mkdir(parents=True)
        (settings.output_dir / "cache").mkdir(parents=True)
        pd.DataFrame(
            [{"date": "2024-01-31", "symbol": "AAA", "weight": 0.5, "selected": True}]
        ).to_csv(settings.output_dir / "rebalance_logs" / "TEST.csv", index=False)
        pd.DataFrame([{"date": "2024-01-31", "AAA": 10.0}]).to_csv(
            settings.output_dir / "cache" / "prices_wide_close.csv",
            index=False,
        )
        return settings

    def test_scheduled_run_writes_success_log(self):
        with tempfile.TemporaryDirectory() as td:
            settings = self._settings_with_inputs(td)

            result = run_scheduled_daily_paper(
                settings,
                daily_args=["--strategy", "TEST", "--no-persist", "--no-report"],
                log_date="2024-01-31",
            )

            self.assertEqual(0, result["exit_code"])
            self.assertTrue(result["log_path"].is_file())
            text = result["log_path"].read_text(encoding="utf-8")
            self.assertIn("exit_code=0", text)
            self.assertIn("strategy=TEST", text)
            self.assertIn("--strategy TEST --no-persist --no-report", text)

    def test_scheduled_run_writes_failure_log(self):
        with tempfile.TemporaryDirectory() as td:
            settings = self._settings_with_inputs(td)

            result = run_scheduled_daily_paper(
                settings,
                daily_args=["--strategy", "TEST", "--trade-date", "2024-02-01", "--no-persist"],
                log_date="2024-02-01",
            )

            self.assertEqual(1, result["exit_code"])
            text = result["log_path"].read_text(encoding="utf-8")
            self.assertIn("exit_code=1", text)
            self.assertIn("不在交易日日历中", text)


if __name__ == "__main__":
    unittest.main()
