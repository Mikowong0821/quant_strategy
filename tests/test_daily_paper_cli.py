"""每日纸面交易命令行辅助逻辑。"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import get_settings
from live.daily_paper_cli import (
    format_daily_paper_summary,
    load_latest_prices,
    load_latest_target_weights,
    run_daily_paper_from_outputs,
)


class TestDailyPaperCli(unittest.TestCase):
    def test_load_latest_inputs_respects_trade_date(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rebalance = root / "rebalance.csv"
            prices = root / "prices.csv"
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "symbol": "AAA", "weight": 0.4, "selected": True},
                    {"date": "2024-01-31", "symbol": "BBB", "weight": 0.6, "selected": True},
                    {"date": "2024-02-29", "symbol": "CCC", "weight": 1.0, "selected": True},
                ]
            ).to_csv(rebalance, index=False)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "AAA": 10.0, "BBB": 20.0, "CCC": 30.0},
                    {"date": "2024-02-29", "AAA": 11.0, "BBB": 21.0, "CCC": 31.0},
                ]
            ).to_csv(prices, index=False)

            target_date, weights = load_latest_target_weights(rebalance, trade_date="2024-02-01")
            price_date, latest_prices = load_latest_prices(prices, trade_date="2024-02-01")

            self.assertEqual(target_date.strftime("%Y-%m-%d"), "2024-01-31")
            self.assertEqual(weights.to_dict(), {"AAA": 0.4, "BBB": 0.6})
            self.assertEqual(price_date.strftime("%Y-%m-%d"), "2024-01-31")
            self.assertAlmostEqual(float(latest_prices["AAA"]), 10.0)

    def test_run_from_outputs_writes_daily_account_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
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
                [
                    {"date": "2024-01-31", "symbol": "AAA", "weight": 0.5, "selected": True},
                ]
            ).to_csv(settings.output_dir / "rebalance_logs" / "TEST.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2024-01-31", "AAA": 10.0},
                ]
            ).to_csv(settings.output_dir / "cache" / "prices_wide_close.csv", index=False)

            result = run_daily_paper_from_outputs(settings, strategy="TEST")
            summary = format_daily_paper_summary(result)

            self.assertIn("strategy=TEST", summary)
            self.assertEqual(list(result["orders"]["symbol"]), ["AAA"])
            self.assertAlmostEqual(result["cash"], 5_000.0)
            self.assertTrue((settings.output_dir / "paper_account" / "TEST" / "snapshots.csv").is_file())


if __name__ == "__main__":
    unittest.main()
