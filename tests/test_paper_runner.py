"""每日纸面交易运行器。"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import get_settings
from live.account_state import load_account_state
from live.paper_runner import run_daily_paper_trade


class TestPaperRunner(unittest.TestCase):
    def test_daily_runner_persists_state_and_continues_next_day(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )

            day1 = run_daily_paper_trade(
                settings,
                strategy="MOMENTUM",
                target_weights={"AAA": 0.5},
                latest_prices={"AAA": 10.0},
                trade_date="2024-01-31",
            )

            self.assertAlmostEqual(day1["starting_cash"], 10_000.0)
            self.assertEqual(list(day1["orders"]["symbol"]), ["AAA"])
            self.assertEqual(str(day1["paper_trades"].loc[0, "fill_status"]), "FILLED")
            self.assertAlmostEqual(day1["cash"], 5_000.0)

            cash, positions = load_account_state(
                settings,
                strategy="MOMENTUM",
                default_cash=0.0,
            )
            self.assertAlmostEqual(cash, 5_000.0)
            self.assertEqual(list(positions["symbol"]), ["AAA"])
            self.assertEqual(float(positions.loc[0, "shares"]), 500.0)

            day2 = run_daily_paper_trade(
                settings,
                strategy="MOMENTUM",
                target_weights={"AAA": 0.0, "BBB": 0.5},
                latest_prices={"AAA": 10.0, "BBB": 20.0},
                trade_date="2024-02-29",
            )

            self.assertAlmostEqual(day2["starting_cash"], 5_000.0)
            self.assertEqual(list(day2["paper_trades"]["side"]), ["SELL", "BUY"])
            self.assertEqual(list(day2["paper_trades"]["fill_status"]), ["FILLED", "FILLED"])
            self.assertAlmostEqual(day2["cash"], 6_000.0)
            got_positions = dict(zip(day2["positions"]["symbol"], day2["positions"]["shares"], strict=True))
            self.assertEqual(got_positions, {"BBB": 200.0})

            snapshots_path = day2["paths"]["account_state"]["snapshots"]
            snapshots = pd.read_csv(snapshots_path)
            self.assertEqual(list(snapshots["date"]), ["2024-01-31", "2024-02-29"])

    def test_runner_handles_empty_orders_and_still_saves_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
            )

            result = run_daily_paper_trade(
                settings,
                strategy="EMPTY",
                target_weights={},
                latest_prices={},
                trade_date="2024-01-31",
            )

            self.assertTrue(result["orders"].empty)
            self.assertTrue(result["order_checks"].empty)
            self.assertTrue(result["paper_trades"].empty)
            self.assertAlmostEqual(result["cash"], 10_000.0)
            self.assertTrue(result["paths"]["account_state"]["snapshots"].is_file())

    def test_daily_runner_can_execute_through_simulated_broker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )

            result = run_daily_paper_trade(
                settings,
                strategy="BROKER_MODE",
                target_weights={"AAA": 0.5},
                latest_prices={"AAA": 10.0},
                trade_date="2024-01-31",
                execution_mode="simulated_broker",
            )

            self.assertEqual(result["execution_mode"], "simulated_broker")
            self.assertEqual(list(result["broker_orders"]["status"]), ["FILLED"])
            self.assertEqual(list(result["paper_trades"]["fill_status"]), ["FILLED"])
            self.assertAlmostEqual(result["cash"], 5_000.0)
            self.assertEqual(list(result["positions"]["symbol"]), ["AAA"])
            self.assertEqual(float(result["positions"].loc[0, "shares"]), 500.0)


if __name__ == "__main__":
    unittest.main()
