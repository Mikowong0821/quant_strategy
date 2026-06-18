import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import get_settings
from live.account_state import save_account_state
from live.paper_run_control import (
    DailyPaperRunControlError,
    has_paper_snapshot,
    load_trading_calendar_from_prices,
    previous_trading_day,
    validate_daily_run_control,
)


class PaperRunControlTest(unittest.TestCase):
    def test_load_calendar_and_previous_trading_day(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "prices.csv"
            pd.DataFrame(
                [
                    {"date": "2024-01-26", "AAA": 10.0},
                    {"date": "2024-01-29", "AAA": 10.5},
                ]
            ).to_csv(path, index=False)

            calendar = load_trading_calendar_from_prices(path)

            self.assertEqual([x.strftime("%Y-%m-%d") for x in calendar], ["2024-01-26", "2024-01-29"])
            self.assertEqual(previous_trading_day("2024-01-28", calendar).strftime("%Y-%m-%d"), "2024-01-26")

    def test_non_trading_day_blocks_by_default(self):
        settings = get_settings()
        calendar = pd.DatetimeIndex(pd.to_datetime(["2024-01-26", "2024-01-29"]))

        with self.assertRaises(DailyPaperRunControlError):
            validate_daily_run_control(
                settings,
                strategy="TEST",
                trade_date="2024-01-27",
                trading_calendar=calendar,
            )

    def test_existing_snapshot_blocks_when_persisting(self):
        with tempfile.TemporaryDirectory() as td:
            settings = replace(get_settings(), output_dir=Path(td) / "output")
            save_account_state(
                settings,
                strategy="TEST",
                cash=1000.0,
                positions={},
                snapshot={
                    "cash": 1000.0,
                    "market_value": 0.0,
                    "total_asset": 1000.0,
                    "n_positions": 0,
                },
                trade_date="2024-01-31",
            )
            calendar = pd.DatetimeIndex(pd.to_datetime(["2024-01-31"]))

            self.assertTrue(has_paper_snapshot(settings, strategy="TEST", trade_date="2024-01-31"))
            with self.assertRaises(DailyPaperRunControlError):
                validate_daily_run_control(
                    settings,
                    strategy="TEST",
                    trade_date="2024-01-31",
                    trading_calendar=calendar,
                    persist_outputs=True,
                )

            validate_daily_run_control(
                settings,
                strategy="TEST",
                trade_date="2024-01-31",
                trading_calendar=calendar,
                persist_outputs=True,
                allow_rerun=True,
            )


if __name__ == "__main__":
    unittest.main()
