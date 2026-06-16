"""纸面账户状态持久化。"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import get_settings
from live.account_state import load_account_state, positions_from_trades, save_account_state


class TestAccountState(unittest.TestCase):
    def test_positions_from_trades_updates_initial_positions(self) -> None:
        trades = pd.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "position_after": [50, 100],
                "fill_status": ["FILLED", "FILLED"],
            }
        )

        positions = positions_from_trades(
            trades,
            {"AAA": 100, "CCC": 200},
            updated_at="2024-01-31",
        )

        got = dict(zip(positions["symbol"], positions["shares"], strict=True))
        self.assertEqual(got, {"AAA": 50.0, "BBB": 100.0, "CCC": 200.0})
        self.assertTrue((positions["updated_at"] == "2024-01-31").all())

    def test_save_and_load_account_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
            )
            paths = save_account_state(
                settings,
                strategy="MOMENTUM",
                cash=1234.5,
                positions={"AAA": 100},
                snapshot={
                    "cash": 1234.5,
                    "market_value": 1000.0,
                    "total_asset": 2234.5,
                    "n_positions": 1,
                },
                trade_date="2024-01-31",
            )

            self.assertTrue(paths["account"].is_file())
            self.assertTrue(paths["positions"].is_file())
            self.assertTrue(paths["snapshots"].is_file())

            cash, positions = load_account_state(
                settings,
                strategy="MOMENTUM",
                default_cash=1_000_000.0,
            )

            self.assertAlmostEqual(cash, 1234.5)
            self.assertEqual(list(positions["symbol"]), ["AAA"])
            self.assertEqual(float(positions.loc[0, "shares"]), 100.0)

    def test_snapshot_replaces_same_date_and_appends_new_date(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
            )
            save_account_state(
                settings,
                strategy="MOMENTUM",
                cash=1000.0,
                positions={},
                snapshot={"cash": 1000.0, "market_value": 0.0, "total_asset": 1000.0, "n_positions": 0},
                trade_date="2024-01-31",
            )
            save_account_state(
                settings,
                strategy="MOMENTUM",
                cash=1100.0,
                positions={},
                snapshot={"cash": 1100.0, "market_value": 0.0, "total_asset": 1100.0, "n_positions": 0},
                trade_date="2024-01-31",
            )
            paths = save_account_state(
                settings,
                strategy="MOMENTUM",
                cash=1200.0,
                positions={},
                snapshot={"cash": 1200.0, "market_value": 0.0, "total_asset": 1200.0, "n_positions": 0},
                trade_date="2024-02-29",
            )

            snapshots = pd.read_csv(paths["snapshots"])
            self.assertEqual(list(snapshots["date"]), ["2024-01-31", "2024-02-29"])
            self.assertEqual(list(snapshots["cash"]), [1100.0, 1200.0])


if __name__ == "__main__":
    unittest.main()
