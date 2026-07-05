"""纸面账户与只读券商账户对账。"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from config import get_settings
from live.account_state import save_account_state
from live.broker import BrokerAccount, RealBrokerConfig, RealBrokerReadOnlyAdapter
from live.broker_reconcile import reconcile_paper_with_broker, save_reconciliation_outputs


class BrokerReconcileTest(unittest.TestCase):
    def test_reconcile_ok_when_account_and_positions_match(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
            )
            save_account_state(
                settings,
                strategy="TEST",
                cash=1_000.0,
                positions={"AAA": 100},
                snapshot={"cash": 1_000.0, "market_value": 2_000.0, "total_asset": 3_000.0, "n_positions": 1},
                trade_date="2024-01-31",
            )
            broker = RealBrokerReadOnlyAdapter(
                RealBrokerConfig(provider="demo"),
                account=BrokerAccount(cash=1_000.0, market_value=2_000.0, total_asset=3_000.0),
                positions={"AAA": 100},
                latest_prices={"AAA": 20.0},
            )

            result = reconcile_paper_with_broker(
                settings,
                strategy="TEST",
                broker=broker,
                trade_date="2024-01-31",
            )

            self.assertEqual(result["issues"], [])
            self.assertEqual(str(result["account_summary"].loc[0, "cash_status"]), "OK")
            self.assertEqual(set(result["position_diff"]["status"]), {"OK"})

            paths = save_reconciliation_outputs(settings, result)
            self.assertTrue(paths["account_summary"].is_file())
            self.assertTrue(paths["position_diff"].is_file())
            self.assertTrue(paths["report"].is_file())

    def test_reconcile_reports_cash_and_position_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
            )
            save_account_state(
                settings,
                strategy="TEST",
                cash=1_000.0,
                positions={"AAA": 100},
                snapshot={"cash": 1_000.0, "market_value": 2_000.0, "total_asset": 3_000.0, "n_positions": 1},
                trade_date="2024-01-31",
            )
            broker = RealBrokerReadOnlyAdapter(
                RealBrokerConfig(provider="demo"),
                account=BrokerAccount(cash=900.0, market_value=4_000.0, total_asset=4_900.0),
                positions={"AAA": 90, "BBB": 10},
                latest_prices={"AAA": 20.0, "BBB": 220.0},
            )

            result = reconcile_paper_with_broker(
                settings,
                strategy="TEST",
                broker=broker,
                trade_date="2024-01-31",
            )

            self.assertIn("cash_mismatch", result["issues"])
            self.assertIn("asset_mismatch", result["issues"])
            self.assertIn("position_mismatch", result["issues"])
            status_by_symbol = dict(zip(result["position_diff"]["symbol"], result["position_diff"]["status"], strict=True))
            self.assertEqual(status_by_symbol["AAA"], "MISMATCH")
            self.assertEqual(status_by_symbol["BBB"], "BROKER_ONLY")


if __name__ == "__main__":
    unittest.main()
