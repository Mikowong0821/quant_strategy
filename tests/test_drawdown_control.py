"""账户级回撤止损与降仓控制。"""
from __future__ import annotations

import unittest

import pandas as pd

from live.drawdown_control import (
    apply_drawdown_control_to_weights,
    build_current_account_snapshot,
    default_drawdown_rules,
    evaluate_drawdown_control,
    summarize_drawdown_control,
)


class TestDrawdownControl(unittest.TestCase):
    def test_no_history_keeps_target_weights(self) -> None:
        control = evaluate_drawdown_control(
            default_drawdown_rules(),
            pd.DataFrame(columns=["date", "total_asset"]),
            {"cash": 10_000.0, "market_value": 0.0, "total_asset": 10_000.0, "n_positions": 0},
            {"AAA": 0.5, "BBB": 0.3},
            trade_date="2024-01-31",
        )

        self.assertEqual(str(control.loc[0, "status"]), "PASS")
        self.assertAlmostEqual(float(control.loc[0, "target_weight_scale"]), 1.0)
        self.assertAlmostEqual(float(control.loc[0, "target_exposure_after"]), 0.8)
        self.assertIn("drawdown=0.00%", summarize_drawdown_control(control)[1])

    def test_watch_drawdown_scales_target_weights(self) -> None:
        snapshots = pd.DataFrame(
            [
                {"date": "2024-01-31", "cash": 0.0, "market_value": 10_000.0, "total_asset": 10_000.0, "n_positions": 1},
                {"date": "2024-02-29", "cash": 0.0, "market_value": 9_800.0, "total_asset": 9_800.0, "n_positions": 1},
            ]
        )
        control = evaluate_drawdown_control(
            default_drawdown_rules(),
            snapshots,
            {"cash": 1_000.0, "market_value": 8_200.0, "total_asset": 9_200.0, "n_positions": 1},
            {"AAA": 0.6},
            trade_date="2024-03-29",
        )
        adjusted = apply_drawdown_control_to_weights({"AAA": 0.6}, control)

        self.assertEqual(str(control.loc[0, "status"]), "WATCH")
        self.assertEqual(str(control.loc[0, "triggered_rule_id"]), "drawdown_watch_5pct")
        self.assertAlmostEqual(float(control.loc[0, "drawdown_abs"]), 0.08)
        self.assertAlmostEqual(float(adjusted["AAA"]), 0.42)

    def test_stop_drawdown_moves_target_to_cash(self) -> None:
        snapshots = pd.DataFrame(
            [
                {"date": "2024-01-31", "cash": 0.0, "market_value": 10_000.0, "total_asset": 10_000.0, "n_positions": 1},
            ]
        )
        control = evaluate_drawdown_control(
            default_drawdown_rules(),
            snapshots,
            {"cash": 500.0, "market_value": 7_500.0, "total_asset": 8_000.0, "n_positions": 1},
            {"AAA": 0.6},
            trade_date="2024-02-29",
        )
        adjusted = apply_drawdown_control_to_weights({"AAA": 0.6}, control)

        self.assertEqual(str(control.loc[0, "status"]), "BLOCK")
        self.assertEqual(str(control.loc[0, "triggered_rule_id"]), "drawdown_stop_15pct")
        self.assertTrue(adjusted.empty)

    def test_build_current_snapshot_uses_latest_prices(self) -> None:
        snapshot = build_current_account_snapshot(
            cash=100.0,
            positions=pd.DataFrame([{"symbol": "AAA", "shares": 10}]),
            latest_prices=pd.Series({"AAA": 12.0}),
        )

        self.assertAlmostEqual(snapshot["market_value"], 120.0)
        self.assertAlmostEqual(snapshot["total_asset"], 220.0)
        self.assertEqual(snapshot["n_positions"], 1.0)


if __name__ == "__main__":
    unittest.main()
