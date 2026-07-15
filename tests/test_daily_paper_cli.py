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
from live.paper_guard import DailyPaperGuardError
from live.paper_run_control import DailyPaperRunControlError


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
            self.assertTrue((settings.output_dir / "paper_reports" / "TEST" / "2024-01-31.md").is_file())
            self.assertTrue((settings.output_dir / "live_orders" / "TEST" / "2024-01-31_manual_confirm.csv").is_file())
            self.assertIn("paper_report", result["paths"])
            self.assertIn("manual_confirmation", result["paths"])

    def test_run_from_outputs_adds_factor_health_to_report_and_summary(self) -> None:
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
            monitor_path = settings.output_dir / "factor_validation" / "factor_decay_monitor.csv"
            monitor_path.parent.mkdir(parents=True)
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
            pd.DataFrame(
                [
                    {
                        "factor": "ML_SCORE",
                        "status": "WATCH",
                        "reasons": "validation_positive_rate_below_threshold",
                        "validation_ic_mean": 0.01,
                        "validation_positive_rate": 0.45,
                    },
                ]
            ).to_csv(monitor_path, index=False)

            result = run_daily_paper_from_outputs(
                settings,
                strategy="TEST",
                factor_decay_monitor_path=monitor_path,
            )
            summary = format_daily_paper_summary(result)
            report_path = settings.output_dir / "paper_reports" / "TEST" / "2024-01-31.md"
            report = report_path.read_text(encoding="utf-8")

            self.assertIn("factor_health=WATCH", summary)
            self.assertIn("risky_factors=1", summary)
            self.assertIn("## 因子健康与失效监控", report)
            self.assertIn("| ML_SCORE | WATCH | validation_positive_rate_below_threshold |", report)

    def test_run_from_outputs_adds_style_exposure_to_report_and_summary(self) -> None:
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
            style_path = settings.output_dir / "factor_diagnostics" / "style_exposure.csv"
            style_path.parent.mkdir(parents=True)
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
            pd.DataFrame(
                [
                    {
                        "date": "2024-01-31",
                        "strategy": "TEST",
                        "style": "PRICE_VOLUME_STYLE",
                        "weighted_exposure": 1.25,
                        "abs_weighted_exposure": 1.25,
                        "score_coverage": 1.0,
                        "n_positions": 5,
                        "n_scored_positions": 5,
                    }
                ]
            ).to_csv(style_path, index=False)

            result = run_daily_paper_from_outputs(settings, strategy="TEST")
            summary = format_daily_paper_summary(result)
            report_path = settings.output_dir / "paper_reports" / "TEST" / "2024-01-31.md"
            report = report_path.read_text(encoding="utf-8")

            self.assertIn("style_exposure=PRICE_VOLUME_STYLE", summary)
            self.assertIn("2024-01-31:PRICE_VOLUME_STYLE:1.2500:positive", summary)
            self.assertIn("## 组合风格暴露", report)
            self.assertIn("| 2024-01-31 | PRICE_VOLUME_STYLE | 1.2500 |", report)

    def test_run_from_outputs_can_use_simulated_broker(self) -> None:
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

            result = run_daily_paper_from_outputs(
                settings,
                strategy="TEST",
                execution_mode="simulated_broker",
            )
            summary = format_daily_paper_summary(result)

            self.assertIn("execution_mode=simulated_broker", summary)
            self.assertEqual(list(result["broker_orders"]["status"]), ["FILLED"])
            self.assertAlmostEqual(result["cash"], 5_000.0)

    def test_run_control_blocks_non_trading_day_by_default(self) -> None:
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
                    {"date": "2024-01-26", "symbol": "AAA", "weight": 0.5, "selected": True},
                ]
            ).to_csv(settings.output_dir / "rebalance_logs" / "TEST.csv", index=False)
            pd.DataFrame(
                [
                    {"date": "2024-01-26", "AAA": 10.0},
                ]
            ).to_csv(settings.output_dir / "cache" / "prices_wide_close.csv", index=False)

            with self.assertRaises(DailyPaperRunControlError):
                run_daily_paper_from_outputs(settings, strategy="TEST", trade_date="2024-01-27")

            result = run_daily_paper_from_outputs(
                settings,
                strategy="TEST",
                trade_date="2024-01-27",
                allow_non_trading_day=True,
            )
            self.assertEqual(pd.Timestamp(result["trade_date"]).strftime("%Y-%m-%d"), "2024-01-27")

    def test_run_control_blocks_duplicate_snapshot_by_default(self) -> None:
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

            run_daily_paper_from_outputs(settings, strategy="TEST", generate_report=False)
            with self.assertRaises(DailyPaperRunControlError):
                run_daily_paper_from_outputs(settings, strategy="TEST", generate_report=False)

            result = run_daily_paper_from_outputs(
                settings,
                strategy="TEST",
                generate_report=False,
                allow_rerun=True,
            )
            self.assertEqual(pd.Timestamp(result["trade_date"]).strftime("%Y-%m-%d"), "2024-01-31")

    def test_guard_blocks_missing_target_price(self) -> None:
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
                    {"date": "2024-01-31", "BBB": 10.0},
                ]
            ).to_csv(settings.output_dir / "cache" / "prices_wide_close.csv", index=False)

            with self.assertRaises(DailyPaperGuardError):
                run_daily_paper_from_outputs(settings, strategy="TEST")

    def test_guard_warning_is_returned_for_stale_price(self) -> None:
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

            result = run_daily_paper_from_outputs(
                settings,
                strategy="TEST",
                trade_date="2024-02-20",
                max_price_age_days=3,
                allow_non_trading_day=True,
            )
            summary = format_daily_paper_summary(result)

            self.assertIn("stale_price_date", summary)
            self.assertEqual(result["guard_issues"][0].severity, "WARNING")


if __name__ == "__main__":
    unittest.main()
