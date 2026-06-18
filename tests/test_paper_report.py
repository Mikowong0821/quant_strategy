"""纸面交易日报。"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import get_settings
from live.paper_guard import GuardIssue
from live.paper_report import build_daily_paper_report, save_daily_paper_report
from live.paper_runner import run_daily_paper_trade


class TestPaperReport(unittest.TestCase):
    def test_report_contains_summary_orders_and_positions(self) -> None:
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
                strategy="REPORT",
                target_weights={"AAA": 0.5},
                latest_prices={"AAA": 10.0},
                trade_date="2024-01-31",
            )
            result["target_date"] = pd.Timestamp("2024-01-31")
            result["price_date"] = pd.Timestamp("2024-01-31")

            report = build_daily_paper_report(result)
            path = save_daily_paper_report(settings, result)

            self.assertIn("# 纸面交易日报 - REPORT - 2024-01-31", report)
            self.assertIn("- 订单数：1", report)
            self.assertIn("| AAA | BUY | 500 |", report)
            self.assertIn("## 当前持仓", report)
            self.assertTrue(path.is_file())
            self.assertIn("纸面交易日报", path.read_text(encoding="utf-8"))

    def test_report_includes_previous_snapshot_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            run_daily_paper_trade(
                settings,
                strategy="REPORT",
                target_weights={"AAA": 0.5},
                latest_prices={"AAA": 10.0},
                trade_date="2024-01-31",
            )
            result = run_daily_paper_trade(
                settings,
                strategy="REPORT",
                target_weights={"AAA": 0.5},
                latest_prices={"AAA": 12.0},
                trade_date="2024-02-29",
            )
            result["target_date"] = pd.Timestamp("2024-02-29")
            result["price_date"] = pd.Timestamp("2024-02-29")

            report = build_daily_paper_report(result)

            self.assertIn("## 较上一快照变化", report)
            self.assertIn("- 上一快照日期：2024-01-31", report)
            self.assertIn("- 资产变化：1000.00", report)

    def test_report_includes_guard_issues(self) -> None:
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
                strategy="REPORT",
                target_weights={"AAA": 0.5},
                latest_prices={"AAA": 10.0},
                trade_date="2024-01-31",
            )
            result["target_date"] = pd.Timestamp("2024-01-31")
            result["price_date"] = pd.Timestamp("2024-01-31")
            result["guard_issues"] = [
                GuardIssue("WARNING", "stale_price_date", "价格日期可能过旧"),
            ]

            report = build_daily_paper_report(result)

            self.assertIn("## 运行检查", report)
            self.assertIn("stale_price_date", report)


if __name__ == "__main__":
    unittest.main()
