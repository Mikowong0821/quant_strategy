"""风险总控日报。"""
from __future__ import annotations

import unittest

import pandas as pd

from live.paper_guard import GuardIssue
from live.risk_control_report import (
    build_risk_control_report,
    summarize_guard_issues_for_report,
    summarize_order_checks_for_report,
    summarize_risk_control_report,
)


class TestRiskControlReport(unittest.TestCase):
    def test_summarize_guard_issues_maps_warning_and_error(self) -> None:
        status, detail = summarize_guard_issues_for_report(
            [
                GuardIssue("WARNING", "stale_price_date", "价格过旧"),
            ]
        )
        self.assertEqual(status, "WATCH")
        self.assertIn("WARNING=1", detail)

        status, detail = summarize_guard_issues_for_report(
            [
                GuardIssue("ERROR", "empty_target", "目标为空"),
                GuardIssue("WARNING", "stale_price_date", "价格过旧"),
            ]
        )
        self.assertEqual(status, "BLOCK")
        self.assertIn("ERROR=1", detail)

    def test_summarize_order_checks_blocks_when_any_order_blocked(self) -> None:
        status, detail = summarize_order_checks_for_report(
            pd.DataFrame(
                [
                    {"check_status": "PASS", "check_reason": "ok"},
                    {"check_status": "BLOCK", "check_reason": "risk_blacklist"},
                ]
            )
        )

        self.assertEqual(status, "BLOCK")
        self.assertIn("BLOCK=1", detail)
        self.assertIn("risk_blacklist", detail)

    def test_build_risk_control_report_combines_modules(self) -> None:
        report = build_risk_control_report(
            trade_date="2026-08-13",
            guard_issues=[],
            risk_gate=pd.DataFrame([{"gate_status": "WATCH"}]),
            risk_blacklist=pd.DataFrame(columns=["symbol"]),
            drawdown_control=pd.DataFrame(
                [
                    {
                        "status": "PASS",
                        "drawdown_abs": 0.01,
                        "target_weight_scale": 1.0,
                        "action": "KEEP",
                        "triggered_rule_id": "",
                    }
                ]
            ),
            capacity_impact_summary=pd.DataFrame([{"status": "NA", "n_missing_liquidity": 1}]),
            order_checks=pd.DataFrame([{"check_status": "PASS"}]),
            risk_limit_checks=pd.DataFrame([{"status": "BLOCK"}]),
            stress_tests=pd.DataFrame([{"status": "PASS", "scenario_id": "market_down", "estimated_loss_pct": 0.01}]),
        )

        by_module = report.set_index("module")
        self.assertEqual(str(by_module.loc["组合风险限额", "status"]), "BLOCK")
        self.assertEqual(str(by_module.loc["统一风险门禁", "status"]), "WATCH")
        self.assertEqual(str(by_module.loc["容量与冲击成本", "status"]), "NA")

        status, detail = summarize_risk_control_report(report)
        self.assertEqual(status, "BLOCK")
        self.assertIn("BLOCK=1", detail)
        self.assertIn("WATCH=1", detail)
        self.assertIn("NA=1", detail)


if __name__ == "__main__":
    unittest.main()

