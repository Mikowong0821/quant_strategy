"""统一风险限额表测试。"""
from __future__ import annotations

import unittest

import pandas as pd

from live.risk_limits import (
    check_risk_limits,
    default_risk_limits,
    summarize_risk_limit_checks,
)


class TestRiskLimits(unittest.TestCase):
    def test_check_risk_limits_flags_concentration_cash_and_industry(self) -> None:
        target = pd.DataFrame(
            [
                {"symbol": "AAA", "weight": 0.45},
                {"symbol": "BBB", "weight": 0.35},
                {"symbol": "CCC", "weight": 0.20},
            ]
        )
        industry = pd.DataFrame(
            [
                {"symbol": "AAA", "industry": "新能源"},
                {"symbol": "BBB", "industry": "新能源"},
                {"symbol": "CCC", "industry": "消费"},
            ]
        )
        risk_gate = pd.DataFrame(
            [
                {"symbol": "AAA", "gate_status": "BLOCK"},
                {"symbol": "BBB", "gate_status": "WATCH"},
            ]
        )
        order_checks = pd.DataFrame(
            [
                {"symbol": "AAA", "check_status": "PASS"},
                {"symbol": "BBB", "check_status": "BLOCK"},
                {"symbol": "CCC", "check_status": "PASS"},
            ]
        )

        checks = check_risk_limits(
            default_risk_limits(),
            target,
            trade_date="2026-08-10",
            industry=industry,
            risk_gate=risk_gate,
            order_checks=order_checks,
        )
        by_id = checks.set_index("limit_id")

        self.assertEqual(str(by_id.loc["max_single_position_weight", "status"]), "BLOCK")
        self.assertEqual(str(by_id.loc["max_industry_weight", "status"]), "BLOCK")
        self.assertEqual(str(by_id.loc["risk_gate_block_count", "status"]), "BLOCK")
        self.assertEqual(str(by_id.loc["order_block_count", "status"]), "BLOCK")
        self.assertEqual(str(by_id.loc["min_cash_weight", "status"]), "WATCH")
        self.assertAlmostEqual(float(by_id.loc["max_industry_weight", "observed_value"]), 0.80)

    def test_turnover_uses_current_weights_when_available(self) -> None:
        limits = default_risk_limits()
        target = {"AAA": 0.4, "BBB": 0.4, "CCC": 0.2}
        current = {"AAA": 0.0, "BBB": 0.4, "DDD": 0.4}

        checks = check_risk_limits(
            limits,
            target,
            trade_date="2026-08-10",
            current_weights=current,
        )
        row = checks.set_index("limit_id").loc["max_rebalance_turnover"]

        self.assertAlmostEqual(float(row["observed_value"]), 0.5)
        self.assertEqual(str(row["status"]), "PASS")

    def test_industry_map_supports_chinese_stock_pool_columns(self) -> None:
        target = {"000001.SZ": 0.3, "000002.SZ": 0.3, "600000.SH": 0.2}
        industry = pd.DataFrame(
            [
                {"股票代码": "000001.SZ", "分类": "银行"},
                {"股票代码": "000002.SZ", "分类": "银行"},
                {"股票代码": "600000.SH", "分类": "地产"},
            ]
        )

        checks = check_risk_limits(
            default_risk_limits(),
            target,
            trade_date="2026-08-10",
            industry=industry,
        )
        row = checks.set_index("limit_id").loc["max_industry_weight"]

        self.assertAlmostEqual(float(row["observed_value"]), 0.6)
        self.assertEqual(str(row["status"]), "BLOCK")

    def test_summarize_risk_limit_checks(self) -> None:
        checks = pd.DataFrame(
            [
                {"status": "PASS"},
                {"status": "WATCH"},
                {"status": "BLOCK"},
                {"status": "NA"},
            ]
        )

        status, detail = summarize_risk_limit_checks(checks)

        self.assertEqual(status, "BLOCK")
        self.assertIn("BLOCK=1", detail)
        self.assertIn("WATCH=1", detail)
        self.assertIn("NA=1", detail)


if __name__ == "__main__":
    unittest.main()
