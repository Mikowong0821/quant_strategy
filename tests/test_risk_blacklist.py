"""风险黑名单标准化与有效期过滤。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from live.risk_blacklist import active_risk_blacklist, load_risk_blacklist, risk_blacklist_map


class TestRiskBlacklist(unittest.TestCase):
    def test_active_risk_blacklist_respects_active_and_dates(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "股票代码": "000001.SZ",
                    "股票名称": "平安银行",
                    "风险等级": "HIGH",
                    "原因": "major_event_pending",
                    "是否启用": True,
                    "生效日期": "2024-01-01",
                    "失效日期": "2024-12-31",
                },
                {
                    "股票代码": "000002.SZ",
                    "股票名称": "万科A",
                    "风险等级": "WATCH",
                    "原因": "expired",
                    "是否启用": True,
                    "生效日期": "2023-01-01",
                    "失效日期": "2023-12-31",
                },
                {
                    "股票代码": "000003.SZ",
                    "股票名称": "关闭项",
                    "风险等级": "HIGH",
                    "原因": "disabled",
                    "是否启用": False,
                },
            ]
        )

        active = active_risk_blacklist(frame, trade_date="2024-06-30")

        self.assertEqual(list(active["symbol"]), ["000001.SZ"])
        self.assertEqual(str(active.loc[0, "reason"]), "major_event_pending")

    def test_load_csv_and_map(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "risk_blacklist.csv"
            pd.DataFrame(
                [
                    {"symbol": "600000.SH", "reason": "manual_review", "severity": "high"},
                ]
            ).to_csv(path, index=False)

            loaded = load_risk_blacklist(path)
            mapped = risk_blacklist_map(loaded)

            self.assertIn("600000.SH", mapped)
            self.assertEqual(mapped["600000.SH"]["severity"], "HIGH")
            self.assertEqual(mapped["600000.SH"]["reason"], "manual_review")


if __name__ == "__main__":
    unittest.main()
