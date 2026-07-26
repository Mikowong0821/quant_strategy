"""公告事件风险过滤测试。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from live.event_risk_filter import (
    build_event_risk_candidates,
    event_risk_candidates_to_blacklist,
    load_event_risk_candidates,
)


class TestEventRiskFilter(unittest.TestCase):
    def test_build_event_risk_candidates_detects_blacklist_and_watch(self) -> None:
        events = pd.DataFrame(
            [
                {
                    "event_date": "2024-01-10",
                    "symbol": "AAA",
                    "event_type": "announcement",
                    "title": "公司收到监管立案调查通知",
                    "event_score": -1.0,
                    "source": "unit_test",
                },
                {
                    "event_date": "2024-01-11",
                    "symbol": "BBB",
                    "event_type": "announcement",
                    "title": "收到交易所问询函",
                    "event_score": -0.4,
                    "source": "unit_test",
                },
                {
                    "event_date": "2024-01-12",
                    "symbol": "CCC",
                    "event_type": "announcement",
                    "title": "公司发布股份回购公告",
                    "event_score": 1.0,
                    "source": "unit_test",
                },
            ]
        )

        candidates = build_event_risk_candidates(
            events,
            as_of_date="2024-01-31",
            lookback_days=30,
            block_days=20,
            watch_days=10,
        )
        by_symbol = candidates.set_index("symbol")

        self.assertEqual(str(by_symbol.loc["AAA", "risk_action"]), "BLACKLIST")
        self.assertIn("block_keyword", str(by_symbol.loc["AAA", "risk_reason"]))
        self.assertEqual(pd.Timestamp(by_symbol.loc["AAA", "blacklist_until"]).strftime("%Y-%m-%d"), "2024-01-30")
        self.assertEqual(str(by_symbol.loc["BBB", "risk_action"]), "WATCH")
        self.assertNotIn("CCC", by_symbol.index)

    def test_candidates_to_blacklist_can_include_watch(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "event_date": "2024-01-10",
                    "symbol": "AAA",
                    "event_type": "announcement",
                    "title": "立案调查",
                    "event_score": -1.0,
                    "risk_level": "",
                    "source": "unit_test",
                    "risk_action": "BLACKLIST",
                    "risk_reason": "block_keyword:立案",
                    "blacklist_until": "2024-01-30",
                },
                {
                    "event_date": "2024-01-11",
                    "symbol": "BBB",
                    "event_type": "announcement",
                    "title": "问询函",
                    "event_score": -0.4,
                    "risk_level": "",
                    "source": "unit_test",
                    "risk_action": "WATCH",
                    "risk_reason": "watch_keyword:问询",
                    "blacklist_until": "2024-01-21",
                },
            ]
        )

        block_only = event_risk_candidates_to_blacklist(candidates)
        with_watch = event_risk_candidates_to_blacklist(candidates, include_watch=True)

        self.assertEqual(list(block_only["symbol"]), ["AAA"])
        self.assertEqual(set(with_watch["symbol"]), {"AAA", "BBB"})
        self.assertEqual(str(with_watch.set_index("symbol").loc["AAA", "severity"]), "HIGH")
        self.assertEqual(str(with_watch.set_index("symbol").loc["BBB", "severity"]), "WATCH")

    def test_load_event_risk_candidates_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "events.csv"
            pd.DataFrame(
                [
                    {
                        "公告日期": "2024-01-10",
                        "股票代码": "000001.SZ",
                        "公告标题": "公司涉及重大诉讼",
                    },
                ]
            ).to_csv(path, index=False)

            candidates = load_event_risk_candidates(path, as_of_date="2024-01-31")

            self.assertEqual(list(candidates["symbol"]), ["000001.SZ"])
            self.assertIn(str(candidates.loc[0, "risk_action"]), {"BLACKLIST", "WATCH"})


if __name__ == "__main__":
    unittest.main()
