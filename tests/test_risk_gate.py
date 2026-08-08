"""统一风险门禁测试。"""
from __future__ import annotations

import unittest

import pandas as pd

from live.risk_gate import build_unified_risk_gate, risk_gate_to_blacklist, summarize_risk_gate_for_report


class TestRiskGate(unittest.TestCase):
    def test_build_unified_risk_gate_merges_sources_with_block_priority(self) -> None:
        pool = pd.DataFrame(
            [
                {"symbol": "000001.SZ", "name": "平安银行"},
                {"symbol": "000002.SZ", "name": "万科A"},
                {"symbol": "000003.SZ", "name": "测试股票"},
            ]
        )
        manual = pd.DataFrame(
            [
                {
                    "symbol": "000003.SZ",
                    "severity": "WATCH",
                    "reason": "manual_watch",
                    "source": "manual",
                    "active": True,
                    "created_at": "2026-07-01",
                    "expires_at": "2026-07-31",
                }
            ]
        )
        events = pd.DataFrame(
            [
                {
                    "event_date": "2026-07-20",
                    "symbol": "000001.SZ",
                    "event_type": "announcement",
                    "title": "收到交易所问询函",
                    "event_score": -0.4,
                    "risk_level": "",
                    "source": "unit_event",
                    "risk_action": "WATCH",
                    "risk_reason": "watch_keyword:问询",
                    "blacklist_until": "2026-07-30",
                }
            ]
        )
        sentiment = pd.DataFrame(
            [
                {
                    "publish_time": "2026-07-24",
                    "symbol": "000001.SZ",
                    "title": "公司被立案调查",
                    "content": "",
                    "sentiment_score": -1.0,
                    "negative_keywords": "立案",
                    "source": "unit_news",
                    "url": "",
                    "risk_action": "BLACKLIST",
                    "risk_reason": "negative_sentiment_block",
                    "blacklist_until": "2026-08-03",
                }
            ]
        )

        gate, details = build_unified_risk_gate(
            trade_date="2026-07-24",
            symbols=pool,
            manual_blacklist=manual,
            event_candidates=events,
            sentiment_candidates=sentiment,
        )
        by_symbol = gate.set_index("symbol")

        self.assertEqual(str(by_symbol.loc["000001.SZ", "gate_status"]), "BLOCK")
        self.assertEqual(int(by_symbol.loc["000001.SZ", "block_count"]), 1)
        self.assertEqual(int(by_symbol.loc["000001.SZ", "watch_count"]), 1)
        self.assertEqual(str(by_symbol.loc["000002.SZ", "gate_status"]), "PASS")
        self.assertEqual(str(by_symbol.loc["000003.SZ", "gate_status"]), "WATCH")
        self.assertEqual(len(details), 3)

    def test_risk_gate_to_blacklist_defaults_to_block_only(self) -> None:
        gate = pd.DataFrame(
            [
                {
                    "trade_date": "2026-07-24",
                    "symbol": "000001.SZ",
                    "name": "平安银行",
                    "gate_status": "BLOCK",
                    "severity": "HIGH",
                    "reason": "negative_sentiment_block",
                    "expires_at": "2026-08-03",
                },
                {
                    "trade_date": "2026-07-24",
                    "symbol": "000002.SZ",
                    "name": "万科A",
                    "gate_status": "WATCH",
                    "severity": "WATCH",
                    "reason": "manual_watch",
                    "expires_at": "2026-07-31",
                },
            ]
        )

        block_only = risk_gate_to_blacklist(gate)
        with_watch = risk_gate_to_blacklist(gate, include_watch=True)

        self.assertEqual(list(block_only["symbol"]), ["000001.SZ"])
        self.assertEqual(set(with_watch["symbol"]), {"000001.SZ", "000002.SZ"})
        self.assertEqual(str(with_watch.set_index("symbol").loc["000001.SZ", "severity"]), "HIGH")
        self.assertEqual(str(with_watch.set_index("symbol").loc["000002.SZ", "severity"]), "WATCH")

    def test_summarize_risk_gate_for_report(self) -> None:
        gate = pd.DataFrame(
            [
                {"symbol": "AAA", "gate_status": "PASS"},
                {"symbol": "BBB", "gate_status": "WATCH"},
                {"symbol": "CCC", "gate_status": "BLOCK"},
            ]
        )

        status, detail = summarize_risk_gate_for_report(gate)

        self.assertEqual(status, "BLOCK")
        self.assertIn("BLOCK=1", detail)
        self.assertIn("WATCH=1", detail)


if __name__ == "__main__":
    unittest.main()
