"""负面舆情过滤测试。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from live.negative_sentiment_filter import (
    build_negative_sentiment_candidates,
    load_sentiment_items,
    negative_sentiment_candidates_to_blacklist,
    normalize_sentiment_items,
)


class TestNegativeSentimentFilter(unittest.TestCase):
    def test_normalize_sentiment_items_supports_chinese_columns(self) -> None:
        items = pd.DataFrame(
            [
                {
                    "股票代码": "1",
                    "发布时间": "2024-01-10 10:00:00",
                    "标题": "公司被立案调查",
                    "摘要": "存在违规风险",
                    "来源": "unit_test",
                }
            ]
        )

        out = normalize_sentiment_items(items)

        self.assertEqual(list(out["symbol"]), ["000001.SZ"])
        self.assertLess(float(out.loc[0, "sentiment_score"]), 0.0)
        self.assertIn("立案", str(out.loc[0, "negative_keywords"]))

    def test_build_candidates_detects_blacklist_and_watch(self) -> None:
        items = pd.DataFrame(
            [
                {
                    "symbol": "000001.SZ",
                    "publish_time": "2024-01-10",
                    "title": "公司被立案调查",
                    "content": "",
                },
                {
                    "symbol": "000002.SZ",
                    "publish_time": "2024-01-11",
                    "title": "公司收到监管函",
                    "content": "",
                },
                {
                    "symbol": "000003.SZ",
                    "publish_time": "2024-01-12",
                    "title": "公司发布回购计划",
                    "content": "",
                },
            ]
        )

        candidates = build_negative_sentiment_candidates(
            items,
            as_of_date="2024-01-15",
            lookback_days=10,
            block_days=10,
            watch_days=5,
        )
        by_symbol = candidates.set_index("symbol")

        self.assertEqual(str(by_symbol.loc["000001.SZ", "risk_action"]), "BLACKLIST")
        self.assertEqual(str(by_symbol.loc["000002.SZ", "risk_action"]), "WATCH")
        self.assertNotIn("000003.SZ", by_symbol.index)
        self.assertEqual(pd.Timestamp(by_symbol.loc["000001.SZ", "blacklist_until"]).strftime("%Y-%m-%d"), "2024-01-20")

    def test_candidates_to_blacklist_defaults_to_block_only(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "publish_time": "2024-01-10",
                    "symbol": "000001.SZ",
                    "title": "立案",
                    "content": "",
                    "sentiment_score": -1.0,
                    "negative_keywords": "立案",
                    "source": "unit_test",
                    "risk_action": "BLACKLIST",
                    "risk_reason": "negative_sentiment_block",
                    "blacklist_until": "2024-01-20",
                },
                {
                    "publish_time": "2024-01-11",
                    "symbol": "000002.SZ",
                    "title": "监管函",
                    "content": "",
                    "sentiment_score": -0.4,
                    "negative_keywords": "监管函",
                    "source": "unit_test",
                    "risk_action": "WATCH",
                    "risk_reason": "negative_sentiment_watch",
                    "blacklist_until": "2024-01-16",
                },
            ]
        )

        block_only = negative_sentiment_candidates_to_blacklist(candidates)
        with_watch = negative_sentiment_candidates_to_blacklist(candidates, include_watch=True)

        self.assertEqual(list(block_only["symbol"]), ["000001.SZ"])
        self.assertEqual(set(with_watch["symbol"]), {"000001.SZ", "000002.SZ"})

    def test_load_sentiment_items_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sentiment.csv"
            pd.DataFrame(
                [
                    {
                        "symbol": "000001.SZ",
                        "publish_time": "2024-01-10",
                        "title": "公司被处罚",
                    }
                ]
            ).to_csv(path, index=False)

            loaded = load_sentiment_items(path)

        self.assertEqual(list(loaded["symbol"]), ["000001.SZ"])
        self.assertLess(float(loaded.loc[0, "sentiment_score"]), 0.0)


if __name__ == "__main__":
    unittest.main()
