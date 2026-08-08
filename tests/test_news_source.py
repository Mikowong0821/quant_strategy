"""新闻 / 舆情数据源标准化测试。"""
from __future__ import annotations

import unittest

import pandas as pd

from live.news_source import (
    fetch_akshare_stock_news_items,
    merge_news_items,
    normalize_akshare_stock_news,
    normalize_tushare_news,
)


class TestNewsSource(unittest.TestCase):
    def test_normalize_akshare_stock_news(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "关键词": "600519",
                    "新闻标题": "公司被立案调查",
                    "新闻内容": "存在违规风险",
                    "发布时间": "2026-07-10 10:00:00",
                    "文章来源": "unit",
                    "新闻链接": "https://example.com/a",
                }
            ]
        )

        out = normalize_akshare_stock_news(raw, symbol="600519.SH")

        self.assertEqual(list(out["symbol"]), ["600519.SH"])
        self.assertEqual(str(out.loc[0, "source"]), "unit")
        self.assertEqual(str(out.loc[0, "url"]), "https://example.com/a")
        self.assertIn("立案", str(out.loc[0, "negative_keywords"]))
        self.assertLess(float(out.loc[0, "sentiment_score"]), 0.0)

    def test_normalize_tushare_news(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "datetime": "2026-07-10 09:30:00",
                    "title": "收到监管函",
                    "content": "",
                    "src": "tushare",
                    "url": "https://example.com/b",
                }
            ]
        )

        out = normalize_tushare_news(raw)

        self.assertEqual(list(out["symbol"]), ["000001.SZ"])
        self.assertEqual(str(out.loc[0, "source"]), "tushare")
        self.assertIn("监管函", str(out.loc[0, "negative_keywords"]))

    def test_fetch_akshare_stock_news_items_uses_provider_schema(self) -> None:
        def fake_fetcher(symbol: str) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "关键词": symbol,
                        "新闻标题": "业绩预亏",
                        "新闻内容": "",
                        "发布时间": "2026-07-10",
                        "文章来源": "fake",
                        "新闻链接": "https://example.com/c",
                    }
                ]
            )

        out = fetch_akshare_stock_news_items(["000001.SZ"], fetcher=fake_fetcher, sleep_seconds=0.0)

        self.assertEqual(list(out["symbol"]), ["000001.SZ"])
        self.assertEqual(str(out.loc[0, "source"]), "fake")

    def test_merge_news_items_deduplicates(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "symbol": "000001.SZ",
                    "publish_time": "2026-07-10",
                    "title": "处罚",
                    "content": "",
                    "source": "unit",
                    "url": "https://example.com/a",
                },
                {
                    "symbol": "000001.SZ",
                    "publish_time": "2026-07-10",
                    "title": "处罚",
                    "content": "",
                    "source": "unit",
                    "url": "https://example.com/a",
                },
            ]
        )

        out = merge_news_items(frame)

        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
