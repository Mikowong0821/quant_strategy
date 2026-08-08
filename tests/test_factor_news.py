"""新闻 / 舆情日频因子测试。"""
from __future__ import annotations

import unittest

import pandas as pd

from factors.factor_news import (
    NEWS_HEAT_7D,
    NEWS_NEGATIVE_COUNT_7D,
    NEWS_NEGATIVE_RISK_SCORE,
    NEWS_SENTIMENT_DECAY,
    calc_news_sentiment_factors,
)


class TestNewsFactors(unittest.TestCase):
    def test_calc_news_sentiment_factors(self) -> None:
        prices_long = pd.DataFrame(
            {
                "trade_date": pd.date_range("2026-07-10", periods=5, freq="D").tolist() * 2,
                "ts_code": ["000001.SZ"] * 5 + ["000002.SZ"] * 5,
                "close": [10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
            }
        )
        news = pd.DataFrame(
            [
                {
                    "symbol": "000001.SZ",
                    "publish_time": "2026-07-10 10:00:00",
                    "title": "公司被立案调查",
                    "content": "",
                },
                {
                    "symbol": "000002.SZ",
                    "publish_time": "2026-07-11",
                    "title": "普通新闻",
                    "content": "",
                    "sentiment_score": 0.2,
                },
            ]
        )

        panel = calc_news_sentiment_factors(news, prices_long, effective_days=3, lookback_days=3)

        self.assertEqual(set(panel.columns), {NEWS_SENTIMENT_DECAY, NEWS_NEGATIVE_RISK_SCORE, NEWS_NEGATIVE_COUNT_7D, NEWS_HEAT_7D})
        self.assertLess(float(panel.loc[(pd.Timestamp("2026-07-10"), "000001.SZ"), NEWS_SENTIMENT_DECAY]), 0.0)
        self.assertGreater(float(panel.loc[(pd.Timestamp("2026-07-10"), "000001.SZ"), NEWS_NEGATIVE_RISK_SCORE]), 0.0)
        self.assertEqual(float(panel.loc[(pd.Timestamp("2026-07-10"), "000001.SZ"), NEWS_NEGATIVE_COUNT_7D]), 1.0)
        self.assertEqual(float(panel.loc[(pd.Timestamp("2026-07-11"), "000002.SZ"), NEWS_HEAT_7D]), 1.0)
        self.assertEqual(float(panel.loc[(pd.Timestamp("2026-07-11"), "000002.SZ"), NEWS_NEGATIVE_COUNT_7D]), 0.0)

    def test_empty_news_returns_zero_panel(self) -> None:
        prices_long = pd.DataFrame(
            {
                "trade_date": ["2026-07-10"],
                "ts_code": ["000001.SZ"],
                "close": [10.0],
            }
        )

        panel = calc_news_sentiment_factors(pd.DataFrame(), prices_long)

        self.assertEqual(float(panel.iloc[0][NEWS_HEAT_7D]), 0.0)


if __name__ == "__main__":
    unittest.main()
