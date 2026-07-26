"""公告事件因子测试。"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from config import get_settings
from factors.factor_events import (
    ANNOUNCEMENT_EVENT_SCORE,
    calc_announcement_event_score,
    load_announcement_events,
)
from factors.panel_builder import build_four_factor_panel


class TestAnnouncementEventFactor(unittest.TestCase):
    def test_event_score_uses_keyword_and_decay(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=6)
        prices_long = pd.DataFrame(
            {
                "trade_date": list(days) * 2,
                "ts_code": ["AAA"] * len(days) + ["BBB"] * len(days),
                "close": [10.0] * (len(days) * 2),
            }
        )
        events = pd.DataFrame(
            [
                {
                    "公告日期": "2024-01-03",
                    "股票代码": "AAA",
                    "公告标题": "公司发布股份回购公告",
                },
                {
                    "公告日期": "2024-01-04",
                    "股票代码": "BBB",
                    "公告标题": "收到监管问询函",
                },
            ]
        )

        score = calc_announcement_event_score(events, prices_long, effective_days=3)

        self.assertEqual(score.name, ANNOUNCEMENT_EVENT_SCORE)
        self.assertAlmostEqual(float(score.loc[(days[2], "AAA")]), 1.0)
        self.assertGreater(float(score.loc[(days[3], "AAA")]), 0.0)
        self.assertLess(float(score.loc[(days[3], "BBB")]), 0.0)
        self.assertAlmostEqual(float(score.loc[(days[0], "AAA")]), 0.0)

    def test_load_events_and_build_panel(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=30)
        rows = []
        for sym in ["AAA", "BBB"]:
            for i, dt in enumerate(days):
                rows.append(
                    {
                        "trade_date": dt,
                        "ts_code": sym,
                        "open": 10.0 + i,
                        "high": 10.5 + i,
                        "low": 9.5 + i,
                        "close": 10.0 + i,
                        "volume": 100.0 + i,
                    }
                )
        long_df = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as td:
            event_path = Path(td) / "events.csv"
            pd.DataFrame(
                [
                    {
                        "event_date": "2024-01-05",
                        "symbol": "AAA",
                        "event_type": "buyback",
                        "event_score": 1.2,
                        "source": "unit_test",
                    }
                ]
            ).to_csv(event_path, index=False)
            loaded = load_announcement_events(event_path)
            settings = replace(
                get_settings(),
                announcement_event_path=event_path,
                announcement_event_effective_days=5,
            )
            with patch("factors.panel_builder.fetch_fina_indicator_panel", return_value=pd.DataFrame()):
                panel = build_four_factor_panel(long_df, long_df, settings)

        self.assertEqual(list(loaded["symbol"]), ["AAA"])
        self.assertIn(ANNOUNCEMENT_EVENT_SCORE, panel.columns)
        self.assertGreater(int(panel[ANNOUNCEMENT_EVENT_SCORE].abs().gt(0.0).sum()), 0)


if __name__ == "__main__":
    unittest.main()
