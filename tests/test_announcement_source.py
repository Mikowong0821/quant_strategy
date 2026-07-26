"""真实公告数据源接入测试。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from live.announcement_source import (
    fetch_akshare_cninfo_announcement_events,
    fetch_tushare_announcement_events,
    normalize_akshare_cninfo_announcements,
    normalize_tushare_announcements,
    save_announcement_events,
)


class _FakeTusharePro:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def anns_d(self, **kwargs: str) -> pd.DataFrame:
        self.calls.append(kwargs)
        return pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20240110",
                    "ann_type": "问询函",
                    "title": "收到交易所问询函",
                },
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20240210",
                    "ann_type": "分红",
                    "title": "年度权益分派实施公告",
                },
                {
                    "ts_code": "000002.SZ",
                    "ann_date": "20240111",
                    "ann_type": "处罚",
                    "title": "收到行政处罚决定书",
                },
            ]
        )


class TestAnnouncementSource(unittest.TestCase):
    def test_normalize_tushare_announcements_maps_columns(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "ann_date": "20240110",
                    "ann_type": "问询函",
                    "title": "收到交易所问询函",
                }
            ]
        )

        events = normalize_tushare_announcements(raw)

        self.assertEqual(list(events["symbol"]), ["000001.SZ"])
        self.assertEqual(pd.Timestamp(events.loc[0, "event_date"]).strftime("%Y-%m-%d"), "2024-01-10")
        self.assertEqual(list(events["source"]), ["tushare"])
        self.assertLess(float(events.loc[0, "event_score"]), 0.0)

    def test_fetch_tushare_announcement_events_filters_date_and_symbol(self) -> None:
        pro = _FakeTusharePro()

        events = fetch_tushare_announcement_events(
            ["000001.SZ"],
            "2024-01-01",
            "2024-01-31",
            pro=pro,
        )

        self.assertEqual(list(events["symbol"]), ["000001.SZ"])
        self.assertEqual(pd.Timestamp(events.loc[0, "event_date"]).strftime("%Y-%m-%d"), "2024-01-10")
        self.assertEqual(pro.calls[0]["ts_code"], "000001.SZ")
        self.assertEqual(pro.calls[0]["start_date"], "20240101")

    def test_normalize_akshare_cninfo_announcements_maps_columns(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "代码": "601288",
                    "简称": "农业银行",
                    "公告标题": "农业银行关于分红派息实施公告",
                    "公告时间": "2024-01-10",
                    "公告链接": "https://example.test/ann",
                }
            ]
        )

        events = normalize_akshare_cninfo_announcements(raw)

        self.assertEqual(list(events["symbol"]), ["601288.SH"])
        self.assertEqual(pd.Timestamp(events.loc[0, "event_date"]).strftime("%Y-%m-%d"), "2024-01-10")
        self.assertEqual(list(events["source"]), ["akshare_cninfo"])
        self.assertGreater(float(events.loc[0, "event_score"]), 0.0)

    def test_fetch_akshare_cninfo_announcement_events_uses_plain_stock_code(self) -> None:
        calls: list[dict[str, str]] = []

        def fake_fetcher(**kwargs: str) -> pd.DataFrame:
            calls.append(kwargs)
            return pd.DataFrame(
                [
                    {
                        "代码": kwargs["symbol"],
                        "简称": "农业银行",
                        "公告标题": "收到监管问询函",
                        "公告时间": "2024-01-10",
                    }
                ]
            )

        events = fetch_akshare_cninfo_announcement_events(
            ["601288.SH"],
            "2024-01-01",
            "2024-01-31",
            fetcher=fake_fetcher,
        )

        self.assertEqual(calls[0]["symbol"], "601288")
        self.assertEqual(calls[0]["start_date"], "20240101")
        self.assertEqual(list(events["symbol"]), ["601288.SH"])

    def test_save_announcement_events_writes_standard_table(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "announcement_events.csv"
            saved = save_announcement_events(
                pd.DataFrame(
                    [
                        {
                            "ts_code": "000001.SZ",
                            "ann_date": "20240110",
                            "title": "收到监管问询函",
                        }
                    ]
                ),
                path,
            )

            loaded = pd.read_csv(saved)

        self.assertTrue(path.name.endswith(".csv"))
        self.assertIn("event_date", loaded.columns)
        self.assertIn("symbol", loaded.columns)


if __name__ == "__main__":
    unittest.main()
