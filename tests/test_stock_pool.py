import tempfile
import unittest
from pathlib import Path

import pandas as pd

from config import get_settings
from live.stock_pool import (
    active_universe_from_report,
    build_stock_pool_filter_report,
    load_stock_pool,
    load_stock_pool_frame,
    normalize_ts_code,
    save_universe_files,
)


class StockPoolTests(unittest.TestCase):
    def test_normalize_ts_code(self):
        cases = {
            "1": "000001.SZ",
            "000001": "000001.SZ",
            "600519": "600519.SH",
            "001309.SZ": "001309.SZ",
            "688981.sh": "688981.SH",
            " 301308 sz ": "301308.SZ",
            None: "",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_ts_code(raw), expected)

    def test_load_stock_pool_from_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.csv"
            pd.DataFrame(
                {
                    "股票代码": ["1", "600519", "001309.SZ", "001309.SZ", None],
                    "股票简称": ["平安银行", "贵州茅台", "德明利", "德明利", ""],
                    "是否启用": [1, 1, 1, 1, 1],
                }
            ).to_csv(path, index=False)

            self.assertEqual(
                load_stock_pool(path),
                ["000001.SZ", "001309.SZ", "600519.SH"],
            )

    def test_load_stock_pool_from_excel(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.xlsx"
            pd.DataFrame({"股票代码": ["688981.SH", "301308"]}).to_excel(path, index=False)

            self.assertEqual(load_stock_pool(path), ["301308.SZ", "688981.SH"])

    def test_missing_code_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.csv"
            pd.DataFrame({"code": ["000001.SZ"]}).to_csv(path, index=False)

            with self.assertRaises(ValueError):
                load_stock_pool(path)

    def test_load_stock_pool_frame_keeps_metadata_and_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pool.csv"
            pd.DataFrame(
                {
                    "股票代码": ["000001.SZ", "600519.SH"],
                    "股票简称": ["平安银行", "贵州茅台"],
                    "主题": ["银行", "白酒"],
                    "子行业": ["股份行", "高端白酒"],
                    "是否启用": [1, 0],
                }
            ).to_csv(path, index=False)

            frame = load_stock_pool_frame(path)

            self.assertEqual(frame.loc[0, "symbol"], "000001.SZ")
            self.assertEqual(frame.loc[0, "theme"], "银行")
            self.assertTrue(bool(frame.loc[0, "enabled"]))
            self.assertFalse(bool(frame.loc[1, "enabled"]))

    def test_build_stock_pool_filter_report_and_active_universe(self):
        pool = pd.DataFrame(
            {
                "symbol": ["000001.SZ", "600519.SH", "001309.SZ", "BAD"],
                "name": ["平安银行", "贵州茅台", "德明利", "坏代码"],
                "theme": ["银行", "白酒", "存储", "测试"],
                "sub_industry": ["股份行", "高端白酒", "存储", ""],
                "enabled": [True, False, True, True],
            }
        )
        days = pd.bdate_range("2024-01-01", periods=25)
        rows = []
        for dt in days:
            rows.append(
                {
                    "trade_date": dt,
                    "ts_code": "000001.SZ",
                    "close": 10.0,
                    "volume": 1000.0,
                    "amount": 10000.0,
                }
            )
            rows.append(
                {
                    "trade_date": dt,
                    "ts_code": "600519.SH",
                    "close": 100.0,
                    "volume": 1000.0,
                    "amount": 100000.0,
                }
            )
        trade_status = pd.DataFrame(
            {
                "symbol": ["000001.SZ", "001309.SZ"],
                "is_suspended": [False, False],
                "is_limit_up": [False, True],
                "is_limit_down": [False, False],
            }
        )

        report = build_stock_pool_filter_report(
            pool,
            price_data=pd.DataFrame(rows),
            trade_status=trade_status,
            as_of_date=days[-1],
            min_price_coverage=0.8,
            min_history_days=20,
        )
        active = active_universe_from_report(report)

        self.assertEqual(active["symbol"].tolist(), ["000001.SZ"])
        reasons = dict(zip(report["symbol"], report["exclude_reason"]))
        self.assertIn("disabled_by_pool", reasons["600519.SH"])
        self.assertIn("missing_price_history", reasons["001309.SZ"])
        self.assertIn("limit_up", reasons["001309.SZ"])
        self.assertIn("invalid_symbol", reasons["BAD"])

    def test_save_universe_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = get_settings()
            settings = settings.__class__(**{**settings.__dict__, "output_dir": Path(tmp) / "output"})
            report = pd.DataFrame(
                {
                    "date": ["2024-01-31"],
                    "symbol": ["000001.SZ"],
                    "name": ["平安银行"],
                    "theme": ["银行"],
                    "sub_industry": ["股份行"],
                    "enabled": [True],
                    "active": [True],
                    "exclude_reason": [""],
                    "latest_price": [10.0],
                    "avg_volume": [1000.0],
                    "avg_amount": [10000.0],
                }
            )

            paths = save_universe_files(settings, report, trade_date="2024-01-31")

            self.assertTrue(paths["filter_report"].is_file())
            self.assertTrue(paths["active_universe"].is_file())


if __name__ == "__main__":
    unittest.main()
