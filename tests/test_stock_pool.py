import tempfile
import unittest
from pathlib import Path

import pandas as pd

from live.stock_pool import load_stock_pool, normalize_ts_code


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


if __name__ == "__main__":
    unittest.main()
