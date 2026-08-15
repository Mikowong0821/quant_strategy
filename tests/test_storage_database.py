"""SQLite 存储层表结构测试。"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from storage.database import (
    CORE_TABLES,
    get_table_columns,
    initialize_database,
    list_database_tables,
    missing_core_tables,
)


class StorageDatabaseTests(unittest.TestCase):
    def test_initialize_database_creates_core_tables_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "quant_strategy.db"

            initialize_database(db_path)
            initialize_database(db_path)

            tables = set(list_database_tables(db_path))
            self.assertTrue(set(CORE_TABLES).issubset(tables))
            self.assertEqual(missing_core_tables(db_path), [])

            self.assertEqual(
                get_table_columns(db_path, "factor_panel_daily")[:5],
                ["trade_date", "ts_code", "factor_name", "factor_value", "factor_version"],
            )
            self.assertIn("amount", get_table_columns(db_path, "prices_daily"))
            self.assertIn("active", get_table_columns(db_path, "universe_snapshot"))

    def test_prices_daily_primary_key_blocks_duplicate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "quant_strategy.db"
            initialize_database(db_path)

            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    """
                    INSERT INTO prices_daily(trade_date, ts_code, open, high, low, close, volume, amount)
                    VALUES ('2026-08-14', '600519.SH', 100, 101, 99, 100.5, 1000, 100500)
                    """
                )
                conn.commit()
                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute(
                        """
                        INSERT INTO prices_daily(trade_date, ts_code, open, high, low, close, volume, amount)
                        VALUES ('2026-08-14', '600519.SH', 100, 101, 99, 100.5, 1000, 100500)
                        """
                    )
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()

