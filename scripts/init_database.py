"""Initialize the local SQLite database schema."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from storage.database import CORE_TABLES, default_database_path, initialize_database


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="初始化本地 SQLite 数据库表结构。")
    parser.add_argument(
        "--database",
        type=Path,
        default=default_database_path(settings),
        help="SQLite 数据库路径，默认 data/quant_strategy.db 或 QUANT_DATABASE_PATH",
    )
    args = parser.parse_args()

    path = initialize_database(args.database)
    print("database=%s" % path)
    print("tables=%s" % ",".join(CORE_TABLES))


if __name__ == "__main__":
    main()
