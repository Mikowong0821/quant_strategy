#!/usr/bin/env python3
"""每日纸面交易命令行入口。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live.daily_paper_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
