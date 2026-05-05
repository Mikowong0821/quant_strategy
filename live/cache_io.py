"""
将一次运行中的行情与因子面板写入磁盘（output/cache），便于复现与离线分析。
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd

from config import Settings


def cache_dir(settings: Settings) -> Path:
    return settings.output_dir / "cache"


def save_run_cache(
    settings: Settings,
    long_df: pd.DataFrame,
    prices_wide: pd.DataFrame,
    panel: pd.DataFrame,
) -> Dict[str, Path]:
    """
    写入：
    - prices_long.csv：日频 OHLCV 长表
    - prices_wide_close.csv：收盘价宽表（索引为日期）
    - factor_panel.csv：因子面板（date, symbol 展开为列）
    - run_meta.txt：区间与写入时间等元数据
    """
    base = cache_dir(settings)
    base.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}

    p_long = base / "prices_long.csv"
    long_df.to_csv(p_long, index=False)
    out["prices_long"] = p_long

    p_wide = base / "prices_wide_close.csv"
    prices_wide.to_csv(p_wide, date_format="%Y-%m-%d")
    out["prices_wide_close"] = p_wide

    p_panel = base / "factor_panel.csv"
    panel_flat = panel.reset_index()
    panel_flat.to_csv(p_panel, index=False, date_format="%Y-%m-%d")
    out["factor_panel"] = p_panel

    meta = base / "run_meta.txt"
    meta.write_text(
        "written_utc=%s\nbacktest_start=%s\nbacktest_end=%s\nprice_col=%s\n"
        % (
            datetime.now(timezone.utc).isoformat(),
            settings.backtest_start,
            settings.backtest_end,
            settings.price_col,
        ),
        encoding="utf-8",
    )
    out["run_meta"] = meta
    return out
