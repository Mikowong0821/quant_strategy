"""
纸面交易日报的风格暴露读取与摘要。

研究主流程会生成 output/factor_diagnostics/style_exposure.csv，本模块只负责
在日终纸面交易时取当前策略、当前运行日之前最近一期风格暴露，用于日报展示。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings


STYLE_EXPOSURE_DISPLAY_COLUMNS = [
    "date",
    "strategy",
    "style",
    "weighted_exposure",
    "abs_weighted_exposure",
    "score_coverage",
    "n_positions",
    "n_scored_positions",
]


def default_style_exposure_path(settings: Settings) -> Path:
    return settings.output_dir / "factor_diagnostics" / "style_exposure.csv"


def load_style_exposure(settings: Settings, path: Path | None = None) -> pd.DataFrame:
    """读取风格暴露表；不存在时返回空表。"""
    exposure_path = path or default_style_exposure_path(settings)
    if not exposure_path.exists():
        return pd.DataFrame(columns=STYLE_EXPOSURE_DISPLAY_COLUMNS)
    frame = pd.read_csv(exposure_path)
    if frame.empty:
        return pd.DataFrame(columns=STYLE_EXPOSURE_DISPLAY_COLUMNS)
    required = {"date", "strategy", "style", "weighted_exposure"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("风格暴露表缺少必要列: %s" % ", ".join(sorted(missing)))
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out[out["date"].notna()]
    for col in [
        "weighted_exposure",
        "abs_weighted_exposure",
        "score_coverage",
        "n_positions",
        "n_scored_positions",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def latest_style_exposure_for_strategy(
    exposure: pd.DataFrame,
    *,
    strategy: str,
    trade_date: Any,
) -> pd.DataFrame:
    """取不晚于 trade_date 的当前策略最近一期风格暴露。"""
    if exposure is None or exposure.empty:
        return pd.DataFrame(columns=STYLE_EXPOSURE_DISPLAY_COLUMNS)
    required = {"date", "strategy", "style", "weighted_exposure"}
    missing = required - set(exposure.columns)
    if missing:
        raise ValueError("风格暴露表缺少必要列: %s" % ", ".join(sorted(missing)))

    df = exposure.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[
        (df["strategy"].astype(str) == str(strategy))
        & (df["date"].notna())
        & (df["date"] <= pd.Timestamp(trade_date))
    ]
    if df.empty:
        return pd.DataFrame(columns=STYLE_EXPOSURE_DISPLAY_COLUMNS)
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].copy()
    for col in STYLE_EXPOSURE_DISPLAY_COLUMNS:
        if col not in latest.columns:
            latest[col] = pd.NA
    return latest[STYLE_EXPOSURE_DISPLAY_COLUMNS].sort_values(
        "abs_weighted_exposure",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)


def summarize_style_exposure_for_report(style_exposure: pd.DataFrame | None) -> tuple[str, str]:
    """把最近一期风格暴露压缩成日报摘要。"""
    if style_exposure is None or style_exposure.empty:
        return "UNKNOWN", "style_exposure_missing"
    df = style_exposure.dropna(subset=["weighted_exposure"]).copy()
    if df.empty:
        return "UNKNOWN", "style_exposure_empty"
    df["abs_weighted_exposure"] = df["weighted_exposure"].abs()
    dominant = df.sort_values("abs_weighted_exposure", ascending=False).iloc[0]
    date_s = pd.Timestamp(dominant["date"]).strftime("%Y-%m-%d")
    style = str(dominant["style"])
    exposure = float(dominant["weighted_exposure"])
    direction = "positive" if exposure > 0 else "negative" if exposure < 0 else "neutral"
    return style, "%s:%s:%.4f:%s" % (date_s, style, exposure, direction)
