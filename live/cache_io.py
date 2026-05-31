"""
将一次运行中的行情、因子面板与实验记录写入磁盘，便于复现与离线分析。
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

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


def _jsonable(value: Any) -> Any:
    """将 Path / Timestamp / dataclass 等转成稳定 JSON 值。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def settings_to_dict(settings: Settings) -> Dict[str, Any]:
    """导出 Settings 快照；Path 转字符串，便于 JSON 落盘。"""
    return _jsonable(settings)


def save_run_config(settings: Settings) -> Path:
    """将本次运行的 Settings 快照写入 output/cache/run_config.json。"""
    base = cache_dir(settings)
    base.mkdir(parents=True, exist_ok=True)
    path = base / "run_config.json"
    payload = settings_to_dict(settings)
    payload["written_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def save_performance_summary(
    settings: Settings,
    performance_by_name: Mapping[str, Mapping[str, Any]],
) -> Path:
    """
    将各策略绩效指标汇总为 output/performance_summary.csv。

    行为策略名，列包含 ann_return / ann_vol / sharpe / max_drawdown 等。
    """
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    path = settings.output_dir / "performance_summary.csv"
    rows: list[dict[str, Any]] = []
    for name, stats in performance_by_name.items():
        row: dict[str, Any] = {"strategy": name}
        row.update({str(k): v for k, v in stats.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("strategy").reset_index(drop=True)
    df.to_csv(path, index=False)
    return path


def _rebalance_log_to_frame(log: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in log:
        dt = rec.get("date")
        date_s = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
        picks = list(rec.get("picks") or [])
        weights = list(rec.get("weights") or [])
        for i, sym in enumerate(picks):
            rows.append(
                {
                    "date": date_s,
                    "symbol": sym,
                    "weight": float(weights[i]) if i < len(weights) else float("nan"),
                    "weighting": rec.get("weighting", ""),
                    "rank": i + 1,
                }
            )
    return pd.DataFrame(rows, columns=["date", "symbol", "weight", "weighting", "rank"])


def save_rebalance_logs(
    settings: Settings,
    meta_by_name: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Path]:
    """将各策略调仓日志拆成 CSV，写入 output/rebalance_logs/<strategy>.csv。"""
    base = settings.output_dir / "rebalance_logs"
    base.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for name, meta in meta_by_name.items():
        safe = str(name).replace("/", "_")
        path = base / ("%s.csv" % safe)
        log = list(meta.get("rebalance_log") or [])
        df = _rebalance_log_to_frame(log)
        df.to_csv(path, index=False)
        out[str(name)] = path
    return out


def save_turnover_logs(
    settings: Settings,
    turnover_by_name: Mapping[str, pd.DataFrame],
) -> Dict[str, Path]:
    """将各策略逐期换手表写入 output/turnover_logs/<strategy>.csv。"""
    base = settings.output_dir / "turnover_logs"
    base.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for name, frame in turnover_by_name.items():
        safe = str(name).replace("/", "_")
        path = base / ("%s.csv" % safe)
        frame.to_csv(path, index=False, date_format="%Y-%m-%d")
        out[str(name)] = path
    return out


def save_data_quality_reports(
    settings: Settings,
    reports: Mapping[str, pd.DataFrame],
) -> Dict[str, Path]:
    """将数据质量报告写入 output/data_quality/<name>.csv。"""
    base = settings.output_dir / "data_quality"
    base.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for name, frame in reports.items():
        safe = str(name).replace("/", "_")
        path = base / ("%s.csv" % safe)
        frame.to_csv(path, index=False, date_format="%Y-%m-%d")
        out[str(name)] = path
    return out
