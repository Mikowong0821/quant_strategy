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
    panel_zscore: pd.DataFrame | None = None,
) -> Dict[str, Path]:
    """
    写入：
    - prices_long.csv：日频 OHLCV 长表
    - prices_wide_close.csv：收盘价宽表（索引为日期）
    - factor_panel.csv：原始因子面板（date, symbol 展开为列）
    - factor_panel_zscore.csv：可选，横截面标准化因子面板
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

    if panel_zscore is not None:
        p_panel_z = base / "factor_panel_zscore.csv"
        panel_zscore.reset_index().to_csv(p_panel_z, index=False, date_format="%Y-%m-%d")
        out["factor_panel_zscore"] = p_panel_z

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
        selected = [str(x) for x in list(rec.get("selected_picks") or picks)]
        selected_rank = {sym: i + 1 for i, sym in enumerate(selected)}
        weights = list(rec.get("weights") or [])
        for i, sym in enumerate(picks):
            ss = str(sym)
            rows.append(
                {
                    "date": date_s,
                    "symbol": ss,
                    "weight": float(weights[i]) if i < len(weights) else float("nan"),
                    "weighting": rec.get("weighting", ""),
                    "rank": i + 1,
                    "selected": ss in selected_rank,
                    "selected_rank": selected_rank.get(ss, ""),
                    "target_turnover": rec.get("target_turnover", ""),
                    "turnover_capped": rec.get("turnover_capped", ""),
                    "turnover_scale": rec.get("turnover_scale", ""),
                    "n_candidates_before_liquidity": rec.get("n_candidates_before_liquidity", ""),
                    "n_candidates_after_liquidity": rec.get("n_candidates_after_liquidity", ""),
                    "liquidity_filter_enabled": rec.get("liquidity_filter_enabled", ""),
                    "liquidity_lookback_days": rec.get("liquidity_lookback_days", ""),
                    "min_avg_volume": rec.get("min_avg_volume", ""),
                    "min_avg_amount": rec.get("min_avg_amount", ""),
                    "liquidity_missing_data": rec.get("liquidity_missing_data", ""),
                    "n_trade_blocked": rec.get("n_trade_blocked", ""),
                    "trade_status_filter_enabled": rec.get("trade_status_filter_enabled", ""),
                    "trade_status_missing_data": rec.get("trade_status_missing_data", ""),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "symbol",
            "weight",
            "weighting",
            "rank",
            "selected",
            "selected_rank",
            "target_turnover",
            "turnover_capped",
            "turnover_scale",
            "n_candidates_before_liquidity",
            "n_candidates_after_liquidity",
            "liquidity_filter_enabled",
            "liquidity_lookback_days",
            "min_avg_volume",
            "min_avg_amount",
            "liquidity_missing_data",
            "n_trade_blocked",
            "trade_status_filter_enabled",
            "trade_status_missing_data",
        ],
    )


def _decision_log_to_frame(log: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in log:
        dt = rec.get("date")
        date_s = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
        rows.append(
            {
                "date": date_s,
                "symbol": rec.get("symbol", ""),
                "factor_score": rec.get("factor_score", ""),
                "factor_rank": rec.get("factor_rank", ""),
                "passed_liquidity_filter": rec.get("passed_liquidity_filter", ""),
                "selected_by_signal": rec.get("selected_by_signal", ""),
                "selected_rank": rec.get("selected_rank", ""),
                "previous_weight": rec.get("previous_weight", ""),
                "raw_target_weight": rec.get("raw_target_weight", ""),
                "final_target_weight": rec.get("final_target_weight", ""),
                "weighting": rec.get("weighting", ""),
                "turnover_capped": rec.get("turnover_capped", ""),
                "is_suspended": rec.get("is_suspended", ""),
                "is_limit_up": rec.get("is_limit_up", ""),
                "is_limit_down": rec.get("is_limit_down", ""),
                "trade_blocked": rec.get("trade_blocked", ""),
                "trade_block_reason": rec.get("trade_block_reason", ""),
                "action": rec.get("action", ""),
                "decision_reason": rec.get("decision_reason", ""),
                "n_candidates_before_liquidity": rec.get("n_candidates_before_liquidity", ""),
                "n_candidates_after_liquidity": rec.get("n_candidates_after_liquidity", ""),
                "liquidity_filter_enabled": rec.get("liquidity_filter_enabled", ""),
                "liquidity_lookback_days": rec.get("liquidity_lookback_days", ""),
                "min_avg_volume": rec.get("min_avg_volume", ""),
                "min_avg_amount": rec.get("min_avg_amount", ""),
                "liquidity_missing_data": rec.get("liquidity_missing_data", ""),
                "trade_status_filter_enabled": rec.get("trade_status_filter_enabled", ""),
                "trade_status_missing_data": rec.get("trade_status_missing_data", ""),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "symbol",
            "factor_score",
            "factor_rank",
            "passed_liquidity_filter",
            "selected_by_signal",
            "selected_rank",
            "previous_weight",
            "raw_target_weight",
            "final_target_weight",
            "weighting",
            "turnover_capped",
            "is_suspended",
            "is_limit_up",
            "is_limit_down",
            "trade_blocked",
            "trade_block_reason",
            "action",
            "decision_reason",
            "n_candidates_before_liquidity",
            "n_candidates_after_liquidity",
            "liquidity_filter_enabled",
            "liquidity_lookback_days",
            "min_avg_volume",
            "min_avg_amount",
            "liquidity_missing_data",
            "trade_status_filter_enabled",
            "trade_status_missing_data",
        ],
    )


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


def save_decision_logs(
    settings: Settings,
    meta_by_name: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Path]:
    """将各策略逐标的调仓决策审计日志写入 output/decision_logs/<strategy>.csv。"""
    base = settings.output_dir / "decision_logs"
    base.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for name, meta in meta_by_name.items():
        safe = str(name).replace("/", "_")
        path = base / ("%s.csv" % safe)
        log = list(meta.get("decision_log") or [])
        df = _decision_log_to_frame(log)
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


def save_risk_exposure_logs(
    settings: Settings,
    concentration_by_name: Mapping[str, pd.DataFrame],
) -> Dict[str, Path]:
    """将各策略逐期集中度表写入 output/risk_exposure/concentration_logs/<strategy>.csv。"""
    base = settings.output_dir / "risk_exposure" / "concentration_logs"
    base.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for name, frame in concentration_by_name.items():
        safe = str(name).replace("/", "_")
        path = base / ("%s.csv" % safe)
        frame.to_csv(path, index=False, date_format="%Y-%m-%d")
        out[str(name)] = path
    return out


def save_risk_exposure_summary(
    settings: Settings,
    concentration_summary_by_name: Mapping[str, Mapping[str, Any]],
) -> Path:
    """将各策略集中度汇总写入 output/risk_exposure/concentration_summary.csv。"""
    base = settings.output_dir / "risk_exposure"
    base.mkdir(parents=True, exist_ok=True)
    path = base / "concentration_summary.csv"
    rows: list[dict[str, Any]] = []
    for name, stats in concentration_summary_by_name.items():
        row: dict[str, Any] = {"strategy": name}
        row.update({str(k): v for k, v in stats.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("strategy").reset_index(drop=True)
    df.to_csv(path, index=False)
    return path


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


def save_factor_diagnostics(
    settings: Settings,
    long_excess_summary: pd.DataFrame,
    group_return_detail: pd.DataFrame | None = None,
    group_return_summary: pd.DataFrame | None = None,
    factor_weight_summary: pd.DataFrame | None = None,
    factor_weight_train_summary: pd.DataFrame | None = None,
    rolling_factor_weight_log: pd.DataFrame | None = None,
) -> Dict[str, Path]:
    """保存因子诊断表：Top-K 多头超额、分组收益、综合评分、训练段权重与滚动权重日志。"""
    base = settings.output_dir / "factor_diagnostics"
    base.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}

    path = base / "long_excess_summary.csv"
    long_excess_summary.to_csv(path, index=False)
    out["long_excess_summary"] = path

    if group_return_detail is not None:
        detail_path = base / "group_return_detail.csv"
        group_return_detail.to_csv(detail_path, index=False, date_format="%Y-%m-%d")
        out["group_return_detail"] = detail_path

    if group_return_summary is not None:
        summary_path = base / "group_return_summary.csv"
        group_return_summary.to_csv(summary_path, index=False)
        out["group_return_summary"] = summary_path

    if factor_weight_summary is not None:
        weight_path = base / "factor_weight_summary.csv"
        factor_weight_summary.to_csv(weight_path, index=False)
        out["factor_weight_summary"] = weight_path

    if factor_weight_train_summary is not None:
        train_weight_path = base / "factor_weight_train_summary.csv"
        factor_weight_train_summary.to_csv(train_weight_path, index=False)
        out["factor_weight_train_summary"] = train_weight_path

    if rolling_factor_weight_log is not None:
        rolling_path = base / "rolling_factor_weight_log.csv"
        rolling_factor_weight_log.to_csv(rolling_path, index=False, date_format="%Y-%m-%d")
        out["rolling_factor_weight_log"] = rolling_path

    return out
