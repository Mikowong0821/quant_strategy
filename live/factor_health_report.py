"""
增强版因子健康日报。

本模块不重新计算因子，只读取研究主流程已经落盘的诊断 CSV，把因子
入选、样本外、滚动样本外、权重漂移、冗余和牛熊市分段压缩成日常
纸面交易可以阅读的健康摘要。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings


HEALTH_REPORT_COLUMNS = [
    "category",
    "status",
    "severity",
    "summary",
    "detail",
    "action",
]

_STATUS_SEVERITY = {
    "OK": 0,
    "PASS": 0,
    "ROBUST": 0,
    "WATCH": 1,
    "DEGRADED": 2,
    "FAILED": 3,
    "REJECT": 3,
    "UNSTABLE": 3,
    "NO_DATA": 1,
    "UNKNOWN": 1,
}


def _empty_report() -> pd.DataFrame:
    return pd.DataFrame(columns=HEALTH_REPORT_COLUMNS)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _status_rank(status: Any) -> int:
    return int(_STATUS_SEVERITY.get(str(status).upper(), 1))


def _worst_status(statuses: list[Any]) -> str:
    clean = [str(x).upper() for x in statuses if str(x).strip()]
    if not clean:
        return "UNKNOWN"
    return max(clean, key=_status_rank)


def _fmt_pct(value: Any) -> str:
    try:
        return "%.2f%%" % (float(value) * 100.0)
    except (TypeError, ValueError):
        return ""


def _fmt_float(value: Any, digits: int = 4) -> str:
    try:
        return ("%." + str(digits) + "f") % float(value)
    except (TypeError, ValueError):
        return ""


def _row(category: str, status: str, summary: str, detail: str, action: str) -> dict[str, Any]:
    status_u = str(status).upper()
    return {
        "category": category,
        "status": status_u,
        "severity": _status_rank(status_u),
        "summary": summary,
        "detail": detail,
        "action": action,
    }


def default_factor_health_paths(settings: Settings) -> dict[str, Path]:
    """返回增强健康日报默认读取的诊断文件路径。"""
    return {
        "factor_decay_monitor": settings.output_dir / "factor_validation" / "factor_decay_monitor.csv",
        "rolling_oos_summary": settings.output_dir / "factor_validation" / "rolling_out_of_sample_summary.csv",
        "factor_selection": settings.output_dir / "factor_diagnostics" / "factor_selection_summary.csv",
        "factor_redundancy": settings.output_dir / "factor_diagnostics" / "factor_redundancy_report.csv",
        "weight_stability": settings.output_dir / "factor_diagnostics" / "factor_weight_stability_summary.csv",
        "weight_drift_events": settings.output_dir / "factor_diagnostics" / "factor_weight_drift_events.csv",
        "market_regime": settings.output_dir / "market_regime" / "strategy_regime_summary.csv",
    }


def _selection_row(selection: pd.DataFrame) -> dict[str, Any]:
    if selection.empty or "decision" not in selection.columns:
        return _row("因子入选", "UNKNOWN", "未找到因子入选表", "factor_selection_summary.csv missing", "先运行研究主流程")

    decision = selection["decision"].astype(str).str.upper()
    n_pass = int((decision == "PASS").sum())
    n_watch = int((decision == "WATCH").sum())
    n_reject = int((decision == "REJECT").sum())
    status = "PASS" if n_reject == 0 and n_watch <= max(1, n_pass) else "WATCH"
    if n_pass == 0:
        status = "FAILED"
    selected = []
    if "selected_for_fusion" in selection.columns and "factor" in selection.columns:
        flag = selection["selected_for_fusion"].astype(str).str.lower().isin({"1", "true", "yes"})
        selected = selection.loc[flag, "factor"].astype(str).tolist()
    detail = "PASS=%d; WATCH=%d; REJECT=%d; selected=%s" % (
        n_pass,
        n_watch,
        n_reject,
        ",".join(selected[:8]) if selected else "none",
    )
    return _row(
        "因子入选",
        status,
        "候选因子 PASS=%d，WATCH=%d，REJECT=%d" % (n_pass, n_watch, n_reject),
        detail,
        "主融合只使用 PASS 因子；WATCH 继续观察；REJECT 暂不进入融合",
    )


def _decay_row(monitor: pd.DataFrame) -> dict[str, Any]:
    if monitor.empty or "status" not in monitor.columns:
        return _row("样本外失效", "UNKNOWN", "未找到因子失效监控表", "factor_decay_monitor.csv missing", "先运行研究主流程")

    status = monitor["status"].astype(str).str.upper()
    counts = status.value_counts().to_dict()
    worst = _worst_status(status.tolist())
    risky = monitor[status.isin(["WATCH", "DEGRADED", "FAILED"])].copy()
    risky_factors = risky["factor"].astype(str).tolist() if "factor" in risky.columns else []
    detail = "OK=%d; WATCH=%d; DEGRADED=%d; FAILED=%d; risky=%s" % (
        int(counts.get("OK", 0)),
        int(counts.get("WATCH", 0)),
        int(counts.get("DEGRADED", 0)),
        int(counts.get("FAILED", 0)),
        ",".join(risky_factors[:8]) if risky_factors else "none",
    )
    return _row(
        "样本外失效",
        worst,
        "最差状态=%s，风险因子=%d" % (worst, int(len(risky))),
        detail,
        "FAILED/DEGRADED 因子暂停加权；WATCH 因子降低权重或等待下一次验证",
    )


def _rolling_oos_row(rolling: pd.DataFrame) -> dict[str, Any]:
    if rolling.empty or "status" not in rolling.columns:
        return _row("滚动样本外", "UNKNOWN", "未找到滚动样本外摘要", "rolling_out_of_sample_summary.csv missing", "定期运行滚动样本外验证")

    status = rolling["status"].astype(str).str.upper()
    unstable = rolling[status == "UNSTABLE"].copy()
    worst = _worst_status(status.tolist())
    stable_rate = ""
    if "stable_window_rate" in rolling.columns:
        stable_rate = _fmt_pct(pd.to_numeric(rolling["stable_window_rate"], errors="coerce").mean())
    detail = "UNSTABLE=%d/%d; avg_stable_window_rate=%s" % (
        int(len(unstable)),
        int(len(rolling)),
        stable_rate,
    )
    if "factor" in unstable.columns and not unstable.empty:
        detail += "; unstable=%s" % ",".join(unstable["factor"].astype(str).head(8))
    return _row(
        "滚动样本外",
        worst,
        "滚动窗口不稳定因子=%d/%d" % (int(len(unstable)), int(len(rolling))),
        detail,
        "连续多个窗口不稳定的因子，不进入主融合或显著降权",
    )


def _weight_stability_row(stability: pd.DataFrame, drift_events: pd.DataFrame) -> dict[str, Any]:
    if stability.empty and drift_events.empty:
        return _row("权重漂移", "UNKNOWN", "未找到权重稳定性与漂移事件表", "factor_weight_stability*.csv missing", "先运行滚动权重融合流程")

    statuses = []
    if not stability.empty and "status" in stability.columns:
        statuses = stability["status"].astype(str).str.upper().tolist()
    n_watch = sum(1 for x in statuses if x not in {"PASS", "OK"})
    n_events = int(len(drift_events)) if drift_events is not None else 0
    high_events = 0
    event_detail = "none"
    if drift_events is not None and not drift_events.empty:
        sev = drift_events.get("severity", pd.Series(dtype=str)).astype(str).str.upper()
        high_events = int((sev == "HIGH").sum())
        event_cols = [c for c in ["date", "factor", "event_type", "abs_weight_change"] if c in drift_events.columns]
        if event_cols:
            last = drift_events.tail(3).loc[:, event_cols].astype(str)
            event_detail = "; ".join(" / ".join(rec.values()) for rec in last.to_dict("records"))
    status = "PASS"
    if high_events > 0:
        status = "DEGRADED"
    elif n_watch > 0 or n_events > 0:
        status = "WATCH"
    detail = "non_pass_factors=%d; drift_events=%d; high_events=%d; latest=%s" % (
        int(n_watch),
        n_events,
        high_events,
        event_detail,
    )
    return _row(
        "权重漂移",
        status,
        "权重异常因子=%d，漂移事件=%d" % (int(n_watch), n_events),
        detail,
        "有 HIGH 漂移时人工复核；WATCH 漂移进入日报观察",
    )


def _redundancy_row(redundancy: pd.DataFrame) -> dict[str, Any]:
    if redundancy.empty:
        return _row("因子冗余", "PASS", "未发现超过阈值的高相关冗余因子对", "redundancy_pairs=0", "维持当前候选池")
    drops = []
    if "recommended_drop" in redundancy.columns:
        drops = sorted(set(redundancy["recommended_drop"].dropna().astype(str)))
    status = "WATCH" if len(redundancy) <= 3 else "DEGRADED"
    detail = "pairs=%d; recommended_drop=%s" % (
        int(len(redundancy)),
        ",".join(drops[:8]) if drops else "none",
    )
    return _row(
        "因子冗余",
        status,
        "高相关因子对=%d" % int(len(redundancy)),
        detail,
        "被重复计票的因子只保留一个主表达，其余降权或转为观察",
    )


def _market_regime_row(regime: pd.DataFrame, strategy: str) -> dict[str, Any]:
    if regime.empty or "strategy" not in regime.columns:
        return _row("牛熊市分段", "UNKNOWN", "未找到牛熊市分段摘要", "strategy_regime_summary.csv missing", "新增策略或实盘前运行牛熊市分段")
    df = regime[regime["strategy"].astype(str) == str(strategy)].copy()
    if df.empty:
        return _row("牛熊市分段", "UNKNOWN", "当前策略没有分段摘要", "strategy=%s not_found" % strategy, "检查策略名或重新运行主流程")
    rec = df.iloc[0].to_dict()
    status = str(rec.get("status", "UNKNOWN")).upper()
    detail = "bull_excess=%s; bear_excess=%s; sideways_excess=%s; positive_excess_regime_rate=%s" % (
        _fmt_float(rec.get("bull_excess_ann_return")),
        _fmt_float(rec.get("bear_excess_ann_return")),
        _fmt_float(rec.get("sideways_excess_ann_return")),
        _fmt_pct(rec.get("positive_excess_regime_rate")),
    )
    return _row(
        "牛熊市分段",
        status,
        "市场状态稳健性=%s" % status,
        detail,
        "UNSTABLE 策略不直接扩大资金；先定位牛市进攻或熊市防守短板",
    )


def build_factor_health_report(
    settings: Settings,
    *,
    strategy: str,
    paths: dict[str, Path] | None = None,
) -> pd.DataFrame:
    """读取诊断 CSV 并生成增强版因子健康总览。"""
    p = default_factor_health_paths(settings)
    if paths:
        p.update(paths)

    rows = [
        _selection_row(_read_csv(p["factor_selection"])),
        _decay_row(_read_csv(p["factor_decay_monitor"])),
        _rolling_oos_row(_read_csv(p["rolling_oos_summary"])),
        _weight_stability_row(_read_csv(p["weight_stability"]), _read_csv(p["weight_drift_events"])),
        _redundancy_row(_read_csv(p["factor_redundancy"])),
        _market_regime_row(_read_csv(p["market_regime"]), strategy),
    ]
    out = pd.DataFrame(rows)
    for col in HEALTH_REPORT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[HEALTH_REPORT_COLUMNS].sort_values(
        ["severity", "category"],
        ascending=[False, True],
    ).reset_index(drop=True)


def summarize_factor_health_report(report: pd.DataFrame | None) -> tuple[str, str]:
    """把增强版健康总览压缩成命令行摘要。"""
    if report is None or report.empty:
        return "UNKNOWN", "factor_health_report_missing"
    df = report.copy()
    if "status" not in df.columns:
        return "UNKNOWN", "factor_health_report_invalid"
    worst = _worst_status(df["status"].astype(str).tolist())
    risky = df[df["status"].astype(str).str.upper().isin(["WATCH", "DEGRADED", "FAILED", "REJECT", "UNSTABLE"])]
    categories = ",".join(risky["category"].astype(str).head(8)) if "category" in risky.columns and not risky.empty else "none"
    return worst, "risky_categories=%d:%s" % (int(len(risky)), categories)
