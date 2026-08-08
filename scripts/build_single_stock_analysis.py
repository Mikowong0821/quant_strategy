#!/usr/bin/env python3
"""Build a small single-stock analysis report from Tushare data."""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_tushare_token
from live.announcement_source import fetch_tushare_announcement_events
from live.stock_pool import normalize_ts_code


DEFAULT_NAMES = {
    "000981.SZ": "山子高科",
    "605366.SH": "宏柏新材",
    "603156.SH": "养元饮品",
    "605128.SH": "上海沿浦",
}


def _norm_date(value: str) -> str:
    return value.replace("-", "")[:8]


def _read_token(use_stdin: bool) -> str:
    if use_stdin:
        token = sys.stdin.readline().strip()
        if not token:
            raise SystemExit("token-stdin enabled but stdin is empty")
        return token
    return get_tushare_token()


def _zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    valid = x.dropna()
    if len(valid) < 2:
        return pd.Series(0.0, index=s.index)
    std = float(valid.std(ddof=0))
    if not math.isfinite(std) or std < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (x - float(valid.mean())) / std


def _max_drawdown(close: pd.Series) -> float:
    px = pd.to_numeric(close, errors="coerce").dropna()
    if px.empty:
        return float("nan")
    nav = px / float(px.iloc[0])
    dd = nav / nav.cummax() - 1.0
    return float(dd.min())


def _last_valid(frame: pd.DataFrame, symbol: str, columns: list[str]) -> dict[str, Any]:
    if frame.empty or "ts_code" not in frame.columns:
        return {}
    sub = frame[frame["ts_code"].astype(str) == symbol].copy()
    if sub.empty:
        return {}
    date_col = "ann_date" if "ann_date" in sub.columns else "trade_date"
    if date_col in sub.columns:
        sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
        sub = sub.sort_values(date_col)
    row = sub.iloc[-1]
    return {col: row.get(col, np.nan) for col in columns if col in sub.columns}


def _fetch_daily(pro: Any, symbols: list[str], start: str, end: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        df = pro.daily(ts_code=symbol, start_date=_norm_date(start), end_date=_norm_date(end))
        if df is None or df.empty:
            continue
        df = df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
        df["ts_code"] = symbol
        if "vol" in df.columns and "volume" not in df.columns:
            df = df.rename(columns={"vol": "volume"})
        keep = [c for c in ["trade_date", "ts_code", "open", "high", "low", "close", "volume", "amount"] if c in df.columns]
        frames.append(df[keep])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def _fetch_daily_basic(pro: Any, symbols: list[str], start: str, end: str) -> pd.DataFrame:
    fields = "ts_code,trade_date,turnover_rate,volume_ratio,pe,pe_ttm,pb,total_mv,circ_mv"
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        try:
            df = pro.daily_basic(ts_code=symbol, start_date=_norm_date(start), end_date=_norm_date(end), fields=fields)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df = df.copy()
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def _fetch_fina(pro: Any, symbols: list[str], start: str, end: str) -> pd.DataFrame:
    start_dt = pd.to_datetime(_norm_date(start), format="%Y%m%d") - pd.DateOffset(years=2)
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        try:
            df = pro.fina_indicator(ts_code=symbol, start_date=start_dt.strftime("%Y%m%d"), end_date=_norm_date(end))
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df = df.copy()
        if "ann_date" in df.columns:
            df["ann_date"] = pd.to_datetime(df["ann_date"], format="%Y%m%d", errors="coerce")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["ts_code", "ann_date"]).reset_index(drop=True)


def _build_snapshot(
    daily: pd.DataFrame,
    daily_basic: pd.DataFrame,
    fina: pd.DataFrame,
    names: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    finance_cols = [
        "ann_date",
        "end_date",
        "eps",
        "roe",
        "grossprofit_margin",
        "netprofit_margin",
        "debt_to_assets",
        "or_yoy",
        "netprofit_yoy",
        "ocfps",
        "cfps",
        "fcff_ps",
        "fcfe_ps",
        "ocf_to_profit",
        "salescash_to_or",
    ]
    basic_cols = ["trade_date", "turnover_rate", "volume_ratio", "pe", "pe_ttm", "pb", "total_mv", "circ_mv"]
    for symbol, sub in daily.groupby("ts_code"):
        sub = sub.sort_values("trade_date")
        close = pd.to_numeric(sub["close"], errors="coerce")
        last = sub.iloc[-1]
        ret = float(close.iloc[-1] / close.iloc[0] - 1.0) if len(close.dropna()) >= 2 else np.nan
        row: dict[str, Any] = {
            "symbol": symbol,
            "name": names.get(symbol, symbol),
            "latest_date": pd.Timestamp(last["trade_date"]).date().isoformat(),
            "latest_close": float(last["close"]),
            "period_return": ret,
            "momentum_20d": float(close.iloc[-1] / close.shift(20).iloc[-1] - 1.0) if len(close) > 20 else np.nan,
            "momentum_60d": float(close.iloc[-1] / close.shift(60).iloc[-1] - 1.0) if len(close) > 60 else np.nan,
            "reversal_5d": float(close.iloc[-1] / close.shift(5).iloc[-1] - 1.0) if len(close) > 5 else np.nan,
            "volatility_20d": float(close.pct_change().tail(20).std(ddof=0) * np.sqrt(252)) if len(close) > 20 else np.nan,
            "max_drawdown": _max_drawdown(close),
            "avg_amount_20d_wan": float(pd.to_numeric(sub.get("amount"), errors="coerce").tail(20).mean() / 10.0)
            if "amount" in sub.columns
            else np.nan,
        }
        row.update(_last_valid(daily_basic, symbol, basic_cols))
        row.update(_last_valid(fina, symbol, finance_cols))
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("symbol").reset_index(drop=True)


def _build_scores(snapshot: pd.DataFrame) -> pd.DataFrame:
    score_inputs = pd.DataFrame(index=snapshot["symbol"])
    score_inputs["MOMENTUM_20D"] = snapshot.set_index("symbol")["momentum_20d"]
    score_inputs["MOMENTUM_60D"] = snapshot.set_index("symbol")["momentum_60d"]
    score_inputs["REVERSAL_5D"] = snapshot.set_index("symbol")["reversal_5d"]
    score_inputs["LOW_VOLATILITY"] = -snapshot.set_index("symbol")["volatility_20d"]
    score_inputs["LIQUIDITY"] = snapshot.set_index("symbol")["avg_amount_20d_wan"]
    if "pe_ttm" in snapshot.columns:
        score_inputs["LOW_PE_TTM"] = -snapshot.set_index("symbol")["pe_ttm"]
    if "pb" in snapshot.columns:
        score_inputs["LOW_PB"] = -snapshot.set_index("symbol")["pb"]
    for col, alias in [
        ("roe", "ROE"),
        ("grossprofit_margin", "GROSS_MARGIN"),
        ("netprofit_margin", "NET_MARGIN"),
        ("debt_to_assets", "LOW_DEBT_TO_ASSETS"),
        ("or_yoy", "REVENUE_GROWTH"),
        ("netprofit_yoy", "PROFIT_GROWTH"),
        ("ocfps", "OCFPS"),
        ("ocf_to_profit", "CASH_PROFIT_QUALITY"),
    ]:
        if col in snapshot.columns:
            val = snapshot.set_index("symbol")[col]
            score_inputs[alias] = -val if col == "debt_to_assets" else val

    z = score_inputs.apply(_zscore, axis=0)
    score = z.mean(axis=1, skipna=True).rename("composite_score")
    out = pd.concat([z, score], axis=1).reset_index().rename(columns={"index": "symbol"})
    out = out.merge(snapshot[["symbol", "name"]], on="symbol", how="left")
    out["rank_in_sample"] = out["composite_score"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("rank_in_sample").reset_index(drop=True)


def _format_pct(x: Any) -> str:
    try:
        if pd.isna(x):
            return "-"
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "-"


def _format_num(x: Any, digits: int = 2) -> str:
    try:
        if pd.isna(x):
            return "-"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "-"


def _plot_prices(daily: pd.DataFrame, names: dict[str, str], path: Path) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".mplconfig"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for symbol, sub in daily.groupby("ts_code"):
        sub = sub.sort_values("trade_date")
        close = pd.to_numeric(sub["close"], errors="coerce")
        nav = close / float(close.iloc[0])
        ax.plot(sub["trade_date"], nav, label=symbol)
    ax.set_title("Single Stock Normalized Price")
    ax.set_ylabel("NAV from start")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_report(
    *,
    output_dir: Path,
    symbols: list[str],
    names: dict[str, str],
    start: str,
    end: str,
    snapshot: pd.DataFrame,
    scores: pd.DataFrame,
    events: pd.DataFrame,
    warnings: list[str],
) -> Path:
    lines: list[str] = []
    lines.append("# 三只股票单票分析报告")
    lines.append("")
    lines.append(f"- 分析区间：{start} 到 {end}")
    lines.append("- 股票：%s" % "、".join(names.get(s, s) for s in symbols))
    lines.append("- 说明：本报告只做三只股票之间的横向比较，不代表全市场排名，也不构成投资建议。")
    lines.append("")
    lines.append("## 综合结论")
    lines.append("")
    if not scores.empty:
        top = scores.iloc[0]
        lines.append(
            "按当前工程的简化综合分，三只里暂时排第一的是：%s（%s），综合分 %s。"
            % (top["name"], top["symbol"], _format_num(top["composite_score"], 3))
        )
        lines.append("")
    lines.append("这类单票分析更适合回答：")
    lines.append("")
    lines.append("```text")
    lines.append("这几只股票最近谁更强？")
    lines.append("谁的波动更大？")
    lines.append("谁的财务质量更好？")
    lines.append("谁最近有没有公告风险？")
    lines.append("如果放进一个更大的股票池，它是否值得进入候选池？")
    lines.append("```")
    lines.append("")
    lines.append("## 三只股票核心指标")
    lines.append("")
    lines.append("| 股票 | 最新日期 | 收盘价 | 区间收益 | 20日动量 | 60日动量 | 20日年化波动 | 最大回撤 | 近20日均成交额(万元) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for rec in snapshot.to_dict("records"):
        lines.append(
            "| {name} | {latest_date} | {close} | {ret} | {m20} | {m60} | {vol} | {mdd} | {amt} |".format(
                name=rec.get("name", rec.get("symbol")),
                latest_date=rec.get("latest_date", "-"),
                close=_format_num(rec.get("latest_close")),
                ret=_format_pct(rec.get("period_return")),
                m20=_format_pct(rec.get("momentum_20d")),
                m60=_format_pct(rec.get("momentum_60d")),
                vol=_format_pct(rec.get("volatility_20d")),
                mdd=_format_pct(rec.get("max_drawdown")),
                amt=_format_num(rec.get("avg_amount_20d_wan")),
            )
        )
    lines.append("")
    lines.append("![三只股票归一化走势](price_nav.png)")
    lines.append("")
    lines.append("## 财务与估值快照")
    lines.append("")
    lines.append("| 股票 | PE_TTM | PB | ROE | 毛利率 | 净利率 | 资产负债率 | 营收增速 | 利润增速 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rec in snapshot.to_dict("records"):
        lines.append(
            "| {name} | {pe} | {pb} | {roe} | {gm} | {nm} | {debt} | {rev} | {profit} |".format(
                name=rec.get("name", rec.get("symbol")),
                pe=_format_num(rec.get("pe_ttm")),
                pb=_format_num(rec.get("pb")),
                roe=_format_num(rec.get("roe")),
                gm=_format_num(rec.get("grossprofit_margin")),
                nm=_format_num(rec.get("netprofit_margin")),
                debt=_format_num(rec.get("debt_to_assets")),
                rev=_format_num(rec.get("or_yoy")),
                profit=_format_num(rec.get("netprofit_yoy")),
            )
        )
    lines.append("")
    lines.append("## 工程因子横向排名")
    lines.append("")
    lines.append("| 排名 | 股票 | 综合分 | 主要含义 |")
    lines.append("|---:|---|---:|---|")
    for rec in scores.to_dict("records"):
        lines.append(
            "| {rank} | {name} | {score} | 三只股票内部的相对 z-score 均值 |".format(
                rank=rec.get("rank_in_sample"),
                name=rec.get("name", rec.get("symbol")),
                score=_format_num(rec.get("composite_score"), 3),
            )
        )
    lines.append("")
    lines.append("注意：这里的综合分不是正式策略权重，只是把动量、低波、流动性、估值、质量、成长等指标做成三只股票内部的相对比较。")
    lines.append("")
    lines.append("## 公告事件")
    lines.append("")
    if events.empty:
        lines.append("这次没有拿到可用公告事件，或者当前 Token / 接口权限下公告接口返回为空。")
    else:
        recent = events.sort_values("event_date").tail(20)
        lines.append("本次拉到公告事件 %d 条，最近 20 条如下：" % len(events))
        lines.append("")
        lines.append("| 日期 | 股票 | 类型 | 标题 | 风险等级 |")
        lines.append("|---|---|---|---|---|")
        for rec in recent.to_dict("records"):
            sym = str(rec.get("symbol", ""))
            lines.append(
                "| {date} | {name} | {etype} | {title} | {risk} |".format(
                    date=str(rec.get("event_date", ""))[:10],
                    name=names.get(sym, sym),
                    etype=rec.get("event_type", ""),
                    title=str(rec.get("title", ""))[:60],
                    risk=rec.get("risk_level", ""),
                )
            )
    if warnings:
        lines.append("")
        lines.append("## 运行提示")
        lines.append("")
        for item in warnings:
            lines.append("- " + item)
    lines.append("")
    lines.append("## 下一步怎么用")
    lines.append("")
    lines.append("如果只是看这三只股票，结果只能说明三者相对强弱。更稳妥的用法是：把它们放入一个明确股票池，再让主策略统一排序、过滤、配权。")
    path = output_dir / "single_stock_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Tushare based single-stock analysis report.")
    parser.add_argument("--symbols", required=True, help="Comma separated ts_code list.")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-08-07")
    parser.add_argument("--output-dir", type=Path, default=Path("output/single_stock_analysis"))
    parser.add_argument("--token-stdin", action="store_true", help="Read Tushare token from stdin first line.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    symbols = [normalize_ts_code(x) for x in args.symbols.split(",") if normalize_ts_code(x)]
    if not symbols:
        raise SystemExit("No valid symbols.")
    names = {symbol: DEFAULT_NAMES.get(symbol, symbol) for symbol in symbols}
    token = _read_token(args.token_stdin)

    try:
        import tushare as ts
    except ImportError as exc:
        raise SystemExit("tushare is required") from exc
    pro = ts.pro_api(token)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    daily = _fetch_daily(pro, symbols, args.start, args.end)
    if daily.empty:
        raise SystemExit("No daily data fetched.")
    daily_basic = _fetch_daily_basic(pro, symbols, args.start, args.end)
    if daily_basic.empty:
        warnings.append("daily_basic 为空：估值、换手率和市值指标可能缺失。")
    fina = _fetch_fina(pro, symbols, args.start, args.end)
    if fina.empty:
        warnings.append("fina_indicator 为空：ROE、利润率、成长和现金流指标可能缺失。")

    try:
        events = fetch_tushare_announcement_events(symbols, args.start, args.end, token=token)
    except Exception as exc:
        events = pd.DataFrame()
        warnings.append("公告接口不可用或权限不足：%s" % str(exc).splitlines()[0][:120])

    snapshot = _build_snapshot(daily, daily_basic, fina, names)
    scores = _build_scores(snapshot)

    daily.to_csv(output_dir / "prices_daily.csv", index=False)
    daily_basic.to_csv(output_dir / "daily_basic.csv", index=False)
    fina.to_csv(output_dir / "fina_indicator.csv", index=False)
    events.to_csv(output_dir / "announcement_events.csv", index=False)
    snapshot.to_csv(output_dir / "latest_snapshot.csv", index=False)
    scores.to_csv(output_dir / "factor_scores.csv", index=False)
    _plot_prices(daily, names, output_dir / "price_nav.png")
    report = _write_report(
        output_dir=output_dir,
        symbols=symbols,
        names=names,
        start=args.start,
        end=args.end,
        snapshot=snapshot,
        scores=scores,
        events=events,
        warnings=warnings,
    )

    print("output_dir=%s" % output_dir)
    print("report=%s" % report)
    if not scores.empty:
        cols = ["rank_in_sample", "symbol", "name", "composite_score"]
        print(scores[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
