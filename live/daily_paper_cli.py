"""
每日纸面交易命令行辅助逻辑。

从已有回测输出读取最近一期目标权重和最新价格，再调用 paper_runner。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from config import Settings, get_settings
from live.paper_guard import (
    format_guard_issues,
    raise_on_guard_errors,
    validate_daily_inputs,
    validate_daily_result,
)
from live.paper_runner import run_daily_paper_trade
from live.paper_report import save_daily_paper_report


DEFAULT_STRATEGY = "FUSED_ROLLING_SCORE_WEIGHTED"


def _to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def load_latest_target_weights(
    path: Path,
    *,
    trade_date: Any = None,
) -> tuple[pd.Timestamp, pd.Series]:
    """从 rebalance_logs/<strategy>.csv 读取不晚于 trade_date 的最近一期目标权重。"""
    if not path.exists():
        raise FileNotFoundError("未找到调仓日志: %s；请先运行 main.py 生成 rebalance_logs" % path)
    frame = pd.read_csv(path)
    required = {"date", "symbol", "weight"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("调仓日志缺少必要列: %s" % ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("调仓日志为空: %s" % path)

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if trade_date is not None:
        dt = pd.Timestamp(trade_date)
        frame = frame[frame["date"] <= dt]
        if frame.empty:
            raise ValueError("调仓日志中没有不晚于 %s 的目标权重" % dt.strftime("%Y-%m-%d"))

    latest_date = frame["date"].max()
    latest = frame[frame["date"] == latest_date].copy()
    if "selected" in latest.columns:
        latest = latest[_to_bool_series(latest["selected"])]
    latest["weight"] = pd.to_numeric(latest["weight"], errors="coerce").fillna(0.0)
    latest = latest[latest["weight"] > 0.0]
    if latest.empty:
        raise ValueError("最近一期调仓日志没有有效目标权重: %s" % latest_date.strftime("%Y-%m-%d"))

    weights = pd.Series(
        latest["weight"].astype(float).to_numpy(),
        index=latest["symbol"].astype(str),
        dtype=float,
    )
    weights = weights.groupby(level=0).sum().sort_index()
    return latest_date, weights


def load_latest_prices(
    path: Path,
    *,
    trade_date: Any = None,
) -> tuple[pd.Timestamp, pd.Series]:
    """从 cache/prices_wide_close.csv 读取最新价格，或读取不晚于 trade_date 的最近价格。"""
    if not path.exists():
        raise FileNotFoundError("未找到价格缓存: %s；请先运行 main.py 生成 prices_wide_close.csv" % path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("价格缓存为空: %s" % path)

    date_col = "date" if "date" in frame.columns else frame.columns[0]
    frame = frame.rename(columns={date_col: "date"}).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if trade_date is not None:
        dt = pd.Timestamp(trade_date)
        frame = frame[frame["date"] <= dt]
        if frame.empty:
            raise ValueError("价格缓存中没有不晚于 %s 的价格" % dt.strftime("%Y-%m-%d"))

    latest = frame.sort_values("date").iloc[-1]
    latest_date = pd.Timestamp(latest["date"])
    prices = latest.drop(labels=["date"]).astype(float)
    prices.index = prices.index.astype(str)
    prices = prices[prices.notna() & (prices > 0.0)].sort_index()
    if prices.empty:
        raise ValueError("最近价格行没有有效价格: %s" % latest_date.strftime("%Y-%m-%d"))
    return latest_date, prices


def load_trade_status(
    path: Path | None,
    *,
    trade_date: Any = None,
) -> pd.DataFrame | None:
    """读取可选交易状态 CSV；若含 date 列，则每个 symbol 取不晚于 trade_date 的最近状态。"""
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError("未找到交易状态文件: %s" % path)
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    symbol_col = "symbol" if "symbol" in frame.columns else "ts_code"
    if symbol_col not in frame.columns:
        raise ValueError("交易状态文件须包含 symbol 或 ts_code 列")

    out = frame.copy()
    if symbol_col != "symbol":
        out = out.rename(columns={symbol_col: "symbol"})
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        if trade_date is not None:
            out = out[out["date"] <= pd.Timestamp(trade_date)]
        if out.empty:
            return out
        out = out.sort_values(["symbol", "date"]).groupby("symbol", as_index=False).tail(1)
    return out


def run_daily_paper_from_outputs(
    settings: Settings,
    *,
    strategy: str = DEFAULT_STRATEGY,
    trade_date: Any = None,
    rebalance_log_path: Path | None = None,
    prices_path: Path | None = None,
    trade_status_path: Path | None = None,
    persist_outputs: bool = True,
    generate_report: bool = True,
    run_guard: bool = True,
    max_price_age_days: int = 7,
) -> dict[str, Any]:
    """从 output/ 下已有文件读取输入并执行单日纸面交易。"""
    rebalance_path = (
        rebalance_log_path
        if rebalance_log_path is not None
        else settings.output_dir / "rebalance_logs" / ("%s.csv" % strategy.replace("/", "_"))
    )
    price_cache_path = (
        prices_path
        if prices_path is not None
        else settings.output_dir / "cache" / "prices_wide_close.csv"
    )

    requested_date = pd.Timestamp(trade_date) if trade_date is not None else None
    price_date, latest_prices = load_latest_prices(price_cache_path, trade_date=requested_date)
    run_date = requested_date if requested_date is not None else price_date
    target_date, target_weights = load_latest_target_weights(rebalance_path, trade_date=run_date)
    trade_status = load_trade_status(trade_status_path, trade_date=run_date)
    guard_issues = (
        validate_daily_inputs(
            target_weights=target_weights,
            latest_prices=latest_prices,
            run_date=run_date,
            target_date=target_date,
            price_date=price_date,
            max_price_age_days=max_price_age_days,
        )
        if run_guard
        else []
    )
    raise_on_guard_errors(guard_issues)

    result = run_daily_paper_trade(
        settings,
        strategy=strategy,
        target_weights=target_weights,
        latest_prices=latest_prices,
        trade_date=run_date,
        trade_status=trade_status,
        persist_outputs=persist_outputs,
    )
    result["input_paths"] = {
        "rebalance_log": rebalance_path,
        "prices": price_cache_path,
        "trade_status": trade_status_path,
    }
    result["target_date"] = target_date
    result["price_date"] = price_date
    if run_guard:
        guard_issues.extend(validate_daily_result(result))
        raise_on_guard_errors(guard_issues)
    result["guard_issues"] = guard_issues
    if persist_outputs and generate_report:
        report_path = save_daily_paper_report(settings, result)
        result.setdefault("paths", {})["paper_report"] = report_path
    return result


def format_daily_paper_summary(result: dict[str, Any]) -> str:
    """生成命令行摘要。"""
    orders = result["orders"]
    checks = result["order_checks"]
    trades = result["paper_trades"]
    snapshot = result["account_snapshot"]
    paths = result.get("paths", {})
    guard_issues = result.get("guard_issues", [])

    n_orders = int(len(orders))
    n_pass = int((checks["check_status"] == "PASS").sum()) if not checks.empty else 0
    n_block = int((checks["check_status"] == "BLOCK").sum()) if not checks.empty else 0
    n_filled = int((trades["fill_status"] == "FILLED").sum()) if not trades.empty else 0
    n_skipped = int((trades["fill_status"] == "SKIPPED").sum()) if not trades.empty else 0

    lines = [
        "每日纸面交易完成",
        "strategy=%s" % result["strategy"],
        "trade_date=%s target_date=%s price_date=%s"
        % (
            pd.Timestamp(result["trade_date"]).strftime("%Y-%m-%d"),
            pd.Timestamp(result["target_date"]).strftime("%Y-%m-%d"),
            pd.Timestamp(result["price_date"]).strftime("%Y-%m-%d"),
        ),
        "orders=%d pass=%d block=%d filled=%d skipped=%d"
        % (n_orders, n_pass, n_block, n_filled, n_skipped),
        "cash=%.2f market_value=%.2f total_asset=%.2f n_positions=%d"
        % (
            float(snapshot.get("cash", 0.0)),
            float(snapshot.get("market_value", 0.0)),
            float(snapshot.get("total_asset", 0.0)),
            int(float(snapshot.get("n_positions", 0.0))),
        ),
    ]
    if paths:
        lines.append("outputs:")
        for key, value in paths.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    lines.append("  %s.%s=%s" % (key, sub_key, sub_value))
            else:
                lines.append("  %s=%s" % (key, value))
    if guard_issues:
        lines.append("guard:")
        lines.append(format_guard_issues(guard_issues))
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行每日纸面交易流程")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, help="策略名，默认 FUSED_ROLLING_SCORE_WEIGHTED")
    parser.add_argument("--trade-date", default=None, help="运行日期；默认使用价格缓存最新日期")
    parser.add_argument("--rebalance-log", type=Path, default=None, help="调仓日志 CSV；默认 output/rebalance_logs/<strategy>.csv")
    parser.add_argument("--prices", type=Path, default=None, help="价格宽表 CSV；默认 output/cache/prices_wide_close.csv")
    parser.add_argument("--trade-status", type=Path, default=None, help="可选交易状态 CSV，含 symbol/ts_code 与 is_suspended/is_limit_up/is_limit_down")
    parser.add_argument("--no-persist", action="store_true", help="只运行不写订单、成交和账户状态文件")
    parser.add_argument("--no-report", action="store_true", help="不生成 Markdown 纸面交易日报")
    parser.add_argument("--no-guard", action="store_true", help="跳过日终输入和结果异常检查")
    parser.add_argument("--max-price-age-days", type=int, default=7, help="价格日期距运行日期超过该天数时给出 warning")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    settings = get_settings()
    result = run_daily_paper_from_outputs(
        settings,
        strategy=args.strategy,
        trade_date=args.trade_date,
        rebalance_log_path=args.rebalance_log,
        prices_path=args.prices,
        trade_status_path=args.trade_status,
        persist_outputs=not args.no_persist,
        generate_report=not args.no_report,
        run_guard=not args.no_guard,
        max_price_age_days=args.max_price_age_days,
    )
    print(format_daily_paper_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
