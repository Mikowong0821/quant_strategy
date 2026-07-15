"""Fetch and cache Tushare fina_indicator data for a stock pool."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from live.data_feed import fetch_fina_indicator_panel
from live.stock_pool import load_stock_pool


_CASH_FLOW_COLUMNS = (
    "fcff_ps",
    "fcfe_ps",
    "free_cashflow_ps",
    "ocfps",
    "cfps",
    "ocf_to_profit",
    "ocf_to_opincome",
    "salescash_to_or",
    "ocf_to_or",
    "netprofit_cash_cover",
    "cashflow_to_profit",
)


def _build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Tushare fina_indicator data for a stock pool and save it as a local CSV cache. "
            "The Tushare token is read only from TUSHARE_TOKEN or config._TUSHARE_TOKEN_LOCAL."
        )
    )
    parser.add_argument("--stock-pool", default=str(settings.stock_pool_path), help="股票池 xlsx/csv 路径")
    parser.add_argument("--code-col", default=settings.stock_pool_code_col, help="股票代码列名")
    parser.add_argument("--start", default=settings.backtest_start, help="回测开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=settings.backtest_end, help="回测结束日期 YYYY-MM-DD")
    parser.add_argument("--history-years", type=int, default=settings.fina_history_years, help="向前多取几年财报")
    parser.add_argument(
        "--output",
        default=str(settings.data_dir / "fina_indicator_cache.csv"),
        help="输出 CSV 路径",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    symbols = load_stock_pool(args.stock_pool, code_col=args.code_col)
    print("准备拉取 fina_indicator: symbols=%d start=%s end=%s history_years=%d" % (
        len(symbols),
        args.start,
        args.end,
        args.history_years,
    ))

    df = fetch_fina_indicator_panel(
        symbols,
        args.start,
        args.end,
        history_years=args.history_years,
    )
    if df.empty:
        raise RuntimeError("Tushare fina_indicator 返回空表，请检查 token、积分权限、股票代码和日期范围")

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    present_cash_cols = [col for col in _CASH_FLOW_COLUMNS if col in df.columns]
    print("财务缓存已保存:", out.resolve())
    print("行数=%d 股票数=%d 字段数=%d" % (
        len(df),
        df["ts_code"].nunique() if "ts_code" in df.columns else 0,
        len(df.columns),
    ))
    print("现金流候选字段命中:", present_cash_cols if present_cash_cols else "无")
    print("字段列表:", list(df.columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
