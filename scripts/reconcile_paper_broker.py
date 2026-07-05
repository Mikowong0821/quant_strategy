#!/usr/bin/env python3
"""纸面账户与只读券商账户对账入口。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from live.broker import BrokerAccount, RealBrokerConfig, RealBrokerReadOnlyAdapter
from live.broker_reconcile import reconcile_paper_with_broker, save_reconciliation_outputs
from live.daily_paper_cli import DEFAULT_STRATEGY


def _load_account(path: Path) -> BrokerAccount:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("券商账户 CSV 为空: %s" % path)
    rec = frame.iloc[-1].to_dict()
    return BrokerAccount(
        cash=float(rec.get("cash", 0.0)),
        market_value=float(rec.get("market_value", 0.0)),
        total_asset=float(rec.get("total_asset", 0.0)),
        updated_at=str(rec.get("updated_at", "")),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="纸面账户与只读券商账户对账")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, help="策略名")
    parser.add_argument("--trade-date", default=None, help="对账日期；默认取纸面账户最新快照")
    parser.add_argument("--broker-provider", default="csv", help="券商/通道名称，仅用于记录")
    parser.add_argument("--broker-account-id", default="", help="非敏感账户标识")
    parser.add_argument("--broker-account", type=Path, required=True, help="券商账户 CSV，含 cash/market_value/total_asset")
    parser.add_argument("--broker-positions", type=Path, required=True, help="券商持仓 CSV，含 symbol/shares，可选 available_shares")
    parser.add_argument("--cash-tolerance", type=float, default=1.0, help="现金差异容忍值")
    parser.add_argument("--asset-tolerance", type=float, default=1.0, help="总资产差异容忍值")
    parser.add_argument("--share-tolerance", type=float, default=0.0, help="持仓股数差异容忍值")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    account = _load_account(args.broker_account)
    positions = pd.read_csv(args.broker_positions)
    broker = RealBrokerReadOnlyAdapter(
        RealBrokerConfig(
            provider=args.broker_provider,
            account_id=args.broker_account_id,
        ),
        account=account,
        positions=positions,
    )
    settings = get_settings()
    result = reconcile_paper_with_broker(
        settings,
        strategy=args.strategy,
        broker=broker,
        trade_date=args.trade_date,
        cash_tolerance=args.cash_tolerance,
        asset_tolerance=args.asset_tolerance,
        share_tolerance=args.share_tolerance,
    )
    paths = save_reconciliation_outputs(settings, result)
    status = "OK" if not result["issues"] else "WARNING"
    print("纸面 / 券商只读对账完成 status=%s issues=%s" % (status, ",".join(result["issues"]) or "none"))
    print("report=%s" % paths["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
