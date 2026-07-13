"""券商 Adapter Factory 自检。"""
from __future__ import annotations

import unittest
from dataclasses import replace

import pandas as pd

from config import get_settings
from live.broker import (
    BROKER_MODE_REAL_READONLY,
    BROKER_MODE_SIMULATED,
    BrokerReadOnlyError,
    RealBrokerReadOnlyAdapter,
    SimulatedBroker,
)
from live.broker_factory import (
    BrokerAdapterNotConfiguredError,
    build_broker_config,
    create_broker_adapter,
    normalize_broker_mode,
    normalize_broker_provider,
)


class BrokerFactoryTests(unittest.TestCase):
    def test_normalize_mode_aliases(self) -> None:
        self.assertEqual(normalize_broker_mode("paper_trading"), BROKER_MODE_SIMULATED)
        self.assertEqual(normalize_broker_mode("READ_ONLY"), BROKER_MODE_REAL_READONLY)
        self.assertEqual(normalize_broker_mode("semi_auto"), BROKER_MODE_REAL_READONLY)

        with self.assertRaises(ValueError):
            normalize_broker_mode("unknown")

    def test_real_mode_requires_provider(self) -> None:
        with self.assertRaises(BrokerAdapterNotConfiguredError):
            normalize_broker_provider("", mode=BROKER_MODE_REAL_READONLY)

    def test_build_broker_config_from_settings(self) -> None:
        settings = replace(
            get_settings(),
            broker_mode="REAL_READONLY",
            broker_provider="QMT",
            broker_account_id="demo",
        )

        config = build_broker_config(settings)

        self.assertEqual(config.mode, BROKER_MODE_REAL_READONLY)
        self.assertEqual(config.provider, "qmt")
        self.assertEqual(config.account_id, "demo")

    def test_create_simulated_broker(self) -> None:
        settings = replace(get_settings(), broker_mode="simulated")

        broker = create_broker_adapter(
            settings,
            cash=10_000.0,
            positions={"AAA": 100},
            latest_prices={"AAA": 10.0},
        )

        self.assertIsInstance(broker, SimulatedBroker)
        self.assertAlmostEqual(broker.get_cash(), 10_000.0)
        self.assertAlmostEqual(broker.get_account().market_value, 1_000.0)

    def test_create_real_readonly_snapshot_adapter(self) -> None:
        settings = replace(
            get_settings(),
            broker_mode="real_readonly",
            broker_provider="readonly_csv",
            broker_account_id="demo",
        )

        broker = create_broker_adapter(
            settings,
            account={"cash": 8_000.0, "market_value": 2_000.0, "total_asset": 10_000.0},
            positions=pd.DataFrame([{"symbol": "AAA", "shares": 100}]),
            latest_prices={"AAA": 20.0},
        )

        self.assertIsInstance(broker, RealBrokerReadOnlyAdapter)
        self.assertAlmostEqual(broker.get_account().total_asset, 10_000.0)
        with self.assertRaises(BrokerReadOnlyError):
            broker.submit_order(symbol="AAA", side="BUY", qty=100, price=10.0)

    def test_real_trading_is_blocked_until_adapter_exists(self) -> None:
        settings = replace(get_settings(), broker_mode="real_trading", broker_provider="qmt")

        with self.assertRaises(BrokerAdapterNotConfiguredError):
            create_broker_adapter(settings)


if __name__ == "__main__":
    unittest.main()
