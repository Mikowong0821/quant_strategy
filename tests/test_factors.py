"""factors：量价因子与面板构建自检。"""
from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

import pandas as pd

from config import get_settings
from factors.factor_momentum import calc_momentum
from factors.factor_reversal import calc_reversal
from factors.factor_volume import calc_volume_ratio
from factors.panel_builder import DEFAULT_FACTOR_ORDER, build_four_factor_panel


class TestPriceVolumeFactors(unittest.TestCase):
    def test_momentum_reversal_and_volume_ratio(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=6)
        prices = pd.DataFrame(
            {"AAA": [10, 11, 12, 13, 14, 15], "BBB": [10, 9, 8, 7, 6, 5]},
            index=days,
        )
        volume = pd.DataFrame(
            {"AAA": [100, 100, 100, 200, 200, 200], "BBB": [50, 50, 100, 100, 100, 200]},
            index=days,
        )

        mom = calc_momentum(prices, lookback=2)
        rev = calc_reversal(prices, lookback=2)
        vr = calc_volume_ratio(volume, window=3)

        self.assertAlmostEqual(float(mom.loc[(days[2], "AAA")]), 12 / 10 - 1.0)
        self.assertAlmostEqual(float(rev.loc[(days[2], "AAA")]), -(12 / 10 - 1.0))
        self.assertGreater(float(rev.loc[(days[2], "BBB")]), 0.0)
        self.assertAlmostEqual(float(vr.loc[(days[3], "AAA")]), 200 / ((100 + 100 + 200) / 3) - 1.0)

    def test_build_panel_contains_price_volume_factors(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=80)
        syms = ["AAA", "BBB"]
        rows = []
        for sym in syms:
            for i, dt in enumerate(days):
                rows.append(
                    {
                        "trade_date": dt,
                        "ts_code": sym,
                        "open": 10.0 + i,
                        "high": 10.5 + i,
                        "low": 9.5 + i,
                        "close": 10.0 + i + (0.1 if sym == "AAA" else 0.0),
                        "volume": 100.0 + i * (2 if sym == "AAA" else 1),
                    }
                )
        long_df = pd.DataFrame(rows)
        settings = replace(
            get_settings(),
            momentum_lookback=5,
            momentum_long_lookback=20,
            reversal_lookback=3,
            volume_ratio_window=10,
        )
        with patch("factors.panel_builder.fetch_fina_indicator_panel", return_value=pd.DataFrame()):
            panel = build_four_factor_panel(long_df, long_df, settings)

        self.assertEqual(list(panel.columns), DEFAULT_FACTOR_ORDER)
        self.assertIn("MOMENTUM_60D", panel.columns)
        self.assertIn("REVERSAL_5D", panel.columns)
        self.assertIn("VOLUME_RATIO_20D", panel.columns)
        self.assertGreater(int(panel["MOMENTUM_60D"].notna().sum()), 0)
        self.assertGreater(int(panel["REVERSAL_5D"].notna().sum()), 0)
        self.assertGreater(int(panel["VOLUME_RATIO_20D"].notna().sum()), 0)


if __name__ == "__main__":
    unittest.main()
