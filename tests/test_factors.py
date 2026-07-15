"""factors：量价因子与面板构建自检。"""
from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from config import get_settings
from factors.factor_finance import (
    calc_cash_profit_quality,
    calc_free_cash_flow_yield,
    calc_gross_margin,
    calc_low_debt_to_assets,
    calc_profit_growth,
    calc_revenue_growth,
)
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

    def test_finance_factors_align_by_announcement_date(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=6)
        prices_long = pd.DataFrame(
            {
                "trade_date": list(days) * 2,
                "ts_code": ["AAA"] * len(days) + ["BBB"] * len(days),
                "close": [10.0] * len(days) + [20.0] * len(days),
            }
        )
        finance = pd.DataFrame(
            [
                {
                    "ts_code": "AAA",
                    "ann_date": "2024-01-03",
                    "grossprofit_margin": 40.0,
                    "debt_to_assets": 30.0,
                    "or_yoy": 10.0,
                    "netprofit_yoy": 20.0,
                    "fcff_ps": 2.0,
                    "ocf_to_profit": 120.0,
                },
                {
                    "ts_code": "BBB",
                    "ann_date": "2024-01-04",
                    "grossprofit_margin": 25.0,
                    "debt_to_assets": 60.0,
                    "or_yoy": -5.0,
                    "netprofit_yoy": -10.0,
                    "fcff_ps": 1.0,
                    "ocf_to_profit": 80.0,
                },
            ]
        )

        gross = calc_gross_margin(finance, prices_long)
        low_debt = calc_low_debt_to_assets(finance, prices_long)
        revenue_growth = calc_revenue_growth(finance, prices_long)
        profit_growth = calc_profit_growth(finance, prices_long)
        fcf_yield = calc_free_cash_flow_yield(finance, prices_long)
        cash_quality = calc_cash_profit_quality(finance, prices_long)

        self.assertTrue(pd.isna(gross.loc[(days[1], "AAA")]))
        self.assertAlmostEqual(float(gross.loc[(days[2], "AAA")]), 40.0)
        self.assertAlmostEqual(float(low_debt.loc[(days[2], "AAA")]), -30.0)
        self.assertAlmostEqual(float(revenue_growth.loc[(days[2], "AAA")]), 10.0)
        self.assertAlmostEqual(float(profit_growth.loc[(days[2], "AAA")]), 20.0)
        self.assertAlmostEqual(float(fcf_yield.loc[(days[2], "AAA")]), 2.0 / 10.0)
        self.assertAlmostEqual(float(cash_quality.loc[(days[2], "AAA")]), 120.0)

    def test_build_panel_contains_finance_extension_factors(self) -> None:
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
                        "close": 10.0 + i,
                        "volume": 100.0 + i,
                    }
                )
        long_df = pd.DataFrame(rows)
        finance = pd.DataFrame(
            [
                {
                    "ts_code": "AAA",
                    "ann_date": "2024-01-02",
                    "eps": 1.0,
                    "roe": 12.0,
                    "grossprofit_margin": 40.0,
                    "netprofit_margin": 15.0,
                    "debt_to_assets": 30.0,
                    "or_yoy": 20.0,
                    "netprofit_yoy": 25.0,
                    "fcff_ps": 2.0,
                    "ocf_to_profit": 120.0,
                },
                {
                    "ts_code": "BBB",
                    "ann_date": "2024-01-02",
                    "eps": 2.0,
                    "roe": 8.0,
                    "grossprofit_margin": 30.0,
                    "netprofit_margin": 10.0,
                    "debt_to_assets": 50.0,
                    "or_yoy": 5.0,
                    "netprofit_yoy": 7.0,
                    "fcff_ps": 0.5,
                    "ocf_to_profit": 70.0,
                },
            ]
        )

        with patch("factors.panel_builder.fetch_fina_indicator_panel", return_value=finance):
            panel = build_four_factor_panel(long_df, long_df, get_settings())

        for col in (
            "GROSS_MARGIN",
            "NET_MARGIN",
            "LOW_DEBT_TO_ASSETS",
            "REVENUE_GROWTH",
            "PROFIT_GROWTH",
            "FREE_CASH_FLOW_YIELD",
            "CASH_PROFIT_QUALITY",
        ):
            self.assertIn(col, panel.columns)
            self.assertGreater(int(panel[col].notna().sum()), 0)

    def test_build_panel_uses_local_fina_indicator_cache(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=20)
        rows = []
        for sym in ["AAA", "BBB"]:
            for i, dt in enumerate(days):
                rows.append(
                    {
                        "trade_date": dt,
                        "ts_code": sym,
                        "open": 10.0 + i,
                        "high": 10.5 + i,
                        "low": 9.5 + i,
                        "close": 10.0 + i,
                        "volume": 100.0 + i,
                    }
                )
        long_df = pd.DataFrame(rows)
        finance = pd.DataFrame(
            [
                {
                    "ts_code": "AAA",
                    "ann_date": "2024-01-02",
                    "eps": 1.0,
                    "roe": 12.0,
                    "grossprofit_margin": 40.0,
                    "netprofit_margin": 15.0,
                    "debt_to_assets": 30.0,
                    "or_yoy": 20.0,
                    "netprofit_yoy": 25.0,
                    "ocfps": 2.0,
                    "ocf_to_profit": 120.0,
                },
                {
                    "ts_code": "BBB",
                    "ann_date": "2024-01-02",
                    "eps": 2.0,
                    "roe": 8.0,
                    "grossprofit_margin": 30.0,
                    "netprofit_margin": 10.0,
                    "debt_to_assets": 50.0,
                    "or_yoy": 5.0,
                    "netprofit_yoy": 7.0,
                    "ocfps": 0.5,
                    "ocf_to_profit": 70.0,
                },
            ]
        )

        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "fina_indicator.csv"
            finance.to_csv(cache_path, index=False)
            settings = replace(get_settings(), fina_indicator_cache_path=cache_path)
            with patch("factors.panel_builder.fetch_fina_indicator_panel") as fetch_mock:
                panel = build_four_factor_panel(long_df, long_df, settings)

        fetch_mock.assert_not_called()
        self.assertGreater(int(panel["FREE_CASH_FLOW_YIELD"].notna().sum()), 0)
        self.assertGreater(int(panel["CASH_PROFIT_QUALITY"].notna().sum()), 0)


if __name__ == "__main__":
    unittest.main()
