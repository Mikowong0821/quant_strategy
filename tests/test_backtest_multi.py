"""backtest_multi：线性合成与 pre_fused 路径自检。"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backtest.backtest_multi import run_multi_backtest
from config import get_settings


def _tiny_prices() -> pd.DataFrame:
    days = pd.bdate_range("2024-01-01", periods=40)
    syms = ["AAA", "BBB"]
    rng = np.random.default_rng(0)
    px = 10.0 * np.cumprod(1.0 + rng.normal(0.001, 0.01, size=(len(days), len(syms))), axis=0)
    return pd.DataFrame(px, index=days, columns=syms)


class TestRunMultiBacktest(unittest.TestCase):
    def test_pre_fused_runs(self) -> None:
        prices = _tiny_prices()
        idx = pd.MultiIndex.from_product([prices.index, prices.columns], names=["date", "symbol"])
        g = np.random.default_rng(1)
        fused = pd.Series(g.standard_normal(len(idx)), index=idx)
        settings = get_settings()
        nav, meta = run_multi_backtest(
            fused=fused,
            prices=prices,
            settings=settings,
            factor_name="TEST_FUSED",
            top_k=2,
        )
        self.assertEqual(meta.get("multi_mode"), "pre_fused")
        self.assertGreater(len(nav), 0)
        self.assertEqual(meta.get("factor_name"), "TEST_FUSED")

    def test_linear_weight_constant_factors(self) -> None:
        prices = _tiny_prices()
        idx = pd.MultiIndex.from_product([prices.index, prices.columns], names=["date", "symbol"])
        f1 = pd.Series(1.0, index=idx, name="a")
        f2 = pd.Series(3.0, index=idx, name="b")
        settings = get_settings()
        nav, meta = run_multi_backtest(
            [f1, f2],
            weights=[0.25, 0.75],
            prices=prices,
            settings=settings,
            factor_name="LIN",
            top_k=2,
        )
        self.assertEqual(meta.get("multi_mode"), "linear_weight")
        self.assertEqual(meta.get("multi_weights"), [0.25, 0.75])


if __name__ == "__main__":
    unittest.main()
