"""backtest_single：配权模式与 _weights_for_rebalance 自检。"""
from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from backtest.backtest_single import _weights_for_rebalance, run_single_backtest
from config import get_settings


def _price_wide_for_weights() -> pd.DataFrame:
    days = pd.bdate_range("2024-01-01", periods=80)
    syms = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(2)
    px = 10.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.012, size=(len(days), len(syms))), axis=0)
    return pd.DataFrame(px, index=days, columns=syms)


class TestWeightsForRebalance(unittest.TestCase):
    def test_equal_mode(self) -> None:
        px = _price_wide_for_weights()
        s = get_settings()
        s2 = replace(s, portfolio_weighting="equal")
        dt = px.index[50]
        w, lab = _weights_for_rebalance(px, ["AAA", "BBB"], dt, s2)
        self.assertEqual(lab, "equal")
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertEqual(len(w), 2)
        self.assertTrue(all(abs(x - 0.5) < 1e-9 for x in w))

    def test_risk_parity_sums_to_one(self) -> None:
        px = _price_wide_for_weights()
        s = replace(get_settings(), portfolio_weighting="risk_parity")
        dt = px.index[50]
        w, lab = _weights_for_rebalance(px, list(px.columns), dt, s)
        self.assertEqual(lab, "risk_parity")
        self.assertEqual(len(w), 3)
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertTrue(all(x >= -1e-12 for x in w))

    def test_risk_parity_fallback_short_history(self) -> None:
        px = _price_wide_for_weights()
        s = replace(
            get_settings(),
            portfolio_weighting="risk_parity",
            optimizer_return_window=200,
            optimizer_min_obs=200,
        )
        dt = px.index[10]
        w, lab = _weights_for_rebalance(px, list(px.columns), dt, s)
        self.assertEqual(lab, "risk_parity_fallback")
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertTrue(all(abs(x - 1.0 / 3.0) < 1e-9 for x in w))


class TestRunSingleRiskParity(unittest.TestCase):
    def test_end_to_end_runs(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=120)
        syms = ["X", "Y"]
        rng = np.random.default_rng(3)
        px = pd.DataFrame(
            50.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, size=(len(days), len(syms))), axis=0),
            index=days,
            columns=syms,
        )
        idx = pd.MultiIndex.from_product([days, syms], names=["date", "symbol"])
        factor = pd.Series(rng.standard_normal(len(idx)), index=idx)
        settings = replace(get_settings(), portfolio_weighting="risk_parity", top_k=2)
        nav, meta = run_single_backtest(
            "TEST",
            factor_values=factor,
            prices=px,
            settings=settings,
            top_k=2,
        )
        self.assertGreater(len(nav), 0)
        self.assertEqual(meta.get("portfolio_weighting"), "risk_parity")
        log = meta.get("rebalance_log") or []
        self.assertGreater(len(log), 0)
        labs = {rec.get("weighting") for rec in log}
        self.assertTrue(labs <= {"risk_parity", "risk_parity_fallback", "equal"})


if __name__ == "__main__":
    unittest.main()
