"""models.fusion：IC 加权融合最小切片。"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from models.fusion import (
    fuse_equal_weight_zscore,
    fuse_ic_weighted_zscore,
    fuse_static_weight_zscore,
)


def _tiny_panel_and_ic() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    days = pd.bdate_range("2024-01-01", periods=30)
    syms = ["A", "B"]
    idx = pd.MultiIndex.from_product([days, syms], names=["date", "symbol"])
    rng = np.random.default_rng(0)
    f1 = pd.Series(rng.normal(0, 1, len(idx)), index=idx, name="F1")
    f2 = pd.Series(rng.normal(0, 1, len(idx)), index=idx, name="F2")
    panel = pd.concat([f1, f2], axis=1)
    panel.columns = ["FAC_A", "FAC_B"]
    # IC：FAC_A 长期略正，FAC_B 略负
    ic_a = pd.Series(np.linspace(0.05, 0.15, len(days)), index=days, name="ic")
    ic_b = pd.Series(np.linspace(-0.08, -0.02, len(days)), index=days, name="ic")
    ic_by = {"FAC_A": ic_a, "FAC_B": ic_b}
    return panel, ic_by


class TestFuseIcWeighted(unittest.TestCase):
    def test_runs_and_differs_from_equal(self) -> None:
        panel, ic_by = _tiny_panel_and_ic()
        s_ic = fuse_ic_weighted_zscore(panel, ic_by, rolling_window=5, min_periods=2)
        s_eq = fuse_equal_weight_zscore(panel)
        self.assertEqual(s_ic.name, "fused_zscore_ic_weighted")
        self.assertEqual(len(s_ic), len(s_eq))
        self.assertFalse(np.allclose(s_ic.values, s_eq.values, equal_nan=True))

    def test_missing_ic_key_raises(self) -> None:
        panel, ic_by = _tiny_panel_and_ic()
        del ic_by["FAC_B"]
        with self.assertRaises(KeyError):
            fuse_ic_weighted_zscore(panel, ic_by, rolling_window=5, min_periods=2)


class TestFuseStaticWeight(unittest.TestCase):
    def test_static_weights_change_fusion(self) -> None:
        panel, _ = _tiny_panel_and_ic()
        s = fuse_static_weight_zscore(panel, {"FAC_A": 1.0, "FAC_B": 0.0})
        s_eq = fuse_equal_weight_zscore(panel)
        self.assertEqual(s.name, "fused_zscore_static_weighted")
        self.assertEqual(len(s), len(s_eq))
        self.assertFalse(np.allclose(s.values, s_eq.values, equal_nan=True))

    def test_invalid_weights_fall_back_to_equal(self) -> None:
        panel, _ = _tiny_panel_and_ic()
        s = fuse_static_weight_zscore(panel, {"FAC_A": 0.0, "FAC_B": float("nan")})
        s_eq = fuse_equal_weight_zscore(panel)
        self.assertTrue(np.allclose(s.values, s_eq.values, equal_nan=True))


if __name__ == "__main__":
    unittest.main()
