"""benchmark：股票池等权基准与超额指标自检。"""
from __future__ import annotations

import math
import unittest

import pandas as pd

from analysis.benchmark import (
    equal_weight_benchmark_nav,
    excess_nav,
    summarize_excess,
)


class TestBenchmark(unittest.TestCase):
    def test_equal_weight_benchmark_nav(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=3)
        prices = pd.DataFrame(
            {
                "AAA": [100.0, 110.0, 121.0],
                "BBB": [100.0, 100.0, 90.0],
            },
            index=days,
        )
        nav = equal_weight_benchmark_nav(prices)
        self.assertEqual(nav.name, "BENCH_EQUAL_WEIGHT")
        self.assertAlmostEqual(float(nav.iloc[0]), 1.0)
        self.assertAlmostEqual(float(nav.iloc[1]), 1.05)
        self.assertAlmostEqual(float(nav.iloc[2]), 1.05)

    def test_excess_summary(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=5)
        strategy = pd.Series([1.0, 1.02, 1.04, 1.06, 1.08], index=days)
        benchmark = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04], index=days)
        xnav = excess_nav(strategy, benchmark)
        stats = summarize_excess(strategy, benchmark, periods=252)
        self.assertGreater(float(xnav.iloc[-1]), 1.0)
        self.assertGreater(float(stats["excess_ann_return"]), 0.0)
        self.assertGreaterEqual(float(stats["tracking_error"]), 0.0)
        self.assertFalse(math.isnan(float(stats["benchmark_ann_return"])))


if __name__ == "__main__":
    unittest.main()
