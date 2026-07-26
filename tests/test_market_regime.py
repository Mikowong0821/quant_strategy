import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.market_regime import (
    build_market_regime_frame,
    save_market_regime_outputs,
    strategy_regime_performance,
    summarize_regime_days,
    summarize_strategy_regime_robustness,
)


class MarketRegimeTests(unittest.TestCase):
    def test_market_regime_performance_and_save(self):
        dates = pd.bdate_range("2024-01-01", periods=90)
        # 先涨、再跌、后震荡，让三类状态都有机会出现。
        bench_values = (
            [1.0 + i * 0.01 for i in range(30)]
            + [1.30 - i * 0.012 for i in range(30)]
            + [0.94 + ((i % 2) * 0.005) for i in range(30)]
        )
        benchmark = pd.Series(bench_values, index=dates, name="BENCH_EQUAL_WEIGHT")
        strategy = benchmark * pd.Series(
            [1.0 + i * 0.001 for i in range(len(dates))],
            index=dates,
        )

        regimes = build_market_regime_frame(
            benchmark,
            lookback_days=10,
            bull_return_threshold=0.05,
            bear_return_threshold=-0.05,
            bear_drawdown_threshold=-0.10,
        )
        self.assertIn("regime", regimes.columns)
        self.assertIn("BULL", set(regimes["regime"]))
        self.assertIn("BEAR", set(regimes["regime"]))

        days = summarize_regime_days(regimes)
        self.assertFalse(days.empty)
        self.assertIn("day_ratio", days.columns)

        detail = strategy_regime_performance(
            {"TEST": strategy},
            benchmark,
            regimes,
            periods=252,
        )
        self.assertFalse(detail.empty)
        self.assertIn("excess_ann_return", detail.columns)

        summary = summarize_strategy_regime_robustness(detail)
        self.assertFalse(summary.empty)
        self.assertIn("status", summary.columns)

        with tempfile.TemporaryDirectory() as tmp:
            paths = save_market_regime_outputs(Path(tmp), regimes, days, detail, summary)
            self.assertTrue(paths["regime_frame"].exists())
            self.assertTrue(paths["strategy_regime_summary"].exists())


if __name__ == "__main__":
    unittest.main()
