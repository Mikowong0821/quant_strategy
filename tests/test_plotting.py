"""plotting：IC / 权重表转换与无界面作图烟测。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.plotting import (
    plot_factor_coverage,
    plot_ic,
    plot_turnover,
    plot_weights,
    rebalance_log_to_weights_frame,
)


class TestRebalanceLogToWeights(unittest.TestCase):
    def test_wide_table(self) -> None:
        log = [
            {
                "date": pd.Timestamp("2024-01-31"),
                "picks": ["AAA", "BBB"],
                "weights": [0.6, 0.4],
                "weighting": "equal",
            },
            {
                "date": pd.Timestamp("2024-02-29"),
                "picks": ["BBB", "CCC"],
                "weights": [0.5, 0.5],
                "weighting": "equal",
            },
        ]
        w = rebalance_log_to_weights_frame(log)
        self.assertEqual(list(w.columns), ["AAA", "BBB", "CCC"])
        self.assertAlmostEqual(float(w.loc[w.index[0], "AAA"]), 0.6)
        self.assertTrue(pd.isna(w.loc[w.index[0], "CCC"]))
        self.assertAlmostEqual(float(w.loc[w.index[1], "CCC"]), 0.5)


class TestPlotSmoke(unittest.TestCase):
    def test_plot_ic_and_weights_files(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=40)
        ic = pd.Series(
            [0.02, -0.01, 0.03] * 13 + [0.01],
            index=days,
            name="ic",
        )
        w = pd.DataFrame(
            [[0.5, 0.5, float("nan")], [0.2, 0.3, 0.5]],
            index=days[[10, 20]],
            columns=["A", "B", "C"],
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            plot_factor_coverage(
                pd.DataFrame({"factor": ["A", "B"], "coverage": [0.8, 0.5]}),
                title="coverage",
                save_path=p / "coverage.png",
            )
            plot_ic(ic, title="t", save_path=p / "ic.png", rolling_window=5)
            plot_turnover(
                pd.DataFrame({"A": [1.0, 0.4], "B": [0.8, 0.7]}, index=days[[10, 20]]),
                title="turnover",
                save_path=p / "turnover.png",
            )
            plot_weights(w, title="w", save_path=p / "w_area.png", kind="area")
            plot_weights(w, title="w", save_path=p / "w_hm.png", kind="heatmap")
            self.assertTrue((p / "coverage.png").is_file())
            self.assertTrue((p / "ic.png").is_file())
            self.assertTrue((p / "turnover.png").is_file())
            self.assertTrue((p / "w_area.png").is_file())
            self.assertTrue((p / "w_hm.png").is_file())


if __name__ == "__main__":
    unittest.main()
