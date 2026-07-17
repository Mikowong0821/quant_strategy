import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.parameter_sensitivity import (
    build_one_way_parameter_variants,
    run_parameter_sensitivity,
    save_parameter_sensitivity_outputs,
    summarize_parameter_sensitivity,
)
from config import get_settings


class ParameterSensitivityTests(unittest.TestCase):
    def _fixtures(self):
        dates = pd.bdate_range("2024-01-01", periods=90)
        symbols = ["AAA", "BBB", "CCC", "DDD"]
        prices = pd.DataFrame(index=dates, columns=symbols, dtype=float)
        for i, sym in enumerate(symbols):
            prices[sym] = 10.0 + i + np.linspace(0, 8 + i, len(dates))

        rows = []
        for dt in dates:
            for i, sym in enumerate(symbols):
                rows.append({"date": dt, "symbol": sym, "TEST": float(i)})
        panel = pd.DataFrame(rows).set_index(["date", "symbol"])
        return prices, panel["TEST"]

    def test_run_and_save_parameter_sensitivity(self):
        prices, factor_values = self._fixtures()
        settings = replace(
            get_settings(),
            rebalance_freq="20B",
            top_k=2,
            portfolio_weighting="equal",
            max_position_weight=0.7,
            max_rebalance_turnover=1.0,
            target_volatility=0.0,
            min_positions=0,
        )
        variants = build_one_way_parameter_variants(
            settings,
            {
                "top_k": [1, 2, 3],
                "portfolio_weighting": ["equal", "risk_parity"],
                "max_rebalance_turnover": [0.5, 1.0],
            },
        )
        self.assertEqual(variants[0]["variant"], "baseline")
        self.assertTrue(any(v["changed_parameter"] == "top_k" for v in variants))

        detail = run_parameter_sensitivity(
            prices=prices,
            factor_values=factor_values,
            base_settings=settings,
            variants=variants,
            factor_name="TEST",
        )
        self.assertFalse(detail.empty)
        self.assertIn("excess_ann_return", detail.columns)
        self.assertIn("baseline", set(detail["variant"]))

        summary = summarize_parameter_sensitivity(detail)
        self.assertFalse(summary.empty)
        self.assertIn("status", summary.columns)

        with tempfile.TemporaryDirectory() as tmp:
            paths = save_parameter_sensitivity_outputs(Path(tmp), detail, summary)
            self.assertTrue(paths["detail"].exists())
            self.assertTrue(paths["summary"].exists())


if __name__ == "__main__":
    unittest.main()
