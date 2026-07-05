import unittest
from dataclasses import replace

import pandas as pd

from analysis.factor_validation import (
    build_factor_decay_monitor,
    build_out_of_sample_validation,
    split_train_validation_dates,
)
from config import get_settings


class FactorValidationTests(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2024-01-01", periods=80)
        self.prices = pd.DataFrame(
            {
                "AAA": [10.0 + i * 0.20 for i in range(len(dates))],
                "BBB": [10.0 + i * 0.05 for i in range(len(dates))],
                "CCC": [20.0 - i * 0.05 for i in range(len(dates))],
                "DDD": [15.0 - i * 0.10 for i in range(len(dates))],
            },
            index=dates,
        )
        idx = pd.MultiIndex.from_product([dates, self.prices.columns], names=["date", "symbol"])
        good_vals = [4.0, 3.0, 2.0, 1.0] * len(dates)
        weak_vals = [1.0, 2.0, 3.0, 4.0] * len(dates)
        self.panel = pd.DataFrame({"GOOD": good_vals, "WEAK": weak_vals}, index=idx)

    def test_split_train_validation_dates(self):
        train, validation = split_train_validation_dates(self.panel, self.prices, train_ratio=0.5)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(validation), 0)
        self.assertLess(train[-1], validation[0])

    def test_build_out_of_sample_validation_and_monitor(self):
        settings = replace(get_settings(), rebalance_freq="D", top_k=1)
        validation = build_out_of_sample_validation(
            self.panel,
            self.prices,
            settings,
            factors=["GOOD", "WEAK"],
            train_ratio=0.5,
        )
        self.assertEqual(set(validation["factor"]), {"GOOD", "WEAK"})
        self.assertIn("validation_excess_ann_return", validation.columns)
        good = validation[validation["factor"] == "GOOD"].iloc[0]
        weak = validation[validation["factor"] == "WEAK"].iloc[0]
        self.assertGreater(float(good["validation_excess_ann_return"]), 0.0)
        self.assertLess(float(weak["validation_excess_ann_return"]), 0.0)

        monitor = build_factor_decay_monitor(validation)
        self.assertEqual(set(monitor["factor"]), {"GOOD", "WEAK"})
        weak_status = str(monitor[monitor["factor"] == "WEAK"]["status"].iloc[0])
        self.assertIn(weak_status, {"WATCH", "DEGRADED", "FAILED"})


if __name__ == "__main__":
    unittest.main()
