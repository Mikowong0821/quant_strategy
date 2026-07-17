import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.multi_universe_validation import (
    collect_factor_universe_performance,
    collect_strategy_universe_performance,
    save_multi_universe_validation_outputs,
    summarize_factor_universe_robustness,
    summarize_strategy_universe_robustness,
)


class MultiUniverseValidationTests(unittest.TestCase):
    def _write_universe(self, root: Path, name: str, strategy_excess: float, factor_excess: float) -> Path:
        base = root / name
        (base / "factor_diagnostics").mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "strategy": "BENCH_EQUAL_WEIGHT",
                    "final_nav": 1.1,
                    "ann_return": 0.10,
                    "ann_vol": 0.20,
                    "sharpe": 0.5,
                    "max_drawdown": -0.1,
                },
                {
                    "strategy": "FUSED_ROLLING_SCORE_WEIGHTED",
                    "final_nav": 1.2 + strategy_excess,
                    "ann_return": 0.15 + strategy_excess,
                    "ann_vol": 0.25,
                    "sharpe": 0.6,
                    "max_drawdown": -0.2,
                    "excess_ann_return": strategy_excess,
                    "information_ratio": strategy_excess / 0.1,
                    "avg_turnover": 0.8,
                    "avg_effective_n": 5.0,
                },
            ]
        ).to_csv(base / "performance_summary.csv", index=False)
        pd.DataFrame(
            [
                {
                    "factor": "GOOD",
                    "ann_return": 0.20 + factor_excess,
                    "excess_ann_return": factor_excess,
                    "tracking_error": 0.1,
                    "information_ratio": factor_excess / 0.1,
                    "n_rebalances": 3,
                }
            ]
        ).to_csv(base / "factor_diagnostics" / "long_excess_summary.csv", index=False)
        return base

    def test_collect_and_summarize_multi_universe_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            u1 = self._write_universe(root, "u1", 0.05, 0.10)
            u2 = self._write_universe(root, "u2", 0.02, 0.05)
            u3 = self._write_universe(root, "u3", -0.01, 0.02)

            universes = {"U1": u1, "U2": u2, "U3": u3}
            strategy_perf = collect_strategy_universe_performance(universes)
            self.assertEqual(set(strategy_perf["universe"]), {"U1", "U2", "U3"})
            strategy_robust = summarize_strategy_universe_robustness(strategy_perf)
            fused = strategy_robust[
                strategy_robust["strategy"] == "FUSED_ROLLING_SCORE_WEIGHTED"
            ].iloc[0]
            self.assertEqual(str(fused["status"]), "ROBUST")
            self.assertAlmostEqual(float(fused["positive_excess_rate"]), 2 / 3)

            factor_perf = collect_factor_universe_performance(universes)
            factor_robust = summarize_factor_universe_robustness(factor_perf)
            good = factor_robust[factor_robust["factor"] == "GOOD"].iloc[0]
            self.assertEqual(str(good["status"]), "ROBUST")

            paths = save_multi_universe_validation_outputs(
                root / "combined",
                strategy_performance=strategy_perf,
                strategy_robustness=strategy_robust,
                factor_performance=factor_perf,
                factor_robustness=factor_robust,
            )
            for path in paths.values():
                self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
