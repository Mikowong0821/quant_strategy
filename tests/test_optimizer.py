"""models.optimizer 数值与契约自检（unittest，无 pytest 依赖）。"""
from __future__ import annotations

import unittest

import numpy as np

from models.optimizer import maximize_sharpe, risk_parity


class TestOptimizer(unittest.TestCase):
    def test_maximize_sharpe_single_asset(self) -> None:
        w = maximize_sharpe(np.array([0.03]), np.array([[0.01]]))
        self.assertEqual(w.shape, (1,))
        self.assertAlmostEqual(float(w[0]), 1.0)

    def test_maximize_sharpe_two_assets_long_only(self) -> None:
        mu = np.array([0.20, 0.05])
        cov = np.array([[0.04, 0.0], [0.0, 0.01]])
        w = maximize_sharpe(mu, cov, risk_free=0.0)
        self.assertAlmostEqual(float(np.sum(w)), 1.0)
        self.assertTrue(np.all(w >= -1e-6))

        def _sharpe(x: np.ndarray) -> float:
            v = float(x @ cov @ x)
            return float((mu @ x) / np.sqrt(max(v, 1e-18)))

        s_opt = _sharpe(w)
        self.assertGreaterEqual(s_opt + 1e-9, _sharpe(np.array([1.0, 0.0])))
        self.assertGreaterEqual(s_opt + 1e-9, _sharpe(np.array([0.0, 1.0])))

    def test_risk_parity_diagonal_equal_risk_contrib(self) -> None:
        cov = np.diag([0.04, 0.01]).astype(float)
        w = risk_parity(cov)
        self.assertAlmostEqual(float(np.sum(w)), 1.0)
        self.assertAlmostEqual(float(w[0]), 1.0 / 3.0, delta=0.02)
        self.assertAlmostEqual(float(w[1]), 2.0 / 3.0, delta=0.02)
        sp = float(np.sqrt(w @ cov @ w))
        sw = cov @ w
        rc = w * sw / sp
        self.assertAlmostEqual(float(rc[0]), float(rc[1]), delta=max(0.05 * abs(rc[0]), 1e-6))

    def test_maximize_sharpe_zero_excess_returns_equal_weight(self) -> None:
        mu = np.array([0.02, 0.02])
        cov = np.eye(2) * 0.01
        w = maximize_sharpe(mu, cov, risk_free=0.02)
        self.assertAlmostEqual(float(w[0]), float(w[1]), delta=0.02)


if __name__ == "__main__":
    unittest.main()
