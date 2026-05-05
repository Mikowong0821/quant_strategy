"""
组合优化：夏普最大化、风险平价（ERC）。
输入 `mu`、`cov` 与 symbol 顺序必须一致（契约见 docs/INTERFACE_AND_CONTRACTS.md）。

实现说明：
- `maximize_sharpe`：在 sum(w)=1 与 `bounds` 下最小化负夏普（SLSQP）；失败时回退等权。
- `risk_parity`：最小化各资产风险贡献与「平均贡献」之差的平方和（SLSQP）；失败时回退逆波动率权重。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize


def _ridge_cov(cov: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """对协方差阵加微量对角岭，减轻数值病态。"""
    c = np.asarray(cov, dtype=float)
    n = c.shape[0]
    tr = float(np.trace(c))
    if tr <= 0.0 or not np.isfinite(tr):
        tr = 1.0
    return c + np.eye(n, dtype=float) * ridge * (tr / max(n, 1))


def maximize_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    *,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    risk_free: float = 0.0,
) -> np.ndarray:
    """
    在约束下最大化夏普比率 (mu - rf)' w / sqrt(w' cov w)。

    :param mu: 预期收益向量 (n,)（若已是超额收益，可将 risk_free 置 0）
    :param cov: 协方差矩阵 (n, n)
    :param bounds: (lower, upper)，各为长度 n 的数组；默认 [0,1] 表示仅做多、全额配置
    :param risk_free: 无风险利率（标量），从 mu 中扣除
    :return: 权重 (n,)，归一化 sum(w)=1
    """
    mu = np.asarray(mu, dtype=float).ravel()
    cov = np.asarray(cov, dtype=float)
    n = mu.size
    if cov.shape != (n, n):
        raise ValueError("cov 须为 (n, n)，且 n 与 mu 长度一致")
    if n == 0:
        raise ValueError("资产数为 0")
    if n == 1:
        return np.ones(1, dtype=float)

    cov = _ridge_cov(cov)
    excess = mu - float(risk_free)
    if np.allclose(excess, 0.0):
        return np.ones(n, dtype=float) / n

    def neg_sharpe(w: np.ndarray) -> float:
        mw = float(excess @ w)
        v = float(w @ cov @ w)
        if v <= 1e-20 or not np.isfinite(v):
            return 1e12
        return -mw / np.sqrt(v)

    cons = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    if bounds is None:
        bnds = tuple((0.0, 1.0) for _ in range(n))
    else:
        lo, hi = np.asarray(bounds[0], dtype=float).ravel(), np.asarray(bounds[1], dtype=float).ravel()
        if lo.size != n or hi.size != n:
            raise ValueError("bounds[0]、bounds[1] 长度须为 n")
        bnds = tuple((float(lo[i]), float(hi[i])) for i in range(n))

    x0 = np.ones(n, dtype=float) / n
    res = minimize(
        neg_sharpe,
        x0,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = np.asarray(res.x, dtype=float)
    if bounds is None or float(np.min(np.asarray(bounds[0], dtype=float).ravel())) >= 0.0:
        w = np.clip(w, 0.0, None)
    s = float(np.sum(w))
    if s <= 1e-12 or not np.all(np.isfinite(w)):
        return np.ones(n, dtype=float) / n
    w = w / s
    if not res.success:
        return np.ones(n, dtype=float) / n
    return w


def risk_parity(
    cov: np.ndarray,
    *,
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> np.ndarray:
    """
    风险平价（等风险贡献 ERC）：使各资产风险贡献 RC_i 尽可能接近 sigma_p / n。

    RC_i = w_i (cov w)_i / sqrt(w' cov w)；满足理想 ERC 时 RC_i 均等于组合波动除以 n。

    :param cov: 协方差矩阵 (n, n)
    :param tol: 传给 SLSQP 的 ftol
    :param max_iter: 最大迭代次数（映射到 SLSQP maxiter）
    :return: 权重 (n,)，和为 1、非负
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if cov.shape != (n, n):
        raise ValueError("cov 须为方阵")
    if n == 0:
        raise ValueError("资产数为 0")
    if n == 1:
        return np.ones(1, dtype=float)

    cov = _ridge_cov(cov)

    def portfolio_vol(w: np.ndarray) -> float:
        v = float(w @ cov @ w)
        return float(np.sqrt(max(v, 1e-20)))

    def erc_objective(w: np.ndarray) -> float:
        sp = portfolio_vol(w)
        sw = cov @ w
        rc = w * sw / sp
        target = sp / max(n, 1)
        d = rc - target
        return float(d @ d)

    cons = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)
    bnds = tuple((1e-10, 1.0) for _ in range(n))
    x0 = np.ones(n, dtype=float) / n
    res = minimize(
        erc_objective,
        x0,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"maxiter": int(max_iter), "ftol": float(tol)},
    )
    w = np.maximum(np.asarray(res.x, dtype=float), 0.0)
    s = float(np.sum(w))
    if s <= 1e-12 or not np.all(np.isfinite(w)):
        return _inv_vol_weights(cov)
    w = w / s
    if not res.success:
        return _inv_vol_weights(cov)
    return w


def _inv_vol_weights(cov: np.ndarray) -> np.ndarray:
    """对角波动率的逆波动率权重（对角协方差下的解析风险平价）。"""
    d = np.clip(np.diag(cov), 1e-18, None)
    iv = 1.0 / np.sqrt(d)
    s = float(iv.sum())
    if s <= 0.0:
        n = cov.shape[0]
        return np.ones(n, dtype=float) / max(n, 1)
    return iv / s
