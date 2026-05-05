"""
因子包。新增因子请在 FACTOR_REGISTRY 注册，供 backtest_single 按名称解析。
"""
from __future__ import annotations

from typing import Callable, Dict

from factors.factor_momentum import calc_momentum
from factors.factor_pe import calc_pe
from factors.factor_roe import calc_roe
from factors.factor_volatility import calc_volatility
from factors.panel_builder import DEFAULT_FACTOR_ORDER, build_four_factor_panel

FACTOR_REGISTRY: Dict[str, Callable[..., object]] = {
    "MOMENTUM": calc_momentum,
    "PE": calc_pe,
    "ROE": calc_roe,
    "VOLATILITY": calc_volatility,
}

__all__ = [
    "FACTOR_REGISTRY",
    "DEFAULT_FACTOR_ORDER",
    "build_four_factor_panel",
    "calc_momentum",
    "calc_pe",
    "calc_roe",
    "calc_volatility",
]
