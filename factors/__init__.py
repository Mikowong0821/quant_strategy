"""
因子包。新增因子请在 FACTOR_REGISTRY 注册，供 backtest_single 按名称解析。
"""
from __future__ import annotations

from typing import Callable, Dict

from factors.factor_events import (
    ANNOUNCEMENT_EVENT_SCORE,
    ANNOUNCEMENT_EVENT_TYPE_PREFIX,
    calc_announcement_event_type_scores,
    calc_announcement_event_score,
    classify_announcement_event,
    load_announcement_events,
)
from factors.factor_finance import (
    calc_gross_margin,
    calc_low_debt_to_assets,
    calc_net_margin,
    calc_profit_growth,
    calc_revenue_growth,
)
from factors.factor_ml import ML_SCORE_NAME, build_ml_score_factor, forward_return_label
from factors.factor_momentum import calc_momentum
from factors.factor_news import (
    NEWS_FACTOR_NAMES,
    NEWS_HEAT_7D,
    NEWS_NEGATIVE_COUNT_7D,
    NEWS_NEGATIVE_RISK_SCORE,
    NEWS_SENTIMENT_DECAY,
    calc_news_sentiment_factors,
)
from factors.factor_pe import calc_pe
from factors.factor_reversal import calc_reversal
from factors.factor_roe import calc_roe
from factors.factor_volume import calc_volume_ratio
from factors.factor_volatility import calc_volatility
from factors.panel_builder import DEFAULT_FACTOR_ORDER, build_four_factor_panel

FACTOR_REGISTRY: Dict[str, Callable[..., object]] = {
    "MOMENTUM": calc_momentum,
    "MOMENTUM_60D": calc_momentum,
    "PE": calc_pe,
    "REVERSAL_5D": calc_reversal,
    "ROE": calc_roe,
    "VOLUME_RATIO_20D": calc_volume_ratio,
    "VOLATILITY": calc_volatility,
}

__all__ = [
    "FACTOR_REGISTRY",
    "DEFAULT_FACTOR_ORDER",
    "build_four_factor_panel",
    "ML_SCORE_NAME",
    "NEWS_FACTOR_NAMES",
    "NEWS_HEAT_7D",
    "NEWS_NEGATIVE_COUNT_7D",
    "NEWS_NEGATIVE_RISK_SCORE",
    "NEWS_SENTIMENT_DECAY",
    "ANNOUNCEMENT_EVENT_SCORE",
    "ANNOUNCEMENT_EVENT_TYPE_PREFIX",
    "build_ml_score_factor",
    "calc_news_sentiment_factors",
    "calc_announcement_event_type_scores",
    "calc_announcement_event_score",
    "classify_announcement_event",
    "calc_momentum",
    "calc_gross_margin",
    "calc_low_debt_to_assets",
    "calc_net_margin",
    "calc_profit_growth",
    "calc_revenue_growth",
    "calc_pe",
    "calc_reversal",
    "calc_roe",
    "calc_volume_ratio",
    "calc_volatility",
    "forward_return_label",
    "load_announcement_events",
]
