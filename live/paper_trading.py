"""
模拟交易：根据信号与行情更新虚拟账户。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def run_paper_trading(
    symbols: List[str],
    *,
    initial_cash: float = 1_000_000.0,
    signals: Optional[pd.Series] = None,
    prices: Optional[pd.DataFrame] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    :return: 成交与持仓日志，至少包含列: date, symbol, side, qty, price（骨架阶段可扩展）
    """
    raise NotImplementedError("撮合规则与回测一致性在实现阶段与 config 对齐")
