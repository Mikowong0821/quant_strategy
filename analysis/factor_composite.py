"""
因子分层与复合因子。

原始因子先经过准入与冗余处理，再按风格层合成为少数几个复合分数。
这一步不删除原始因子，只是让主融合层面对更清晰的风格信号。
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import pandas as pd


DEFAULT_STYLE_GROUPS: "OrderedDict[str, list[str]]" = OrderedDict(
    [
        (
            "PRICE_VOLUME_STYLE",
            ["MOMENTUM", "MOMENTUM_60D", "REVERSAL_5D", "VOLATILITY", "VOLUME_RATIO_20D"],
        ),
        ("VALUE_STYLE", ["PE"]),
        ("QUALITY_STYLE", ["ROE", "GROSS_MARGIN", "NET_MARGIN", "LOW_DEBT_TO_ASSETS"]),
        ("GROWTH_STYLE", ["REVENUE_GROWTH", "PROFIT_GROWTH"]),
        ("CASHFLOW_STYLE", ["FREE_CASH_FLOW_YIELD", "CASH_PROFIT_QUALITY"]),
        ("ML_STYLE", ["ML_SCORE"]),
    ]
)


COMPOSITE_COMPONENT_COLUMNS = [
    "composite_factor",
    "style",
    "candidate_factors",
    "eligible_components",
    "missing_candidates",
    "n_components",
    "coverage",
    "valid_cells",
    "total_cells",
]


def _ensure_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise TypeError("panel 须为二级 MultiIndex(date, symbol)")
    out = panel.copy()
    out.index = out.index.set_names(["date", "symbol"])
    return out.sort_index()


def build_factor_composite_scores(
    panel: pd.DataFrame,
    *,
    eligible_factors: list[str] | None = None,
    style_groups: Mapping[str, list[str]] | None = None,
    min_components: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    生成风格复合因子面板。

    :param panel: 已标准化的因子面板，索引为 (date, symbol)，列为原始因子。
    :param eligible_factors: 允许进入复合层的因子，通常来自准入 + 去冗余后的候选池。
    :param style_groups: 风格分层定义，key 为复合因子名，value 为候选原始因子列表。
    :param min_components: 单个样本至少有多少个有效组件才计算复合分数。
    :return: (复合因子面板, 复合因子构成表)
    """
    p = _ensure_panel(panel)
    groups = style_groups or DEFAULT_STYLE_GROUPS
    eligible = set(str(x) for x in eligible_factors) if eligible_factors is not None else set(p.columns.astype(str))

    composite_cols: dict[str, pd.Series] = {}
    rows: list[dict[str, object]] = []
    total_cells = int(len(p.index))
    min_count = max(1, int(min_components))

    for composite_name, candidates in groups.items():
        candidate_list = [str(x) for x in candidates]
        components = [f for f in candidate_list if f in p.columns and f in eligible]
        missing = [f for f in candidate_list if f not in components]
        if not components:
            continue

        raw = p[components]
        valid_count = raw.notna().sum(axis=1)
        score = raw.mean(axis=1, skipna=True).where(valid_count >= min_count)
        score.name = str(composite_name)
        composite_cols[str(composite_name)] = score

        valid_cells = int(score.notna().sum())
        rows.append(
            {
                "composite_factor": str(composite_name),
                "style": str(composite_name).replace("_STYLE", "").lower(),
                "candidate_factors": ",".join(candidate_list),
                "eligible_components": ",".join(components),
                "missing_candidates": ",".join(missing),
                "n_components": int(len(components)),
                "coverage": float(valid_cells / total_cells) if total_cells else 0.0,
                "valid_cells": valid_cells,
                "total_cells": total_cells,
            }
        )

    if not composite_cols:
        return pd.DataFrame(index=p.index), pd.DataFrame(columns=COMPOSITE_COMPONENT_COLUMNS)

    composite_panel = pd.DataFrame(composite_cols, index=p.index).sort_index()
    component_summary = pd.DataFrame(rows)
    for col in COMPOSITE_COMPONENT_COLUMNS:
        if col not in component_summary.columns:
            component_summary[col] = ""
    return composite_panel, component_summary[COMPOSITE_COMPONENT_COLUMNS].reset_index(drop=True)
