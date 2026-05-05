"""
截面 IC：因子在交易日 t 的取值 vs 前瞻收盘收益（避免与「仅用 ≤t 信息」的因子重叠同一根价格 bar）。

工程约定（与机构常见口径一致，可改 `config.ic_forward_days`）：
- 前瞻收益 r：close(t+h)/close(t) - 1，h 默认 1（亦可 5、20 等）；若策略为月度调仓，可另选
  「持有至下次调仓日收益」以与回测对齐，本模块当前实现固定 h 日。
- 信息集：因子列在 (t, symbol) 处仅使用 ≤t 已可得数据（见各因子实现）；IC 用 t 日因子对齐
  从 t 到 t+h 的收益，收益起点为 t 收盘、终点为 t+h 收盘，不包含 t 之后才能知道的因子修订。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from config import Settings


def forward_returns_long(
    prices_wide: pd.DataFrame,
    *,
    forward_days: int = 1,
) -> pd.Series:
    """
    宽表收盘价 -> 长表前瞻简单收益：r_{t,s} = close_{t+h}/close_{t,s} - 1。
    """
    if forward_days < 1:
        raise ValueError("forward_days 须 >= 1")
    px = prices_wide.sort_index().sort_index(axis=1).astype(float)
    ret = px.shift(-forward_days) / px - 1.0
    s = ret.stack()
    s.index.names = ["date", "symbol"]
    return s


def daily_ic_spearman(
    factor: pd.Series,
    prices_wide: pd.DataFrame,
    *,
    forward_days: int = 1,
    min_names: int = 3,
) -> pd.Series:
    """
    每个交易日截面上，因子与未来收益的 Spearman 相关系数（秩相关，对极端值更稳健）。

    :param factor: MultiIndex (date, symbol)，与 prices 列名（标的）一致。
    :param min_names: 当日有效 (因子, 收益) 对少于该数则当日 IC 记为 NaN。
    :return: 日频 Series，索引为 date，名称为 ic。
    """
    if not isinstance(factor.index, pd.MultiIndex) or factor.index.nlevels != 2:
        raise TypeError("factor 须为二级 MultiIndex (date, symbol)")
    fac = factor.astype(float).copy()
    fac.index = fac.index.set_names(["date", "symbol"])
    fwd = forward_returns_long(prices_wide, forward_days=forward_days)
    df = pd.concat([fac.rename("f"), fwd.rename("r")], axis=1, join="inner")
    out: dict[Any, float] = {}
    for dt, g in df.groupby(level=0, sort=True):
        gg = g[["f", "r"]].dropna(how="any")
        if len(gg) < min_names:
            out[dt] = np.nan
            continue
        out[dt] = float(gg["f"].corr(gg["r"], method="spearman"))
    ser = pd.Series(out, name="ic")
    ser.index = pd.to_datetime(ser.index)
    ser = ser.sort_index()
    return ser


def summarize_ic(ic: pd.Series) -> Dict[str, Any]:
    """均值、标准差、IC_IR（均值/标准差）、胜率、有效日数。"""
    v = ic.dropna().astype(float)
    if v.empty:
        return {
            "mean_ic": np.nan,
            "std_ic": np.nan,
            "ic_ir": np.nan,
            "hit_rate": np.nan,
            "n_days": 0,
        }
    m = float(v.mean())
    s = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    ir = (m / s) if s > 0 else np.nan
    hr = float((v > 0).mean())
    return {
        "mean_ic": m,
        "std_ic": s,
        "ic_ir": ir,
        "hit_rate": hr,
        "n_days": int(len(v)),
    }


def save_ic_series(settings: Settings, ic_by_name: Dict[str, pd.Series]) -> Dict[str, Path]:
    """将各因子/融合的日 IC 写入 output/cache/ic_<name>.csv。"""
    base = settings.output_dir / "cache"
    base.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for name, ser in ic_by_name.items():
        safe = name.replace("/", "_")
        p = base / ("ic_%s.csv" % safe)
        ser.to_csv(p, date_format="%Y-%m-%d", header=True)
        out[name] = p
    return out
