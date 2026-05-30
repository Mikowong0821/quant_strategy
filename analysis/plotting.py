"""
净值、IC、权重等图表。

说明（与回测 / IC 模块的衔接）：
- **IC 图**：`daily_ic_spearman` 产出的是「每个交易日截面上因子与前瞻收益的 Spearman 相关系数」日序列。
  横轴为交易日，纵轴为 IC。IC 在 0 上方居多表示因子与未来收益同向；若剧烈摆动或长期贴 0，提示因子不稳定或噪音大。
  单条 `Series` 时可叠加 **滚动均值**（默认约一月，20 个交易日），便于看趋势；多条因子对比时用 `DataFrame` 多曲线，默认不叠滚动线以免太乱。
- **权重图**：输入为「再平衡日 × 标的」宽表：行索引为调仓日，列为 `ts_code`，值为该日目标权重（未持仓为 NaN）。
  可用 `rebalance_log_to_weights_frame(meta["rebalance_log"])` 从 `run_single_backtest` 的 meta 生成。
  **堆叠面积图**（默认）在每期将权重视为资金占比，未入选标的按 0，每期堆叠高度为 1；适合 Top-K 较小。
  **热力图**可选，适合标的较多时看稀疏持仓模式。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import matplotlib

import numpy as np
import pandas as pd

NavInput = Union[pd.Series, pd.DataFrame]
IcInput = Union[pd.Series, pd.DataFrame]
WeightsKind = Literal["area", "heatmap"]


def _pyplot_zh(save_path: Optional[Path] = None):
    """无显示器时若需写文件则启用 Agg，并设置中文与负号显示。"""
    if save_path is not None:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_nav(
    nav: NavInput,
    *,
    title: str = "净值曲线",
    save_path: Optional[Path] = None,
    normalize: bool = True,
    figsize: tuple[float, float] = (11, 5.5),
) -> None:
    """
    绘制一条或多条净值曲线。

    :param nav: 单日频净值 `Series`，或多列 `DataFrame`（每列一条曲线，索引为日期）。
    :param title: 图标题。
    :param save_path: 若给定则保存为 PNG（使用 Agg，适合无显示器环境）。
    :param normalize: True 时各列除以各自首行有效值，使起点约为 1，便于对比形态。
    """
    plt = _pyplot_zh(save_path)

    if isinstance(nav, pd.Series):
        df = nav.astype(float).to_frame(name=getattr(nav, "name", None) or "nav")
    else:
        df = nav.astype(float).copy()

    df = df.sort_index()
    if normalize:
        base = df.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else float("nan"))
        df = df.div(base, axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    for col in df.columns:
        ax.plot(df.index, df[col].values, label=str(col), linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("日期")
    ax.set_ylabel("归一化净值" if normalize else "净值")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_turnover(
    turnover: pd.DataFrame,
    *,
    title: str = "换手率对比",
    save_path: Optional[Path] = None,
    figsize: tuple[float, float] = (11, 4.5),
) -> None:
    """
    绘制多条策略的逐期换手率。

    :param turnover: 行为调仓日、列为策略名、值为 turnover（成交金额 / 组合净值）。
    """
    plt = _pyplot_zh(save_path)

    df = turnover.sort_index().astype(float)
    if df.empty or df.shape[1] == 0:
        raise ValueError("plot_turnover: turnover 为空或无列")

    fig, ax = plt.subplots(figsize=figsize)
    for col in df.columns:
        ax.plot(df.index, df[col].values, marker="o", markersize=3, linewidth=1.2, label=str(col))
    ax.set_title(title)
    ax.set_xlabel("调仓日")
    ax.set_ylabel("换手率（成交金额 / 净值）")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_ic(
    ic: IcInput,
    *,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    rolling_window: Optional[int] = 20,
    figsize: tuple[float, float] = (11, 4.5),
) -> None:
    """
    绘制日频 IC 时间序列（来自 `analysis.ic.daily_ic_spearman` 等）。

    **如何读图**：Spearman IC 衡量当日因子排序与「未来 h 日收益」排序的一致性；序列在 0 轴上下波动属正常。
    若叠加了滚动均值（仅对单列 `Series` 且 `rolling_window` 非空），粗线为短期噪声平滑后的方向，便于与 `summarize_ic` 中的 mean_IC、IC_IR 对照。

    :param ic: 单列日频 IC（`Series`，索引为 date），或多因子对比（`DataFrame`，每列一条曲线）。
    :param title: 图标题；默认按列名拼接。
    :param save_path: 若给定则保存 PNG。
    :param rolling_window: 对 **单列** IC 计算滚动均值并叠画（虚线）；`None` 关闭。多列 `DataFrame` 时忽略。
    """
    plt = _pyplot_zh(save_path)

    if isinstance(ic, pd.Series):
        df = ic.astype(float).to_frame(name=getattr(ic, "name", None) or "ic")
        use_rolling = rolling_window is not None and int(rolling_window) > 1
        rw = int(rolling_window) if use_rolling else 0
    else:
        df = ic.astype(float).copy()
        use_rolling = False
        rw = 0

    df = df.sort_index()
    if df.shape[1] == 0 or len(df) == 0:
        raise ValueError("plot_ic: ic 无有效列或长度为 0")

    if title is None:
        title = "日 IC（Spearman）" if df.shape[1] == 1 else "日 IC 对比（Spearman）"

    fig, ax = plt.subplots(figsize=figsize)
    for col in df.columns:
        y = df[col].values
        ax.plot(df.index, y, label=str(col), linewidth=1.0, alpha=0.85)
        if use_rolling and df.shape[1] == 1 and rw > 1:
            roll = df[col].rolling(window=rw, min_periods=max(2, rw // 2)).mean()
            ax.plot(
                df.index,
                roll.values,
                color="C0",
                linestyle="--",
                linewidth=1.4,
                label="滚动均值(%d日)" % rw,
                alpha=0.9,
            )
    ax.axhline(0.0, color="gray", linewidth=0.9, linestyle="-", alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("日期")
    ax.set_ylabel("IC")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def rebalance_log_to_weights_frame(rebalance_log: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """
    将 `run_single_backtest` / `run_multi_backtest` 返回的 `meta["rebalance_log"]` 转为权重宽表。

    每行对应一次再平衡：仅当期入选标的列有权重（和为 1），其余标的为 NaN，便于 `plot_weights` 堆叠或热力图。

    :param rebalance_log: 元素含 `date`、`picks`、`weights`（与 picks 等长）的记录列表。
    :return: 索引为 `DatetimeIndex`，列为出现过的全部 `ts_code`。
    """
    if not rebalance_log:
        return pd.DataFrame()
    syms: set[str] = set()
    for rec in rebalance_log:
        for p in rec.get("picks") or []:
            syms.add(str(p))
    col_order = sorted(syms)
    rows: list[dict[str, float]] = []
    idx: list[pd.Timestamp] = []
    for rec in sorted(rebalance_log, key=lambda r: pd.Timestamp(r["date"])):
        dt = pd.Timestamp(rec["date"])
        picks = list(rec.get("picks") or [])
        wts = list(rec.get("weights") or [])
        row = {c: float("nan") for c in col_order}
        for i, sym in enumerate(picks):
            if sym not in row:
                continue
            if i < len(wts):
                row[str(sym)] = float(wts[i])
        rows.append(row)
        idx.append(dt)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="date"))


def plot_weights(
    weights: pd.DataFrame,
    *,
    title: str = "再平衡权重",
    save_path: Optional[Path] = None,
    kind: WeightsKind = "area",
    figsize: Optional[tuple[float, float]] = None,
) -> None:
    """
    绘制再平衡权重随时间变化。

    **输入形态**：行 = 再平衡日，列 = 标的，值为当期目标权重（未持仓为 NaN）。通常由 `rebalance_log_to_weights_frame` 生成。

    **area（堆叠面积）**：每期将 NaN 视为 0，堆叠高度为当期各标的权重之和（调仓记录里每期应为 1）。颜色区分标的，便于看集中度是否漂移。

    **heatmap**：行为时间、列为标的，颜色深浅为权重；标的很多时比堆叠更易扫一眼。

    :param weights: 宽表；空表将抛出 `ValueError`。
    :param title: 图标题。
    :param save_path: 若给定则保存 PNG。
    :param kind: `area` 或 `heatmap`。
    :param figsize: 默认随列数略调宽。
    """
    plt = _pyplot_zh(save_path)

    df = weights.sort_index().astype(float)
    if df.empty or df.shape[1] == 0:
        raise ValueError("plot_weights: weights 为空或无列")

    ncols = df.shape[1]
    if figsize is None:
        figsize = (max(10.0, min(22.0, 6.0 + len(df) * 0.15)), max(4.0, min(10.0, 2.5 + ncols * 0.22)))

    fig, ax = plt.subplots(figsize=figsize)

    if kind == "area":
        df0 = df.fillna(0.0)
        layers = [df0[c].values for c in df.columns]
        ax.stackplot(
            df.index,
            *layers,
            labels=[str(c) for c in df.columns],
            alpha=0.85,
        )
        ax.set_ylabel("权重（堆叠和为 1）")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, ncol=1)
        ax.set_ylim(0.0, 1.0)
    else:
        # 热力图：行为时间步，列为标的；权重 NaN 画成白色
        mat = df.T.fillna(0.0).values
        im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0.0, vmax=max(0.01, float(np.nanmax(mat))))
        ax.set_yticks(np.arange(len(df.columns)))
        ax.set_yticklabels([str(c) for c in df.columns], fontsize=7)
        step = max(1, len(df) // 12)
        xt = np.arange(0, len(df), step)
        ax.set_xticks(xt)
        ax.set_xticklabels([df.index[i].strftime("%Y-%m-%d") for i in xt], rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("再平衡日（序号采样）")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="权重")

    ax.set_title(title)
    if kind == "area":
        ax.set_xlabel("再平衡日")
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
