"""
单因子回测。契约：输入因子名或预计算 PanelLong，输出 NavSeries + 元信息。

月末调仓、Top-K 多头、收盘价成交、单边手续费；持仓权重见 config.portfolio_weighting（equal / max_sharpe / risk_parity）。
若 config.max_position_weight / max_rebalance_turnover 可行，会在目标权重生成后限制单票上限与单次换手。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.backtest_utils import prices_to_wide_close, to_returns, wide_to_long
from config import Settings, get_settings
from factors import FACTOR_REGISTRY
from models.optimizer import maximize_sharpe, risk_parity


def _resample_freq_alias(freq: str) -> str:
    """Pandas 2.2+ 推荐 ME/YE；旧别名 M/Q/A 在此映射。"""
    return {"M": "ME", "Q": "QE", "A": "YE", "Y": "YE"}.get(freq, freq)


def _portfolio_value(cash: float, shares: pd.Series, px: pd.Series) -> float:
    stock_val = 0.0
    for s in shares.index:
        p = px.get(s)
        if p is not None and np.isfinite(float(p)):
            stock_val += float(shares[s]) * float(p)
    return cash + stock_val


def _rebalance_topk_equal_weight(
    cash: float,
    shares: pd.Series,
    px: pd.Series,
    picks: List[str],
    commission_rate: float,
) -> Tuple[float, pd.Series]:
    """先卖后买；买端若现金不足则按比例缩放。"""
    shares = shares.copy()
    if not picks:
        return cash, shares

    w = 1.0 / len(picks)
    nav0 = _portfolio_value(cash, shares, px)
    tgt_dollar = {s: (nav0 * w if s in picks else 0.0) for s in shares.index}

    cur_dollar = pd.Series(0.0, index=shares.index)
    for s in shares.index:
        p = px.get(s)
        if p is not None and np.isfinite(float(p)):
            cur_dollar[s] = float(shares[s]) * float(p)

    for s in shares.index:
        tgt = tgt_dollar[s]
        cur = float(cur_dollar[s])
        if tgt >= cur - 1e-9:
            continue
        sell_dollar = cur - tgt
        p = px[s]
        if not np.isfinite(float(p)) or sell_dollar <= 0:
            continue
        fee = commission_rate * sell_dollar
        cash += sell_dollar - fee
        shares[s] -= sell_dollar / float(p)
        cur_dollar[s] = float(shares[s]) * float(p) if np.isfinite(float(p)) else 0.0

    buy_list: List[Tuple[str, float]] = []
    for s in picks:
        p = px.get(s)
        if p is None or not np.isfinite(float(p)):
            continue
        tgt = nav0 * w
        cur = float(shares[s]) * float(p)
        need = tgt - cur
        if need > 1e-9:
            buy_list.append((s, need))

    total_with_fee = sum(n * (1.0 + commission_rate) for _, n in buy_list)
    scale = 1.0
    if total_with_fee > cash + 1e-9 and total_with_fee > 0:
        scale = cash / total_with_fee

    for s, need in buy_list:
        p = px[s]
        buy_dollar = need * scale
        if buy_dollar <= 1e-12:
            continue
        fee = commission_rate * buy_dollar
        pay = buy_dollar + fee
        if pay > cash + 1e-9:
            buy_dollar = cash / (1.0 + commission_rate)
            fee = commission_rate * buy_dollar
            pay = buy_dollar + fee
        cash -= pay
        shares[s] += buy_dollar / float(p)

    return cash, shares


def _rebalance_to_target_weights(
    cash: float,
    shares: pd.Series,
    px: pd.Series,
    picks: List[str],
    pick_weights: List[float],
    commission_rate: float,
) -> Tuple[float, pd.Series]:
    """与等权调仓同一撮合逻辑，但 picks[i] 的目标资金占比为 pick_weights[i]（须非负且和为 1）。"""
    shares = shares.copy()
    if not picks or not pick_weights:
        return cash, shares
    if len(picks) != len(pick_weights):
        raise ValueError("picks 与 pick_weights 长度须一致")
    ssum = float(sum(pick_weights))
    if ssum <= 1e-12:
        return cash, shares
    pw = [float(w) / ssum for w in pick_weights]
    wmap = dict(zip(picks, pw))
    nav0 = _portfolio_value(cash, shares, px)
    tgt_dollar = {s: (nav0 * float(wmap[s]) if s in wmap else 0.0) for s in shares.index}

    cur_dollar = pd.Series(0.0, index=shares.index)
    for s in shares.index:
        p = px.get(s)
        if p is not None and np.isfinite(float(p)):
            cur_dollar[s] = float(shares[s]) * float(p)

    for s in shares.index:
        tgt = tgt_dollar[s]
        cur = float(cur_dollar[s])
        if tgt >= cur - 1e-9:
            continue
        sell_dollar = cur - tgt
        p = px[s]
        if not np.isfinite(float(p)) or sell_dollar <= 0:
            continue
        fee = commission_rate * sell_dollar
        cash += sell_dollar - fee
        shares[s] -= sell_dollar / float(p)
        cur_dollar[s] = float(shares[s]) * float(p) if np.isfinite(float(p)) else 0.0

    buy_list: List[Tuple[str, float]] = []
    for s in picks:
        p = px.get(s)
        if p is None or not np.isfinite(float(p)):
            continue
        tgt = nav0 * float(wmap[s])
        cur = float(shares[s]) * float(p)
        need = tgt - cur
        if need > 1e-9:
            buy_list.append((s, need))

    total_with_fee = sum(n * (1.0 + commission_rate) for _, n in buy_list)
    scale = 1.0
    if total_with_fee > cash + 1e-9 and total_with_fee > 0:
        scale = cash / total_with_fee

    for s, need in buy_list:
        p = px[s]
        buy_dollar = need * scale
        if buy_dollar <= 1e-12:
            continue
        fee = commission_rate * buy_dollar
        pay = buy_dollar + fee
        if pay > cash + 1e-9:
            buy_dollar = cash / (1.0 + commission_rate)
            fee = commission_rate * buy_dollar
            pay = buy_dollar + fee
        cash -= pay
        shares[s] += buy_dollar / float(p)

    return cash, shares


def _apply_max_position_cap(
    weights: List[float],
    max_weight: float,
) -> Tuple[List[float], bool]:
    """
    对权重施加单票上限，并把被裁掉的权重迭代分配给未触顶标的。

    返回 (新权重, 是否发生裁剪)。当 max_weight 不可行（例如 2 只股票上限 40%，总和无法到 100%）
    时返回归一后的原权重并标记未裁剪。
    """
    if not weights:
        return [], False
    arr = np.maximum(np.asarray(weights, dtype=float), 0.0)
    total = float(arr.sum())
    if total <= 1e-12:
        arr = np.full(len(weights), 1.0 / len(weights), dtype=float)
    else:
        arr = arr / total

    cap = float(max_weight)
    if not np.isfinite(cap) or cap <= 0.0 or cap >= 1.0:
        return arr.tolist(), False
    if cap * len(arr) < 1.0 - 1e-12:
        return arr.tolist(), False

    capped = np.minimum(arr, cap)
    changed = bool(np.any(arr > cap + 1e-12))
    for _ in range(len(arr) + 2):
        gap = 1.0 - float(capped.sum())
        if gap <= 1e-12:
            break
        room = np.maximum(cap - capped, 0.0)
        room_sum = float(room.sum())
        if room_sum <= 1e-12:
            break
        add = np.minimum(room, gap * room / room_sum)
        capped += add

    s = float(capped.sum())
    if s <= 1e-12:
        return arr.tolist(), False
    capped = capped / s
    return capped.tolist(), changed or bool(np.any(capped > cap + 1e-9))


def _cap_label(label: str) -> str:
    if label.endswith("_capped"):
        return label
    return "%s_capped" % label


def _target_weight_map(picks: List[str], weights: List[float]) -> Dict[str, float]:
    if not picks or not weights:
        return {}
    arr = np.maximum(np.asarray(weights, dtype=float), 0.0)
    total = float(arr.sum())
    if total <= 1e-12:
        return {}
    out: Dict[str, float] = {}
    for sym, w in zip(picks, arr / total):
        if float(w) <= 1e-12:
            continue
        out[str(sym)] = out.get(str(sym), 0.0) + float(w)
    return out


def _target_weight_lists(
    target: Dict[str, float],
    preferred_order: List[str],
) -> Tuple[List[str], List[float]]:
    ordered: List[str] = []
    seen: set[str] = set()
    for sym in preferred_order:
        ss = str(sym)
        if ss in seen or target.get(ss, 0.0) <= 1e-12:
            continue
        ordered.append(ss)
        seen.add(ss)
    for sym in sorted(target):
        if sym in seen or target.get(sym, 0.0) <= 1e-12:
            continue
        ordered.append(sym)
        seen.add(sym)
    weights = [float(target[s]) for s in ordered]
    return ordered, weights


def _apply_rebalance_turnover_cap(
    prev_target: Dict[str, float],
    target: Dict[str, float],
    max_turnover: float,
) -> Tuple[Dict[str, float], bool, float, float]:
    """
    限制单次目标权重变化：target' = prev + scale * (target - prev)。

    返回 (新目标权重, 是否节流, 原始目标换手, scale)。首次建仓不节流。
    """
    clean_target = {str(k): float(v) for k, v in target.items() if float(v) > 1e-12}
    if not clean_target:
        return {}, False, 0.0, 1.0

    clean_prev = {str(k): float(v) for k, v in prev_target.items() if float(v) > 1e-12}
    symbols = sorted(set(clean_prev) | set(clean_target))
    turnover = float(
        sum(abs(clean_target.get(sym, 0.0) - clean_prev.get(sym, 0.0)) for sym in symbols)
    )

    cap = float(max_turnover)
    if not clean_prev or not np.isfinite(cap) or cap <= 0.0 or turnover <= cap + 1e-12:
        return clean_target, False, turnover, 1.0

    scale = max(0.0, min(1.0, cap / turnover))
    capped: Dict[str, float] = {}
    for sym in symbols:
        w = clean_prev.get(sym, 0.0) + scale * (clean_target.get(sym, 0.0) - clean_prev.get(sym, 0.0))
        if w > 1e-12:
            capped[sym] = float(w)
    total = float(sum(capped.values()))
    if total > 1e-12:
        capped = {k: v / total for k, v in capped.items()}
    return capped, True, turnover, scale


def _turnover_cap_label(label: str) -> str:
    if label.endswith("_turnover_capped"):
        return label
    return "%s_turnover_capped" % label


def _estimate_mu_cov_for_picks(
    prices_wide: pd.DataFrame,
    picks: List[str],
    end_dt: pd.Timestamp,
    *,
    window: int,
    min_obs: int,
) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    用 end_dt 当日及之前收盘价，取 [end_dt 往前 window 个交易日] 的日简单收益样本，
    估计 mu（日均收益向量）、cov（样本协方差）；列顺序与 picks 一致。
    """
    if not picks:
        return None
    try:
        sub = prices_wide.loc[:end_dt, picks]
    except (KeyError, TypeError):
        return None
    if sub.shape[1] < len(picks):
        return None
    sub = sub.dropna(axis=0, how="any")
    if len(sub) < 2:
        return None
    tail = sub.iloc[-min(len(sub), window + 1) :]
    rets = tail.pct_change().iloc[1:].dropna(how="any")
    if len(rets) < min_obs:
        return None
    mu = rets.mean().reindex(picks).to_numpy(dtype=float)
    cov = rets[picks].cov().reindex(index=picks, columns=picks).to_numpy(dtype=float)
    if not np.all(np.isfinite(mu)) or mu.shape[0] != len(picks):
        return None
    if cov.shape != (len(picks), len(picks)) or not np.all(np.isfinite(cov)):
        return None
    return mu, cov


def _weights_for_rebalance(
    prices_wide: pd.DataFrame,
    picks: List[str],
    dt: pd.Timestamp,
    settings: Settings,
) -> Tuple[List[float], str]:
    """
    返回与 picks 对齐的权重列表（和为 1）及模式标签：
    equal / max_sharpe / max_sharpe_fallback / risk_parity / risk_parity_fallback。
    """
    n = len(picks)
    if n == 0:
        return [], "equal"
    eq = [1.0 / n] * n
    max_pos = float(getattr(settings, "max_position_weight", 0.0) or 0.0)
    mode = str(getattr(settings, "portfolio_weighting", "equal") or "equal").strip().lower()
    if n < 2 or mode == "equal":
        w, capped = _apply_max_position_cap(eq, max_pos)
        return w, _cap_label("equal") if capped else "equal"

    win = int(getattr(settings, "optimizer_return_window", 60))
    min_obs = int(getattr(settings, "optimizer_min_obs", 15))

    if mode == "max_sharpe":
        est = _estimate_mu_cov_for_picks(prices_wide, picks, dt, window=win, min_obs=min_obs)
        if est is None:
            return eq, "max_sharpe_fallback"
        mu, cov = est
        try:
            w = maximize_sharpe(mu, cov, risk_free=0.0)
        except Exception:
            return eq, "max_sharpe_fallback"
        w = np.maximum(np.asarray(w, dtype=float), 0.0)
        s = float(w.sum())
        if s <= 1e-12:
            return eq, "max_sharpe_fallback"
        w = (w / s).tolist()
        if len(w) != n:
            return eq, "max_sharpe_fallback"
        w, capped = _apply_max_position_cap(w, max_pos)
        return w, _cap_label("max_sharpe") if capped else "max_sharpe"

    if mode == "risk_parity":
        est = _estimate_mu_cov_for_picks(prices_wide, picks, dt, window=win, min_obs=min_obs)
        if est is None:
            return eq, "risk_parity_fallback"
        _, cov = est
        try:
            w = np.asarray(risk_parity(cov), dtype=float)
        except Exception:
            return eq, "risk_parity_fallback"
        w = np.maximum(w, 0.0)
        s = float(w.sum())
        if s <= 1e-12:
            return eq, "risk_parity_fallback"
        w = (w / s).tolist()
        if len(w) != n:
            return eq, "risk_parity_fallback"
        w, capped = _apply_max_position_cap(w, max_pos)
        return w, _cap_label("risk_parity") if capped else "risk_parity"

    w, capped = _apply_max_position_cap(eq, max_pos)
    return w, _cap_label("equal") if capped else "equal"


def run_single_backtest(
    factor_name: str,
    *,
    factor_values: Optional[pd.Series] = None,
    prices: Optional[pd.DataFrame] = None,
    settings: Optional[Settings] = None,
    top_k: Optional[int] = None,
    lookback: Optional[int] = None,
    **kwargs: Any,
) -> tuple[pd.Series, Dict[str, Any]]:
    """
    :param factor_name: FACTOR_REGISTRY 中的键；若已传 factor_values 则仅用于 meta 记录
    :param factor_values: MultiIndex(date, symbol) 因子
    :param prices: 收盘价宽表（日期索引 × ts_code 列）或契约长表
    :param top_k: 多头只数，默认取 settings.top_k
    :param lookback: 动量窗口（仅当自动计算 MOMENTUM 时）
    :return: (nav, meta)；nav 索引为 date，值为净值
    """
    _ = kwargs
    if prices is None:
        raise ValueError("run_single_backtest 需要 prices")

    settings = settings or get_settings()
    k = int(top_k if top_k is not None else settings.top_k)
    if k < 1:
        raise ValueError("top_k 须 >= 1")

    prices_wide = prices_to_wide_close(
        prices,
        date_col="trade_date",
        symbol_col="ts_code",
        close_col=settings.price_col,
    )
    prices_wide = prices_wide.sort_index().sort_index(axis=1)

    if factor_values is None:
        if factor_name not in FACTOR_REGISTRY:
            raise KeyError(f"未知因子: {factor_name}，可注册于 factors.FACTOR_REGISTRY")
        fn = FACTOR_REGISTRY[factor_name]
        if factor_name in ("MOMENTUM", "MOMENTUM_60D"):
            default_lb = (
                getattr(settings, "momentum_long_lookback", 60)
                if factor_name == "MOMENTUM_60D"
                else settings.momentum_lookback
            )
            lb = int(lookback if lookback is not None else default_lb)
            factor_values = fn(prices_wide, lookback=lb)
        elif factor_name == "REVERSAL_5D":
            lb = int(lookback if lookback is not None else getattr(settings, "reversal_lookback", 5))
            factor_values = fn(prices_wide, lookback=lb)
        elif factor_name == "VOLATILITY":
            ret = to_returns(prices_wide)
            vw = int(kwargs.get("vol_window") or settings.vol_window)
            factor_values = fn(
                ret,
                window=vw,
                annualize=True,
                trading_days=settings.trading_days_per_year,
            )
        elif factor_name == "VOLUME_RATIO_20D":
            long_px = kwargs.get("long_prices")
            if long_px is None or "volume" not in getattr(long_px, "columns", []):
                raise ValueError("VOLUME_RATIO_20D 需要 long_prices 且含 volume 列")
            vol_wide = long_px.pivot(index="trade_date", columns="ts_code", values="volume")
            factor_values = fn(
                vol_wide,
                window=int(kwargs.get("volume_ratio_window") or getattr(settings, "volume_ratio_window", 20)),
            )
        elif factor_name in ("PE", "ROE"):
            long_px = kwargs.get("long_prices")
            if long_px is None:
                long_px = wide_to_long(prices_wide, settings.price_col)
            fina = kwargs.get("finance_df")
            if fina is None or getattr(fina, "empty", True):
                from live.data_feed import fetch_fina_indicator_panel

                fina = fetch_fina_indicator_panel(
                    list(prices_wide.columns),
                    settings.backtest_start,
                    settings.backtest_end,
                    history_years=settings.fina_history_years,
                    token=kwargs.get("token"),
                )
            if fina.empty:
                raise ValueError(
                    "财务数据为空，无法计算 PE/ROE（检查 Tushare 积分/权限或 finance_df）"
                )
            factor_values = fn(fina, long_px)
        else:
            raise ValueError(f"因子 {factor_name} 需预计算 factor_values 或未支持自动计算")

    factor_values = factor_values.copy()
    factor_values.index = factor_values.index.set_names(["date", "symbol"])

    rf = _resample_freq_alias(settings.rebalance_freq)
    rebalance_dates = prices_wide.resample(rf).last().index.intersection(prices_wide.index)

    symbols = list(prices_wide.columns)
    shares = pd.Series(0.0, index=symbols)
    cash = 1.0
    nav_records: List[Tuple[pd.Timestamp, float]] = []
    n_rebalances = 0
    rebalance_log: List[Dict[str, Any]] = []
    prev_target_weights: Dict[str, float] = {}

    for dt in prices_wide.index:
        px = prices_wide.loc[dt]
        nav = _portfolio_value(cash, shares, px)
        nav_records.append((dt, nav))

        if dt not in rebalance_dates:
            continue

        try:
            sc = factor_values.xs(dt, level=0)
        except KeyError:
            continue
        sc = sc.dropna()
        if sc.empty:
            continue
        sc = sc.sort_values(ascending=False)
        picks = []
        for sym in sc.index:
            if sym not in px.index:
                continue
            if not np.isfinite(float(px[sym])):
                continue
            picks.append(sym)
            if len(picks) >= k:
                break
        if not picks:
            continue

        selected_picks = list(picks)
        pick_weights, w_label = _weights_for_rebalance(prices_wide, selected_picks, dt, settings)
        target_map = _target_weight_map(selected_picks, pick_weights)
        target_map, turnover_capped, target_turnover, turnover_scale = _apply_rebalance_turnover_cap(
            prev_target_weights,
            target_map,
            float(getattr(settings, "max_rebalance_turnover", 0.0) or 0.0),
        )
        if turnover_capped:
            w_label = _turnover_cap_label(w_label)
        target_picks, target_weights = _target_weight_lists(
            target_map,
            selected_picks + list(prev_target_weights.keys()),
        )
        if not target_picks:
            continue

        cash, shares = _rebalance_to_target_weights(
            cash, shares, px, target_picks, target_weights, settings.commission_rate
        )

        rebalance_log.append(
            {
                "date": dt,
                "picks": list(target_picks),
                "selected_picks": list(selected_picks),
                "weights": [float(x) for x in target_weights],
                "weighting": w_label,
                "target_turnover": float(target_turnover),
                "turnover_capped": bool(turnover_capped),
                "turnover_scale": float(turnover_scale),
            }
        )
        prev_target_weights = dict(zip(target_picks, [float(x) for x in target_weights]))
        n_rebalances += 1

    nav = pd.Series(
        [v for _, v in nav_records],
        index=pd.DatetimeIndex([d for d, _ in nav_records], name="date"),
        name="nav",
        dtype=float,
    )
    meta: Dict[str, Any] = {
        "factor_name": factor_name,
        "top_k": k,
        "rebalance_freq": settings.rebalance_freq,
        "commission_rate": settings.commission_rate,
        "n_rebalances": n_rebalances,
        "portfolio_weighting": getattr(settings, "portfolio_weighting", "equal"),
        "max_position_weight": getattr(settings, "max_position_weight", 0.0),
        "max_rebalance_turnover": getattr(settings, "max_rebalance_turnover", 0.0),
        "rebalance_log": rebalance_log,
    }
    return nav, meta
