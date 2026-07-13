"""
股票池读取与代码规范化。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from config import Settings


_DEFAULT_NAME_COLUMNS = ("股票简称", "name", "symbol_name", "证券简称")
_DEFAULT_THEME_COLUMNS = ("主题", "分类", "theme", "category")
_DEFAULT_SUB_INDUSTRY_COLUMNS = ("子行业", "sub_industry", "industry", "分类", "category")
_DEFAULT_ENABLED_COLUMNS = ("是否启用", "启用", "enabled", "active")


def normalize_ts_code(value: object) -> str:
    """将常见股票代码规范成 Tushare `ts_code`。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    code = str(value).strip().upper()
    if not code or code == "NAN":
        return ""
    code = code.replace(" ", "")
    if "." in code:
        left, right = code.split(".", 1)
        left = left.zfill(6) if left.isdigit() else left
        return "%s.%s" % (left, right)
    digits = "".join(ch for ch in code if ch.isdigit())
    if not digits:
        return code
    if len(digits) < 6:
        digits = digits.zfill(6)
    if len(digits) != 6:
        return code
    suffix = "SH" if digits.startswith(("5", "6", "9")) else "SZ"
    return "%s.%s" % (digits, suffix)


def is_valid_ts_code(value: object) -> bool:
    """判断是否为当前工程支持的 A 股 Tushare 代码。"""
    code = normalize_ts_code(value)
    if len(code) != 9 or code[6] != ".":
        return False
    left, right = code.split(".", 1)
    return left.isdigit() and len(left) == 6 and right in {"SH", "SZ"}


def _read_stock_pool_file(path: str | Path) -> pd.DataFrame:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError("股票池文件仅支持 xlsx/xls/csv: %s" % p)


def _first_existing(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def _to_enabled(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    text = str(value).strip().lower()
    if text in {"", "1", "true", "yes", "y", "是", "启用", "active"}:
        return True
    if text in {"0", "false", "no", "n", "否", "停用", "disabled", "disable"}:
        return False
    return bool(value)


def load_stock_pool_frame(
    path: str | Path,
    *,
    code_col: str = "股票代码",
    name_col: str | None = None,
    theme_col: str | None = None,
    sub_industry_col: str | None = None,
    enabled_col: str | None = None,
) -> pd.DataFrame:
    """
    读取完整股票池表，输出统一列：
    `symbol/name/theme/sub_industry/enabled/raw_symbol/source_path`。

    这个函数保留人工研究池的元信息，供实盘前过滤报告和确认文件使用。
    """
    p = Path(path).expanduser()
    frame = _read_stock_pool_file(p)
    if code_col not in frame.columns:
        raise ValueError("股票池缺少代码列 %r；实际列=%s" % (code_col, list(frame.columns)))

    columns = list(frame.columns)
    name_col = name_col or _first_existing(columns, _DEFAULT_NAME_COLUMNS)
    theme_col = theme_col or _first_existing(columns, _DEFAULT_THEME_COLUMNS)
    sub_industry_col = sub_industry_col or _first_existing(columns, _DEFAULT_SUB_INDUSTRY_COLUMNS)
    enabled_col = enabled_col or _first_existing(columns, _DEFAULT_ENABLED_COLUMNS)

    out = pd.DataFrame()
    out["raw_symbol"] = frame[code_col]
    out["symbol"] = frame[code_col].map(normalize_ts_code)
    out["name"] = frame[name_col].fillna("").astype(str) if name_col else ""
    out["theme"] = frame[theme_col].fillna("").astype(str) if theme_col else ""
    out["sub_industry"] = frame[sub_industry_col].fillna("").astype(str) if sub_industry_col else ""
    out["enabled"] = frame[enabled_col].map(_to_enabled) if enabled_col else True
    out["source_path"] = str(p)

    out = out.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)
    if out.empty:
        raise ValueError("股票池没有记录: %s" % p)
    return out


def load_stock_pool(path: str | Path, *, code_col: str = "股票代码") -> list[str]:
    """从 Excel/CSV 读取股票池，返回去重后的 Tushare 代码列表。"""
    frame = load_stock_pool_frame(path, code_col=code_col)
    symbols = [x for x in frame.loc[frame["enabled"], "symbol"].tolist() if is_valid_ts_code(x)]
    out = sorted({x for x in symbols if x})
    if not out:
        raise ValueError("股票池没有有效股票代码: %s" % Path(path).expanduser())
    return out


def _status_map(trade_status: pd.DataFrame | Mapping[str, Mapping[str, Any]] | None) -> dict[str, dict[str, bool]]:
    if trade_status is None:
        return {}
    if isinstance(trade_status, pd.DataFrame):
        symbol_col = "symbol" if "symbol" in trade_status.columns else "ts_code"
        if symbol_col not in trade_status.columns:
            return {}
        out: dict[str, dict[str, bool]] = {}
        for rec in trade_status.to_dict("records"):
            sym = normalize_ts_code(rec.get(symbol_col))
            if not sym:
                continue
            out[sym] = {
                "is_suspended": _to_enabled(rec.get("is_suspended", False)),
                "is_limit_up": _to_enabled(rec.get("is_limit_up", False)),
                "is_limit_down": _to_enabled(rec.get("is_limit_down", False)),
            }
        return out
    return {
        normalize_ts_code(symbol): {
            "is_suspended": _to_enabled(flags.get("is_suspended", False)),
            "is_limit_up": _to_enabled(flags.get("is_limit_up", False)),
            "is_limit_down": _to_enabled(flags.get("is_limit_down", False)),
        }
        for symbol, flags in trade_status.items()
    }


def _price_wide(price_data: pd.DataFrame | None) -> pd.DataFrame:
    if price_data is None or price_data.empty:
        return pd.DataFrame()
    if {"trade_date", "ts_code", "close"}.issubset(price_data.columns):
        px = price_data.copy()
        px["trade_date"] = pd.to_datetime(px["trade_date"])
        px["ts_code"] = px["ts_code"].map(normalize_ts_code)
        return px.pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    out = price_data.copy()
    out.index = pd.to_datetime(out.index)
    out.columns = [normalize_ts_code(c) for c in out.columns]
    return out.sort_index()


def _latest_liquidity(
    price_data: pd.DataFrame | None,
    *,
    as_of_date: pd.Timestamp,
    lookback_days: int,
) -> pd.DataFrame:
    if price_data is None or price_data.empty or not {"trade_date", "ts_code"}.issubset(price_data.columns):
        return pd.DataFrame(columns=["symbol", "avg_volume", "avg_amount"])
    df = price_data.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["symbol"] = df["ts_code"].map(normalize_ts_code)
    df = df[df["trade_date"] <= as_of_date].sort_values(["symbol", "trade_date"])
    tail = df.groupby("symbol", group_keys=False).tail(max(int(lookback_days), 1))
    rows = []
    for symbol, group in tail.groupby("symbol"):
        avg_volume = float(group["volume"].mean()) if "volume" in group.columns else float("nan")
        if "amount" in group.columns:
            avg_amount = float(group["amount"].mean())
        elif {"close", "volume"}.issubset(group.columns):
            avg_amount = float((group["close"] * group["volume"]).mean())
        else:
            avg_amount = float("nan")
        rows.append({"symbol": symbol, "avg_volume": avg_volume, "avg_amount": avg_amount})
    return pd.DataFrame(rows)


def build_stock_pool_filter_report(
    pool: pd.DataFrame,
    *,
    price_data: pd.DataFrame | None = None,
    trade_status: pd.DataFrame | Mapping[str, Mapping[str, Any]] | None = None,
    as_of_date: Any | None = None,
    min_price_coverage: float = 0.8,
    min_history_days: int = 20,
    liquidity_lookback_days: int = 20,
    min_avg_volume: float = 0.0,
    min_avg_amount: float = 0.0,
    exclude_limit_up: bool = True,
    exclude_limit_down: bool = True,
) -> pd.DataFrame:
    """
    基于人工股票池生成实盘前过滤报告。

    输出每只股票是否进入 `active_universe`，以及未进入的原因。
    """
    required = {"symbol", "name", "theme", "sub_industry", "enabled"}
    missing = required - set(pool.columns)
    if missing:
        raise ValueError("pool 缺少列: %s" % sorted(missing))

    prices = _price_wide(price_data)
    if as_of_date is None:
        as_of = pd.Timestamp(prices.index[-1]) if not prices.empty else pd.Timestamp.today().normalize()
    else:
        as_of = pd.Timestamp(as_of_date)

    if not prices.empty:
        prices = prices[prices.index <= as_of]
    liq = _latest_liquidity(
        price_data,
        as_of_date=as_of,
        lookback_days=liquidity_lookback_days,
    ).set_index("symbol")
    statuses = _status_map(trade_status)

    rows: list[dict[str, Any]] = []
    for rec in pool.to_dict("records"):
        symbol = normalize_ts_code(rec.get("symbol"))
        reasons: list[str] = []
        valid_code = is_valid_ts_code(symbol)
        enabled = bool(rec.get("enabled", True))
        if not valid_code:
            reasons.append("invalid_symbol")
        if not enabled:
            reasons.append("disabled_by_pool")

        series = prices[symbol].dropna() if valid_code and symbol in prices.columns else pd.Series(dtype=float)
        n_obs = int(series.shape[0])
        coverage = float(n_obs / len(prices.index)) if len(prices.index) else 0.0
        latest_price = float(series.iloc[-1]) if n_obs else float("nan")
        latest_price_date = series.index[-1].strftime("%Y-%m-%d") if n_obs else ""
        price_available = n_obs > 0 and latest_price_date == as_of.strftime("%Y-%m-%d")
        if n_obs == 0:
            reasons.append("missing_price_history")
        elif not price_available:
            reasons.append("latest_price_missing")
        if n_obs < int(min_history_days):
            reasons.append("history_days_below_min")
        if coverage < float(min_price_coverage):
            reasons.append("price_coverage_below_min")

        avg_volume = float(liq.loc[symbol, "avg_volume"]) if symbol in liq.index else float("nan")
        avg_amount = float(liq.loc[symbol, "avg_amount"]) if symbol in liq.index else float("nan")
        if min_avg_volume > 0 and (pd.isna(avg_volume) or avg_volume < min_avg_volume):
            reasons.append("avg_volume_below_min")
        if min_avg_amount > 0 and (pd.isna(avg_amount) or avg_amount < min_avg_amount):
            reasons.append("avg_amount_below_min")

        status = statuses.get(symbol, {})
        is_suspended = bool(status.get("is_suspended", False))
        is_limit_up = bool(status.get("is_limit_up", False))
        is_limit_down = bool(status.get("is_limit_down", False))
        if is_suspended:
            reasons.append("suspended")
        if exclude_limit_up and is_limit_up:
            reasons.append("limit_up")
        if exclude_limit_down and is_limit_down:
            reasons.append("limit_down")

        active = len(reasons) == 0
        rows.append(
            {
                "date": as_of.strftime("%Y-%m-%d"),
                "symbol": symbol,
                "name": rec.get("name", ""),
                "theme": rec.get("theme", ""),
                "sub_industry": rec.get("sub_industry", ""),
                "enabled": enabled,
                "active": active,
                "exclude_reason": "" if active else ";".join(reasons),
                "price_available": bool(price_available),
                "latest_price_date": latest_price_date,
                "latest_price": latest_price,
                "history_days": n_obs,
                "price_coverage": coverage,
                "avg_volume": avg_volume,
                "avg_amount": avg_amount,
                "is_suspended": is_suspended,
                "is_limit_up": is_limit_up,
                "is_limit_down": is_limit_down,
            }
        )
    return pd.DataFrame(rows).sort_values(["active", "symbol"], ascending=[False, True]).reset_index(drop=True)


def active_universe_from_report(report: pd.DataFrame) -> pd.DataFrame:
    """从过滤报告提取实盘目标池确认表。"""
    cols = [
        "date",
        "symbol",
        "name",
        "theme",
        "sub_industry",
        "latest_price",
        "avg_volume",
        "avg_amount",
    ]
    if report.empty:
        return pd.DataFrame(columns=cols)
    return report.loc[report["active"], cols].reset_index(drop=True)


def save_universe_files(
    settings: Settings,
    report: pd.DataFrame,
    *,
    trade_date: Any | None = None,
    subdir: str = "live_universe",
) -> dict[str, Path]:
    """保存股票池过滤报告与实盘目标池确认文件。"""
    if report.empty and trade_date is None:
        dt = pd.Timestamp.today().strftime("%Y-%m-%d")
    elif trade_date is not None:
        dt = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    else:
        dt = str(report["date"].iloc[0])
    base = settings.output_dir / subdir
    base.mkdir(parents=True, exist_ok=True)
    report_path = base / ("stock_pool_filter_report_%s.csv" % dt)
    active_path = base / ("active_universe_%s.csv" % dt)
    report.to_csv(report_path, index=False)
    active_universe_from_report(report).to_csv(active_path, index=False)
    return {"filter_report": report_path, "active_universe": active_path}
