"""
股票池读取与代码规范化。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


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
    if len(digits) < 6:
        digits = digits.zfill(6)
    if len(digits) != 6:
        return code
    suffix = "SH" if digits.startswith(("5", "6", "9")) else "SZ"
    return "%s.%s" % (digits, suffix)


def load_stock_pool(path: str | Path, *, code_col: str = "股票代码") -> list[str]:
    """从 Excel/CSV 读取股票池，返回去重后的 Tushare 代码列表。"""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(p)
    elif p.suffix.lower() == ".csv":
        frame = pd.read_csv(p)
    else:
        raise ValueError("股票池文件仅支持 xlsx/xls/csv: %s" % p)
    if code_col not in frame.columns:
        raise ValueError("股票池缺少代码列 %r；实际列=%s" % (code_col, list(frame.columns)))
    symbols = [normalize_ts_code(x) for x in frame[code_col].tolist()]
    out = sorted({x for x in symbols if x})
    if not out:
        raise ValueError("股票池没有有效股票代码: %s" % p)
    return out
