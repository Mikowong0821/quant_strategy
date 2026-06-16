"""
纸面账户状态持久化：保存 / 读取虚拟账户现金、持仓和每日快照。

本模块不生成订单、不做成交，只负责把纸面交易后的账户状态稳定落盘。
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from config import Settings


ACCOUNT_COLUMNS = ["cash", "updated_at"]
POSITION_COLUMNS = ["symbol", "shares", "available_shares", "updated_at"]
SNAPSHOT_COLUMNS = ["date", "cash", "market_value", "total_asset", "n_positions"]


def account_state_dir(settings: Settings, strategy: str = "default") -> Path:
    safe = str(strategy).replace("/", "_")
    return settings.output_dir / "paper_account" / safe


def _date_to_str(value: Any) -> str:
    if value is None or value == "":
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _positions_to_frame(
    positions: pd.DataFrame | Mapping[str, float] | pd.Series | None,
    *,
    updated_at: Any = "",
) -> pd.DataFrame:
    if positions is None:
        return pd.DataFrame(columns=POSITION_COLUMNS)
    if isinstance(positions, pd.DataFrame):
        symbol_col = "symbol" if "symbol" in positions.columns else "ts_code"
        if symbol_col not in positions.columns or "shares" not in positions.columns:
            raise ValueError("positions DataFrame 须包含 symbol/ts_code 与 shares 列")
        df = pd.DataFrame(
            {
                "symbol": positions[symbol_col].astype(str),
                "shares": positions["shares"].astype(float),
                "available_shares": (
                    positions["available_shares"].astype(float)
                    if "available_shares" in positions.columns
                    else positions["shares"].astype(float)
                ),
            }
        )
    elif isinstance(positions, pd.Series):
        df = pd.DataFrame(
            {
                "symbol": positions.index.astype(str),
                "shares": positions.astype(float).to_numpy(),
                "available_shares": positions.astype(float).to_numpy(),
            }
        )
    else:
        df = pd.DataFrame(
            {
                "symbol": [str(k) for k in positions.keys()],
                "shares": [float(v) for v in positions.values()],
            }
        )
        df["available_shares"] = df["shares"]
    if df.empty:
        return pd.DataFrame(columns=POSITION_COLUMNS)
    df = df.groupby("symbol", as_index=False)[["shares", "available_shares"]].sum()
    df = df[df["shares"] > 0].sort_values("symbol").reset_index(drop=True)
    df["updated_at"] = _date_to_str(updated_at)
    return df[POSITION_COLUMNS]


def positions_from_trades(
    trades: pd.DataFrame,
    current_positions: pd.DataFrame | Mapping[str, float] | pd.Series | None = None,
    *,
    updated_at: Any = "",
) -> pd.DataFrame:
    """根据纸面交易日志和初始持仓生成最新持仓表。"""
    frame = _positions_to_frame(current_positions, updated_at=updated_at)
    positions = {
        str(rec["symbol"]): float(rec["shares"])
        for rec in frame.to_dict("records")
        if float(rec.get("shares", 0.0)) > 0
    }
    if not trades.empty:
        for rec in trades[trades["fill_status"] == "FILLED"].to_dict("records"):
            symbol = str(rec["symbol"])
            positions[symbol] = float(rec["position_after"])
    return _positions_to_frame(positions, updated_at=updated_at)


def save_account_state(
    settings: Settings,
    *,
    strategy: str,
    cash: float,
    positions: pd.DataFrame | Mapping[str, float] | pd.Series | None,
    snapshot: Mapping[str, Any] | None = None,
    trade_date: Any = None,
) -> dict[str, Path]:
    """
    保存纸面账户状态。

    输出：
    - account.csv：当前现金
    - positions.csv：当前持仓
    - snapshots.csv：追加每日账户快照
    """
    if cash < 0:
        raise ValueError("cash 不能为负")
    base = account_state_dir(settings, strategy)
    base.mkdir(parents=True, exist_ok=True)
    date_s = _date_to_str(trade_date)

    account_path = base / "account.csv"
    pd.DataFrame(
        [{"cash": float(cash), "updated_at": date_s}],
        columns=ACCOUNT_COLUMNS,
    ).to_csv(account_path, index=False)

    positions_path = base / "positions.csv"
    positions_df = _positions_to_frame(positions, updated_at=date_s)
    positions_df.to_csv(positions_path, index=False)

    snapshots_path = base / "snapshots.csv"
    if snapshot is None:
        snapshot_row = {
            "date": date_s,
            "cash": float(cash),
            "market_value": float("nan"),
            "total_asset": float("nan"),
            "n_positions": float(len(positions_df)),
        }
    else:
        snapshot_row = {
            "date": date_s,
            "cash": float(snapshot.get("cash", cash)),
            "market_value": float(snapshot.get("market_value", float("nan"))),
            "total_asset": float(snapshot.get("total_asset", float("nan"))),
            "n_positions": float(snapshot.get("n_positions", len(positions_df))),
        }
    if snapshots_path.exists():
        snapshots = pd.read_csv(snapshots_path)
        snapshots = snapshots[snapshots["date"].astype(str) != date_s]
        snapshots = pd.concat([snapshots, pd.DataFrame([snapshot_row])], ignore_index=True)
    else:
        snapshots = pd.DataFrame([snapshot_row], columns=SNAPSHOT_COLUMNS)
    snapshots = snapshots.sort_values("date").reset_index(drop=True)
    snapshots.to_csv(snapshots_path, index=False)

    return {
        "account": account_path,
        "positions": positions_path,
        "snapshots": snapshots_path,
    }


def load_account_state(
    settings: Settings,
    *,
    strategy: str,
    default_cash: float = 0.0,
) -> tuple[float, pd.DataFrame]:
    """读取纸面账户现金和持仓；不存在时返回默认现金和空持仓。"""
    base = account_state_dir(settings, strategy)
    account_path = base / "account.csv"
    positions_path = base / "positions.csv"

    cash = float(default_cash)
    if account_path.exists():
        account = pd.read_csv(account_path)
        if not account.empty and "cash" in account.columns:
            cash = float(account["cash"].iloc[-1])

    if positions_path.exists():
        positions = pd.read_csv(positions_path)
        for col in POSITION_COLUMNS:
            if col not in positions.columns:
                positions[col] = "" if col == "updated_at" else 0.0
        positions = positions[POSITION_COLUMNS]
    else:
        positions = pd.DataFrame(columns=POSITION_COLUMNS)
    return cash, positions
