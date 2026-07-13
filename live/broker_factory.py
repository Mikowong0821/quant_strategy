"""
券商通道选择与 Adapter Factory。

本模块只负责把 Settings 中的券商配置转换成统一 `BrokerAdapter`。
具体 QMT / PTrade / 掘金接入时，应在这里注册对应 Adapter，而不是让上层流程
直接依赖某一家券商 SDK。
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from config import Settings
from live.broker import (
    BROKER_MODE_REAL_READONLY,
    BROKER_MODE_REAL_TRADING,
    BROKER_MODE_SIMULATED,
    BrokerAdapter,
    RealBrokerConfig,
    RealBrokerReadOnlyAdapter,
    SimulatedBroker,
)


BROKER_PROVIDER_SIMULATED = "simulated"
BROKER_PROVIDER_READONLY_CSV = "readonly_csv"
SUPPORTED_REAL_PROVIDERS = {"qmt", "ptrade", "gm"}


class BrokerAdapterNotConfiguredError(RuntimeError):
    """券商配置不足或请求了尚未实现的真实交易通道。"""


def normalize_broker_mode(value: str | None) -> str:
    mode = str(value or BROKER_MODE_SIMULATED).strip().lower()
    aliases = {
        "paper": BROKER_MODE_SIMULATED,
        "paper_trading": BROKER_MODE_SIMULATED,
        "sim": BROKER_MODE_SIMULATED,
        "readonly": BROKER_MODE_REAL_READONLY,
        "read_only": BROKER_MODE_REAL_READONLY,
        "semi_auto": BROKER_MODE_REAL_READONLY,
        "trading": BROKER_MODE_REAL_TRADING,
        "real": BROKER_MODE_REAL_TRADING,
    }
    mode = aliases.get(mode, mode)
    allowed = {BROKER_MODE_SIMULATED, BROKER_MODE_REAL_READONLY, BROKER_MODE_REAL_TRADING}
    if mode not in allowed:
        raise ValueError("broker_mode 仅支持: %s" % ", ".join(sorted(allowed)))
    return mode


def normalize_broker_provider(value: str | None, *, mode: str) -> str:
    provider = str(value or "").strip().lower()
    if mode == BROKER_MODE_SIMULATED:
        return provider or BROKER_PROVIDER_SIMULATED
    if not provider:
        raise BrokerAdapterNotConfiguredError(
            "真实券商模式需要配置 broker_provider，例如 qmt、ptrade、gm 或 readonly_csv"
        )
    return provider


def build_broker_config(settings: Settings) -> RealBrokerConfig:
    """从 Settings 构造真实券商非敏感配置。"""
    mode = normalize_broker_mode(settings.broker_mode)
    if mode == BROKER_MODE_SIMULATED:
        raise ValueError("simulated 模式不需要 RealBrokerConfig")
    provider = normalize_broker_provider(settings.broker_provider, mode=mode)
    return RealBrokerConfig(
        provider=provider,
        account_id=settings.broker_account_id,
        mode=mode,
    )


def create_broker_adapter(
    settings: Settings,
    *,
    cash: float | None = None,
    positions: pd.DataFrame | Mapping[str, float] | pd.Series | None = None,
    latest_prices: pd.DataFrame | Mapping[str, float] | pd.Series | None = None,
    account: Mapping[str, Any] | None = None,
    orders: pd.DataFrame | None = None,
) -> BrokerAdapter:
    """
    根据 Settings 创建统一券商 Adapter。

    - `simulated`：返回 `SimulatedBroker`，用于本地纸面 / 协议验证。
    - `real_readonly` + `readonly_csv`：返回只读快照 Adapter，用于读取外部导出的账户/持仓。
    - `real_readonly` + qmt/ptrade/gm：目前返回只读骨架；后续接 SDK 时在这里替换为具体 Adapter。
    - `real_trading`：当前明确报错，避免未验证前打开真实交易能力。
    """
    mode = normalize_broker_mode(settings.broker_mode)
    provider = normalize_broker_provider(settings.broker_provider, mode=mode)

    if mode == BROKER_MODE_SIMULATED:
        return SimulatedBroker(
            cash=float(settings.paper_initial_cash if cash is None else cash),
            positions=positions,
            latest_prices=latest_prices,
            commission_rate=float(settings.commission_rate),
        )

    config = RealBrokerConfig(
        provider=provider,
        account_id=settings.broker_account_id,
        mode=mode,
    )
    if mode == BROKER_MODE_REAL_READONLY:
        if provider == BROKER_PROVIDER_READONLY_CSV or provider in SUPPORTED_REAL_PROVIDERS:
            return RealBrokerReadOnlyAdapter(
                config,
                account=account,
                positions=positions,
                orders=orders,
                latest_prices=latest_prices,
            )
        raise BrokerAdapterNotConfiguredError("暂不支持的只读券商 provider: %s" % provider)

    raise BrokerAdapterNotConfiguredError(
        "真实交易模式尚未实现；请先使用 broker_mode=real_readonly 完成只读对账"
    )
