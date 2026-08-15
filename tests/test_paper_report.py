"""纸面交易日报。"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from config import get_settings
from live.paper_guard import GuardIssue
from live.paper_report import build_daily_paper_report, save_daily_paper_report
from live.paper_runner import run_daily_paper_trade


class TestPaperReport(unittest.TestCase):
    def test_report_contains_summary_orders_and_positions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            result = run_daily_paper_trade(
                settings,
                strategy="REPORT",
                target_weights={"AAA": 0.5},
                latest_prices={"AAA": 10.0},
                trade_date="2024-01-31",
            )
            result["target_date"] = pd.Timestamp("2024-01-31")
            result["price_date"] = pd.Timestamp("2024-01-31")

            report = build_daily_paper_report(result)
            path = save_daily_paper_report(settings, result)

            self.assertIn("# 纸面交易日报 - REPORT - 2024-01-31", report)
            self.assertIn("- 订单数：1", report)
            self.assertIn("| AAA | BUY | 500 |", report)
            self.assertIn("## 当前持仓", report)
            self.assertTrue(path.is_file())
            self.assertIn("纸面交易日报", path.read_text(encoding="utf-8"))

    def test_report_includes_previous_snapshot_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            run_daily_paper_trade(
                settings,
                strategy="REPORT",
                target_weights={"AAA": 0.5},
                latest_prices={"AAA": 10.0},
                trade_date="2024-01-31",
            )
            result = run_daily_paper_trade(
                settings,
                strategy="REPORT",
                target_weights={"AAA": 0.5},
                latest_prices={"AAA": 12.0},
                trade_date="2024-02-29",
            )
            result["target_date"] = pd.Timestamp("2024-02-29")
            result["price_date"] = pd.Timestamp("2024-02-29")

            report = build_daily_paper_report(result)

            self.assertIn("## 较上一快照变化", report)
            self.assertIn("- 上一快照日期：2024-01-31", report)
            self.assertIn("- 资产变化：1000.00", report)

    def test_report_includes_guard_issues(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            result = run_daily_paper_trade(
                settings,
                strategy="REPORT",
                target_weights={"AAA": 0.5},
                latest_prices={"AAA": 10.0},
                trade_date="2024-01-31",
            )
            result["target_date"] = pd.Timestamp("2024-01-31")
            result["price_date"] = pd.Timestamp("2024-01-31")
            result["guard_issues"] = [
                GuardIssue("WARNING", "stale_price_date", "价格日期可能过旧"),
            ]

            report = build_daily_paper_report(result)

            self.assertIn("## 运行检查", report)
            self.assertIn("stale_price_date", report)

    def test_report_includes_factor_decay_monitor(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            result = run_daily_paper_trade(
                settings,
                strategy="REPORT",
                target_weights={"AAA": 0.5},
                latest_prices={"AAA": 10.0},
                trade_date="2024-01-31",
            )
            result["target_date"] = pd.Timestamp("2024-01-31")
            result["price_date"] = pd.Timestamp("2024-01-31")
            result["factor_decay_monitor"] = pd.DataFrame(
                [
                    {
                        "factor": "ML_SCORE",
                        "status": "DEGRADED",
                        "reasons": "validation_ic_mean_below_threshold",
                        "validation_ic_mean": -0.02,
                        "validation_positive_rate": 0.42,
                        "validation_excess_ann_return": -0.1,
                        "validation_top_minus_bottom_ann": -0.2,
                        "validation_monotonicity_score": 0.2,
                    },
                    {
                        "factor": "ROE",
                        "status": "OK",
                        "reasons": "",
                        "validation_ic_mean": 0.03,
                        "validation_positive_rate": 0.58,
                        "validation_excess_ann_return": 0.12,
                        "validation_top_minus_bottom_ann": 0.18,
                        "validation_monotonicity_score": 0.8,
                    },
                ]
            )

            report = build_daily_paper_report(result)

            self.assertIn("## 因子健康与失效监控", report)
            self.assertIn("- 整体状态：`DEGRADED`", report)
            self.assertIn("- 风险因子数量：1", report)
            self.assertIn("| ML_SCORE | DEGRADED | validation_ic_mean_below_threshold |", report)

    def test_report_includes_style_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            result = run_daily_paper_trade(
                settings,
                strategy="REPORT",
                target_weights={"AAA": 0.5},
                latest_prices={"AAA": 10.0},
                trade_date="2024-01-31",
            )
            result["target_date"] = pd.Timestamp("2024-01-31")
            result["price_date"] = pd.Timestamp("2024-01-31")
            result["style_exposure"] = pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp("2024-01-31"),
                        "strategy": "REPORT",
                        "style": "QUALITY_STYLE",
                        "weighted_exposure": 0.8,
                        "abs_weighted_exposure": 0.8,
                        "score_coverage": 1.0,
                        "n_positions": 5,
                        "n_scored_positions": 5,
                    }
                ]
            )

            report = build_daily_paper_report(result)

            self.assertIn("## 组合风格暴露", report)
            self.assertIn("- 主导风格：`QUALITY_STYLE`", report)
            self.assertIn("| 2024-01-31 | QUALITY_STYLE | 0.8000 |", report)

    def test_report_includes_enhanced_factor_health_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            result = run_daily_paper_trade(
                settings,
                strategy="REPORT",
                target_weights={"AAA": 0.5},
                latest_prices={"AAA": 10.0},
                trade_date="2024-01-31",
            )
            result["target_date"] = pd.Timestamp("2024-01-31")
            result["price_date"] = pd.Timestamp("2024-01-31")
            result["factor_health_report"] = pd.DataFrame(
                [
                    {
                        "category": "滚动样本外",
                        "status": "UNSTABLE",
                        "summary": "滚动窗口不稳定因子=1/2",
                        "detail": "UNSTABLE=1/2",
                        "action": "连续多个窗口不稳定的因子不进入主融合",
                    }
                ]
            )

            report = build_daily_paper_report(result)

            self.assertIn("## 增强因子健康总览", report)
            self.assertIn("- 整体状态：`UNSTABLE`", report)
            self.assertIn("| 滚动样本外 | UNSTABLE | 滚动窗口不稳定因子=1/2 |", report)

    def test_report_includes_unified_risk_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            result = run_daily_paper_trade(
                settings,
                strategy="REPORT",
                target_weights={"AAA": 0.5},
                latest_prices={"AAA": 10.0},
                trade_date="2024-01-31",
            )
            result["target_date"] = pd.Timestamp("2024-01-31")
            result["price_date"] = pd.Timestamp("2024-01-31")
            result["risk_gate"] = pd.DataFrame(
                [
                    {
                        "trade_date": "2024-01-31",
                        "symbol": "AAA",
                        "name": "测试股票",
                        "gate_status": "BLOCK",
                        "severity": "HIGH",
                        "risk_count": 2,
                        "block_count": 1,
                        "watch_count": 1,
                        "sources": "announcement_event;negative_sentiment",
                        "reason": "处罚 | 违规",
                        "latest_triggered_at": "2024-01-30",
                        "expires_at": "2024-02-09",
                    }
                ]
            )

            report = build_daily_paper_report(result)

            self.assertIn("## 统一风险门禁", report)
            self.assertIn("- 整体状态：`BLOCK`", report)
            self.assertIn("| AAA | 测试股票 | BLOCK | 2 |", report)

    def test_report_includes_portfolio_risk_limits(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            result = run_daily_paper_trade(
                settings,
                strategy="REPORT",
                target_weights={"AAA": 0.5},
                latest_prices={"AAA": 10.0},
                trade_date="2024-01-31",
            )
            result["target_date"] = pd.Timestamp("2024-01-31")
            result["price_date"] = pd.Timestamp("2024-01-31")
            result["risk_limit_checks"] = pd.DataFrame(
                [
                    {
                        "trade_date": "2024-01-31",
                        "limit_id": "max_single_position_weight",
                        "category": "portfolio",
                        "metric": "max_single_position_weight",
                        "status": "BLOCK",
                        "observed_value": 0.5,
                        "warning_threshold": 0.3,
                        "block_threshold": 0.4,
                        "direction": "max",
                        "unit": "weight",
                        "description": "单票过高",
                        "action": "降低单票权重",
                        "details": "top=AAA 50.00%",
                    }
                ]
            )

            report = build_daily_paper_report(result)

            self.assertIn("## 组合风险限额", report)
            self.assertIn("- 整体状态：`BLOCK`", report)
            self.assertIn("| max_single_position_weight | portfolio | BLOCK | 0.5000 |", report)

    def test_report_includes_portfolio_stress_tests(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            result = run_daily_paper_trade(
                settings,
                strategy="REPORT",
                target_weights={"AAA": 0.5},
                latest_prices={"AAA": 10.0},
                trade_date="2024-01-31",
            )
            result["target_date"] = pd.Timestamp("2024-01-31")
            result["price_date"] = pd.Timestamp("2024-01-31")
            result["stress_tests"] = pd.DataFrame(
                [
                    {
                        "trade_date": "2024-01-31",
                        "scenario_id": "largest_position_down_10pct",
                        "category": "single_name",
                        "shock_type": "largest_position_down",
                        "status": "BLOCK",
                        "shock_value": -0.1,
                        "affected_weight": 0.5,
                        "estimated_portfolio_return": -0.05,
                        "estimated_loss_pct": 0.05,
                        "estimated_loss_amount": 500.0,
                        "warning_loss": 0.02,
                        "block_loss": 0.04,
                        "affected_symbols": "AAA",
                        "description": "单票冲击",
                        "action": "降低单票权重",
                        "details": "largest=AAA",
                    }
                ]
            )

            report = build_daily_paper_report(result)

            self.assertIn("## 组合压力测试", report)
            self.assertIn("- 整体状态：`BLOCK`", report)
            self.assertIn("| largest_position_down_10pct | single_name | BLOCK |", report)

    def test_report_includes_drawdown_control(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            result = run_daily_paper_trade(
                settings,
                strategy="REPORT",
                target_weights={"AAA": 0.5},
                latest_prices={"AAA": 10.0},
                trade_date="2024-01-31",
            )
            result["target_date"] = pd.Timestamp("2024-01-31")
            result["price_date"] = pd.Timestamp("2024-01-31")
            result["drawdown_control"] = pd.DataFrame(
                [
                    {
                        "trade_date": "2024-01-31",
                        "status": "WATCH",
                        "action": "REDUCE_EXPOSURE",
                        "current_total_asset": 9200.0,
                        "peak_total_asset": 10000.0,
                        "current_drawdown": -0.08,
                        "drawdown_abs": 0.08,
                        "previous_total_asset": 9800.0,
                        "latest_return": -0.0612,
                        "current_exposure": 0.89,
                        "target_exposure_before": 0.8,
                        "target_exposure_after": 0.56,
                        "target_weight_scale": 0.7,
                        "triggered_rule_id": "drawdown_watch_5pct",
                        "description": "回撤观察",
                        "details": "drawdown=8.00%",
                    }
                ]
            )

            report = build_daily_paper_report(result)

            self.assertIn("## 回撤止损与降仓控制", report)
            self.assertIn("- 整体状态：`WATCH`", report)
            self.assertIn("| WATCH | REDUCE_EXPOSURE | 9200.0000 |", report)

    def test_report_includes_capacity_impact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            result = run_daily_paper_trade(
                settings,
                strategy="REPORT",
                target_weights={"AAA": 0.5},
                latest_prices={"AAA": 10.0},
                trade_date="2024-01-31",
            )
            result["target_date"] = pd.Timestamp("2024-01-31")
            result["price_date"] = pd.Timestamp("2024-01-31")
            result["capacity_impact"] = pd.DataFrame(
                [
                    {
                        "trade_date": "2024-01-31",
                        "symbol": "AAA",
                        "side": "BUY",
                        "estimated_amount": 5000.0,
                        "avg_amount": 1000000.0,
                        "participation_rate": 0.005,
                        "impact_cost_bps": 7.0711,
                        "impact_cost_amount": 3.5355,
                        "max_order_amount_at_warning": 50000.0,
                        "capacity_multiplier_at_warning": 10.0,
                        "status": "PASS",
                        "details": "amount=5000.00",
                    }
                ]
            )
            result["capacity_impact_summary"] = pd.DataFrame(
                [
                    {
                        "trade_date": "2024-01-31",
                        "status": "PASS",
                        "n_orders": 1,
                        "n_with_liquidity": 1,
                        "n_missing_liquidity": 0,
                        "max_participation_rate": 0.005,
                        "total_order_amount": 5000.0,
                        "estimated_impact_cost_amount": 3.5355,
                        "estimated_impact_cost_bps": 7.0711,
                        "portfolio_capacity_multiplier_at_warning": 10.0,
                        "details": "lookback=20",
                    }
                ]
            )

            report = build_daily_paper_report(result)

            self.assertIn("## 容量与冲击成本", report)
            self.assertIn("- 整体状态：`PASS`", report)
            self.assertIn("| PASS | 1 | 1 | 0 | 0.0050 |", report)
            self.assertIn("| AAA | BUY | 5000.0000 |", report)

    def test_report_includes_risk_control_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            settings = replace(
                get_settings(),
                output_dir=Path(td) / "output",
                data_dir=Path(td) / "data",
                paper_initial_cash=10_000.0,
                commission_rate=0.0,
            )
            result = run_daily_paper_trade(
                settings,
                strategy="REPORT",
                target_weights={"AAA": 0.5},
                latest_prices={"AAA": 10.0},
                trade_date="2024-01-31",
            )
            result["target_date"] = pd.Timestamp("2024-01-31")
            result["price_date"] = pd.Timestamp("2024-01-31")
            result["risk_control_report"] = pd.DataFrame(
                [
                    {
                        "trade_date": "2024-01-31",
                        "module": "组合风险限额",
                        "status": "BLOCK",
                        "severity_rank": 0,
                        "summary": "max_single_position_weight",
                        "action": "暂停自动执行，先人工复核或重新生成订单。",
                    },
                    {
                        "trade_date": "2024-01-31",
                        "module": "容量与冲击成本",
                        "status": "PASS",
                        "severity_rank": 3,
                        "summary": "全部通过",
                        "action": "无需额外动作，继续监控。",
                    },
                ]
            )

            report = build_daily_paper_report(result)

            self.assertIn("## 风险总控日报", report)
            self.assertIn("- 总控状态：`BLOCK`", report)
            self.assertIn("| 组合风险限额 | BLOCK | max_single_position_weight |", report)


if __name__ == "__main__":
    unittest.main()
