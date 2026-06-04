"""
主入口（**MVP 主流程**）：
1）先构建多因子原始面板（只算一次）；
2）数据质量与覆盖率报告；
3）IC 与可选落盘；
4）各因子独立回测（`config.portfolio_weighting`：equal / max_sharpe / risk_parity）并打印每期 Top-K 及权重、绩效；
5）多因子融合：默认用 **IC 滞后滚动均值** 驱动 z-score 列权（`fusion_use_ic_weights` 可关回等权），再跑 FUSED 回测。
6）构造股票池等权基准，计算超额收益、跟踪误差与信息比率。
7）由调仓日志估算换手率与交易成本。
8）由调仓日志计算 HHI / effective_n 等持仓集中度指标。
9）可选保存运行配置、绩效汇总、调仓日志与图表，形成可复现实验记录。
非 MVP：`live` 信号/模拟盘、`fuse_models` 高阶 method；详见 README「MVP 定稿」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.benchmark import (
    equal_weight_benchmark_nav,
    excess_nav_frame,
    summarize_excess,
)
from analysis.data_quality import (
    factor_coverage,
    factor_daily_coverage,
    price_coverage,
    rebalance_coverage,
)
from analysis.ic import daily_ic_spearman, save_ic_series, summarize_ic
from analysis.performance import summarize
from analysis.plotting import (
    plot_effective_n,
    plot_factor_coverage,
    plot_ic,
    plot_nav,
    plot_turnover,
    plot_weights,
    rebalance_log_to_weights_frame,
)
from analysis.risk_exposure import (
    concentration_frame,
    effective_n_wide,
    summarize_concentration,
)
from analysis.turnover import summarize_turnover, turnover_frame, turnover_wide
from backtest.backtest_multi import run_multi_backtest
from backtest.backtest_single import run_single_backtest
from backtest.backtest_utils import long_to_wide, wide_to_long
from config import Settings, get_settings
from factors.panel_builder import DEFAULT_FACTOR_ORDER, build_four_factor_panel
from live.cache_io import (
    save_performance_summary,
    save_rebalance_logs,
    save_risk_exposure_logs,
    save_risk_exposure_summary,
    save_run_cache,
    save_run_config,
    save_data_quality_reports,
    save_turnover_logs,
)
from live.data_feed import fetch_daily_panel, load_prices_from_csv
from models.fusion import fuse_equal_weight_zscore, fuse_ic_weighted_zscore


def _build_fused_zscore_panel(
    sub: pd.DataFrame,
    ic_by_name: dict[str, pd.Series],
    settings: Settings,
) -> tuple[pd.Series, str]:
    """
    融合得分：默认 IC 滞后滚动列权；缺 IC / 配置关闭 / 异常时回退等权 z-score。
    返回 (Series, 模式标签 ic_rolling_weighted | equal_zscore)。
    """
    cols = list(sub.columns)
    use_ic = bool(getattr(settings, "fusion_use_ic_weights", True))
    win = int(getattr(settings, "fusion_ic_rolling_window", 60))
    min_p = int(getattr(settings, "fusion_ic_min_periods", 20))
    ic_sub = {k: ic_by_name[k] for k in cols if k in ic_by_name}
    if not use_ic:
        return fuse_equal_weight_zscore(sub), "equal_zscore"
    if len(ic_sub) != len(cols) or not cols:
        miss = sorted(set(cols) - set(ic_sub.keys()))
        if miss:
            print("【融合】IC 加权需各列均有 IC，缺 %s → 等权 z-score 融合" % miss)
        return fuse_equal_weight_zscore(sub), "equal_zscore"
    try:
        return (
            fuse_ic_weighted_zscore(sub, ic_sub, rolling_window=win, min_periods=min_p),
            "ic_rolling_weighted",
        )
    except Exception as e:
        print("【融合】IC 加权失败，回退等权 z-score:", e)
        return fuse_equal_weight_zscore(sub), "equal_zscore"


# 无 CSV 时用于 Tushare 拉取（可按需增删；需与积分权限匹配）
_DEFAULT_TS_SYMBOLS = [
    "600519.SH",
    "601318.SH",
    "600036.SH",
    "601166.SH",
    "600030.SH",
    "601398.SH",
    "601288.SH",
    "000858.SZ",
]


def _demo_price_wide() -> pd.DataFrame:
    """合成收盘价宽表（工作日 × 多标的），便于无 CSV / 无 Token 时跑通流水线。"""
    rng = np.random.default_rng(42)
    days = pd.bdate_range("2023-01-01", periods=280)
    syms = ["600519.SH", "000001.SZ", "601318.SH", "600036.SH", "601166.SH"]
    px = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.015, size=(len(days), len(syms))), axis=0)
    return pd.DataFrame(px, index=days, columns=syms)


def _resample_freq_alias(freq: str) -> str:
    return {"M": "ME", "Q": "QE", "A": "YE", "Y": "YE"}.get(freq, freq)


def _print_backtest_block(title: str, nav: pd.Series, meta: dict, stats: dict) -> None:
    print(title)
    print(
        "  回测设置: top_k=%s, 再平衡=%s, 手续费率=%s, 持仓权重=%s"
        % (
            meta.get("top_k"),
            meta.get("rebalance_freq"),
            meta.get("commission_rate"),
            meta.get("portfolio_weighting", "equal"),
        )
    )
    print("  单票权重上限: %s" % meta.get("max_position_weight", ""))
    print("  单次换手上限: %s" % meta.get("max_rebalance_turnover", ""))
    print("  调仓次数: %s" % meta.get("n_rebalances"))
    log = meta.get("rebalance_log") or []
    if log:
        print("  —— 各再平衡日 Top-%s 选股与权重 ——" % meta.get("top_k", "?"))
        for rec in log:
            dt = rec.get("date")
            dts = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
            picks = rec.get("picks") or []
            wt = rec.get("weights") or []
            wl = rec.get("weighting", "")
            parts = []
            for i, p in enumerate(picks):
                wv = wt[i] if i < len(wt) else float("nan")
                parts.append("%s(%.3f)" % (p, wv))
            print("    %s  [%s]  %s" % (dts, wl, ", ".join(parts)))
    print("  —— 绩效指标 ——")
    print("  年化收益:     %.4f  (约 %.2f%%)" % (stats["ann_return"], stats["ann_return"] * 100))
    print("  年化波动率:   %.4f  (约 %.2f%%)" % (stats["ann_vol"], stats["ann_vol"] * 100))
    print("  夏普比率:     %.4f  (无风险利率按 0)" % stats["sharpe"])
    print("  最大回撤:     %.4f  (约 %.2f%%)" % (stats["max_drawdown"], stats["max_drawdown"] * 100))
    print("  期末净值:     %.6f  (期初归一 1.0)" % nav.iloc[-1])
    print()


def main() -> None:
    settings = get_settings()
    print(
        "配置: portfolio_weighting=%s（equal=等权；max_sharpe=夏普；risk_parity=风险平价）"
        % settings.portfolio_weighting
    )
    demo_path = settings.data_dir / "prices_demo.csv"
    long_df: pd.DataFrame | None = None

    if demo_path.is_file():
        print("加载本地数据:", demo_path)
        long_df = load_prices_from_csv(demo_path)
        prices = long_to_wide(long_df, settings.price_col)
    else:
        try:
            print(
                "未找到 data/prices_demo.csv，尝试从 Tushare 拉取（区间 %s ~ %s）"
                % (settings.backtest_start, settings.backtest_end)
            )
            long_df = fetch_daily_panel(
                _DEFAULT_TS_SYMBOLS,
                settings.backtest_start,
                settings.backtest_end,
            )
            prices = long_to_wide(long_df, settings.price_col)
            print(
                "Tushare 已对齐: %d 只股票, %d 个交易日"
                % (prices.shape[1], prices.shape[0])
            )
        except Exception as e:
            print("Tushare 不可用，回退合成数据:", e)
            prices = _demo_price_wide()
            long_df = wide_to_long(prices, settings.price_col)

    if long_df is None:
        long_df = wide_to_long(prices, settings.price_col)

    print("\n构建因子面板（多列原始因子，只计算一次）…")
    try:
        panel = build_four_factor_panel(prices, long_df, settings)
    except Exception as e:
        print("因子面板构建失败:", e)
        return

    print(
        "面板形状: %d 行 × %d 列 (列=%s)"
        % (panel.shape[0], panel.shape[1], list(panel.columns))
    )

    data_quality_reports: dict[str, pd.DataFrame] = {}
    try:
        rf = _resample_freq_alias(settings.rebalance_freq)
        rebalance_dates = prices.resample(rf).last().index.intersection(prices.index)
        data_quality_reports = {
            "price_coverage": price_coverage(prices),
            "factor_coverage": factor_coverage(panel),
            "factor_daily_coverage": factor_daily_coverage(panel),
            "rebalance_coverage": rebalance_coverage(
                panel,
                prices,
                rebalance_dates,
                factors=DEFAULT_FACTOR_ORDER,
            ),
        }
        fc = data_quality_reports["factor_coverage"]
        print("\n========== 数据质量与覆盖率 ==========\n")
        for rec in fc.to_dict("records"):
            print(
                "【覆盖】%s  valid=%d/%d  coverage=%.2f%%"
                % (
                    rec["factor"],
                    rec["valid_cells"],
                    rec["total_cells"],
                    rec["coverage"] * 100.0,
                )
            )
        print()
    except Exception as e:
        print("数据质量报告生成失败（不影响主流程）:", e)

    if settings.persist_run_outputs:
        try:
            paths = save_run_cache(settings, long_df, prices, panel)
            print(
                "数据已落盘: %s"
                % ", ".join("%s=%s" % (k, v) for k, v in paths.items())
            )
        except Exception as e:
            print("落盘失败（不影响回测）:", e)
        if data_quality_reports:
            try:
                dq_paths = save_data_quality_reports(settings, data_quality_reports)
                print(
                    "数据质量报告已保存: %d 份 → 目录 %s"
                    % (len(dq_paths), (settings.output_dir / "data_quality").resolve())
                )
                plot_factor_coverage(
                    data_quality_reports["factor_coverage"],
                    title="因子有效覆盖率",
                    save_path=settings.output_dir / "data_quality" / "factor_coverage.png",
                )
                print(
                    "因子覆盖率图已保存:",
                    (settings.output_dir / "data_quality" / "factor_coverage.png").resolve(),
                )
            except Exception as e:
                print("数据质量报告落盘失败（不影响主流程）:", e)

    ic_by_name: dict[str, pd.Series] = {}
    backtest_meta_by_name: dict[str, dict] = {}
    performance_by_name: dict[str, dict] = {}
    print(
        "\n========== IC（截面 Spearman：因子@t vs 前瞻 %d 日收盘收益）==========\n"
        % settings.ic_forward_days
    )
    for fname in DEFAULT_FACTOR_ORDER:
        if fname not in panel.columns:
            print("【IC】%s 跳过: 面板中无该列\n" % fname)
            continue
        col = panel[fname]
        if col.notna().sum() == 0:
            print("【IC】%s 跳过: 整列无有效值\n" % fname)
            continue
        try:
            ic_ser = daily_ic_spearman(
                col,
                prices,
                forward_days=settings.ic_forward_days,
            )
            ic_by_name[fname] = ic_ser
            st = summarize_ic(ic_ser)
            print("【IC】%s" % fname)
            print(
                "  mean_IC=%.4f  std_IC=%.4f  IC_IR=%.4f  胜率=%.2f%%  有效日=%d"
                % (
                    st["mean_ic"],
                    st["std_ic"],
                    st["ic_ir"],
                    st["hit_rate"] * 100.0,
                    st["n_days"],
                )
            )
            print()
        except Exception as e:
            print("【IC】%s 跳过: %s\n" % (fname, e))
    try:
        sub_ic = panel[DEFAULT_FACTOR_ORDER].dropna(axis=1, how="all")
        if sub_ic.shape[1] > 0:
            fused_ic, fusion_mode = _build_fused_zscore_panel(sub_ic, ic_by_name, settings)
            ic_f = daily_ic_spearman(
                fused_ic,
                prices,
                forward_days=settings.ic_forward_days,
            )
            ic_by_name["FUSED_ZSCORE"] = ic_f
            stf = summarize_ic(ic_f)
            print("【IC】FUSED_ZSCORE（列: %s，融合=%s）" % (list(sub_ic.columns), fusion_mode))
            print(
                "  mean_IC=%.4f  std_IC=%.4f  IC_IR=%.4f  胜率=%.2f%%  有效日=%d"
                % (
                    stf["mean_ic"],
                    stf["std_ic"],
                    stf["ic_ir"],
                    stf["hit_rate"] * 100.0,
                    stf["n_days"],
                )
            )
            print()
    except Exception as e:
        print("【IC】FUSED_ZSCORE 跳过: %s\n" % e)

    if settings.persist_run_outputs and ic_by_name:
        try:
            ic_paths = save_ic_series(settings, ic_by_name)
            print("IC 序列已落盘:", ", ".join("%s=%s" % (k, v) for k, v in ic_paths.items()))
            print()
        except Exception as e:
            print("IC 落盘失败（不影响回测）:", e)
            print()

    nav_curves: dict[str, pd.Series] = {}

    print("\n========== 一、单因子回测（各列独立，横截面 Top-K + 配置持仓权重）==========\n")
    for fname in DEFAULT_FACTOR_ORDER:
        if fname not in panel.columns:
            print("【因子】%s 跳过: 面板中无该列\n" % fname)
            continue
        col = panel[fname]
        if col.notna().sum() == 0:
            print("【因子】%s 跳过: 整列无有效值\n" % fname)
            continue
        try:
            nav, meta = run_single_backtest(
                fname,
                prices=prices,
                settings=settings,
                factor_values=col,
            )
            stats = summarize(nav, periods=settings.trading_days_per_year)
            nav_curves[fname] = nav
            backtest_meta_by_name[fname] = meta
            performance_by_name[fname] = stats
            _print_backtest_block("【因子】%s" % fname, nav, meta, stats)
        except Exception as e:
            print("【因子】%s 跳过: %s\n" % (fname, e))

    print("========== 二、多因子融合（z-score；默认 IC 滞后滚动列权）==========\n")
    try:
        sub = panel[DEFAULT_FACTOR_ORDER].dropna(axis=1, how="all")
        if sub.shape[1] == 0:
            print("【融合】跳过: 无有效因子列\n")
            return
        fused, fusion_mode = _build_fused_zscore_panel(sub, ic_by_name, settings)
        nav_f, meta_f = run_multi_backtest(
            fused=fused,
            prices=prices,
            settings=settings,
            factor_name="FUSED_ZSCORE",
        )
        stats_f = summarize(nav_f, periods=settings.trading_days_per_year)
        nav_curves["FUSED_ZSCORE"] = nav_f
        backtest_meta_by_name["FUSED_ZSCORE"] = meta_f
        performance_by_name["FUSED_ZSCORE"] = stats_f
        _print_backtest_block(
            "【融合】FUSED_ZSCORE（参与列: %s；融合=%s）" % (list(sub.columns), fusion_mode),
            nav_f,
            meta_f,
            stats_f,
        )
    except Exception as e:
        print("【融合】跳过: %s\n" % e)

    benchmark_nav: pd.Series | None = None
    if nav_curves:
        try:
            nav_index = pd.DataFrame(nav_curves).index
            benchmark_nav = equal_weight_benchmark_nav(
                prices,
                dates=nav_index,
                price_col=settings.price_col,
            )
            if not benchmark_nav.empty:
                benchmark_stats = summarize(
                    benchmark_nav,
                    periods=settings.trading_days_per_year,
                )
                performance_by_name[benchmark_nav.name] = benchmark_stats
                for name, nav in nav_curves.items():
                    performance_by_name[name].update(
                        summarize_excess(
                            nav,
                            benchmark_nav,
                            periods=settings.trading_days_per_year,
                        )
                    )
                print("========== 三、基准与超额收益 ==========\n")
                print("【基准】%s（股票池每日等权）" % benchmark_nav.name)
                print(
                    "  年化收益: %.4f  年化波动: %.4f  夏普: %.4f  最大回撤: %.4f"
                    % (
                        benchmark_stats["ann_return"],
                        benchmark_stats["ann_vol"],
                        benchmark_stats["sharpe"],
                        benchmark_stats["max_drawdown"],
                    )
                )
                for name in nav_curves:
                    st = performance_by_name[name]
                    print(
                        "【超额】%s  excess_ann_return=%.4f  tracking_error=%.4f  IR=%.4f"
                        % (
                            name,
                            st["excess_ann_return"],
                            st["tracking_error"],
                            st["information_ratio"],
                        )
                    )
                print()
            else:
                print("【基准】跳过: 股票池等权基准为空\n")
        except Exception as e:
            print("【基准】跳过: %s\n" % e)

    turnover_by_name: dict[str, pd.DataFrame] = {}
    if backtest_meta_by_name:
        print("========== 四、换手率与交易成本 ==========\n")
        for name, meta in backtest_meta_by_name.items():
            log = meta.get("rebalance_log") or []
            tf = turnover_frame(log, commission_rate=settings.commission_rate)
            turnover_by_name[name] = tf
            st = summarize_turnover(log, commission_rate=settings.commission_rate)
            if name in performance_by_name:
                performance_by_name[name].update(st)
            print(
                "【换手】%s  avg=%.4f  max=%.4f  total=%.4f  estimated_cost=%.6f"
                % (
                    name,
                    st["avg_turnover"],
                    st["max_turnover"],
                    st["total_turnover"],
                    st["estimated_total_cost"],
                )
            )
        print()

    concentration_by_name: dict[str, pd.DataFrame] = {}
    concentration_summary_by_name: dict[str, dict] = {}
    if backtest_meta_by_name:
        print("========== 五、风险暴露与持仓集中度 ==========\n")
        for name, meta in backtest_meta_by_name.items():
            log = meta.get("rebalance_log") or []
            cf = concentration_frame(log)
            concentration_by_name[name] = cf
            st = summarize_concentration(log)
            concentration_summary_by_name[name] = st
            if name in performance_by_name:
                performance_by_name[name].update(st)
            print(
                "【集中度】%s  avg_effective_n=%.2f  min_effective_n=%.2f  avg_top1=%.2f%%  max_hhi=%.4f"
                % (
                    name,
                    st["avg_effective_n"],
                    st["min_effective_n"],
                    st["avg_top1_weight"] * 100.0,
                    st["max_hhi"],
                )
            )
        print()

    if settings.persist_run_outputs and ic_by_name:
        outd = settings.output_dir
        outd.mkdir(parents=True, exist_ok=True)
        try:
            ic_frame = pd.DataFrame(ic_by_name)
            plot_ic(
                ic_frame,
                title="日 IC 对比（Spearman，前瞻 %d 交易日）" % settings.ic_forward_days,
                save_path=outd / "ic_compare.png",
                rolling_window=None,
            )
            for name, ser in ic_by_name.items():
                safe = name.replace("/", "_")
                plot_ic(
                    ser,
                    title="%s 日 IC（前瞻 %d 日）" % (name, settings.ic_forward_days),
                    save_path=outd / ("ic_timeseries_%s.png" % safe),
                    rolling_window=20,
                )
            print("IC 图已保存:", outd / "ic_compare.png", "及各因子 ic_timeseries_*.png")
        except Exception as e:
            print("IC 作图失败（不影响主流程）:", e)

    if settings.persist_run_outputs and backtest_meta_by_name:
        outd = settings.output_dir
        outd.mkdir(parents=True, exist_ok=True)
        try:
            cfg_path = save_run_config(settings)
            perf_path = save_performance_summary(settings, performance_by_name)
            log_paths = save_rebalance_logs(settings, backtest_meta_by_name)
            turnover_paths = save_turnover_logs(settings, turnover_by_name)
            risk_paths = save_risk_exposure_logs(settings, concentration_by_name)
            risk_summary_path = save_risk_exposure_summary(settings, concentration_summary_by_name)
            print("运行配置已保存:", cfg_path.resolve())
            print("绩效汇总已保存:", perf_path.resolve())
            if log_paths:
                print("调仓日志已保存: %d 份 → 目录 %s" % (len(log_paths), (outd / "rebalance_logs").resolve()))
            if turnover_paths:
                print("换手日志已保存: %d 份 → 目录 %s" % (len(turnover_paths), (outd / "turnover_logs").resolve()))
            if risk_paths:
                print(
                    "集中度日志已保存: %d 份 → 目录 %s"
                    % (len(risk_paths), (outd / "risk_exposure" / "concentration_logs").resolve())
                )
                print("集中度汇总已保存:", risk_summary_path.resolve())
        except Exception as e:
            print("实验记录落盘失败（不影响主流程）:", e)
        try:
            n_w = 0
            for name, meta in backtest_meta_by_name.items():
                log = meta.get("rebalance_log") or []
                if not log:
                    continue
                wf = rebalance_log_to_weights_frame(log)
                if wf.empty:
                    continue
                safe = name.replace("/", "_")
                plot_weights(
                    wf,
                    title="%s 再平衡权重（堆叠）" % name,
                    save_path=outd / ("weights_%s.png" % safe),
                    kind="area",
                )
                n_w += 1
            if n_w:
                print("权重堆叠图已保存: %d 张 → 目录 %s" % (n_w, outd.resolve()))
        except Exception as e:
            print("权重作图失败（不影响主流程）:", e)
        try:
            tw = turnover_wide(turnover_by_name)
            if not tw.empty:
                plot_turnover(
                    tw,
                    title="各策略逐期换手率",
                    save_path=outd / "turnover_compare.png",
                )
                print("换手率对比图已保存:", (outd / "turnover_compare.png").resolve())
        except Exception as e:
            print("换手率作图失败（不影响主流程）:", e)
        try:
            en = effective_n_wide(concentration_by_name)
            if not en.empty:
                plot_effective_n(
                    en,
                    title="各策略有效持仓数（1 / HHI）",
                    save_path=outd / "risk_exposure" / "effective_n_compare.png",
                )
                print(
                    "有效持仓数对比图已保存:",
                    (outd / "risk_exposure" / "effective_n_compare.png").resolve(),
                )
        except Exception as e:
            print("集中度作图失败（不影响主流程）:", e)

    if nav_curves:
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        chart_path = settings.output_dir / "nav_compare.png"
        try:
            plot_df = pd.DataFrame(nav_curves)
            if benchmark_nav is not None and not benchmark_nav.empty:
                plot_df[benchmark_nav.name] = benchmark_nav
            plot_nav(
                plot_df,
                title="单因子 / 融合 / 基准（归一化净值，起点≈1）",
                save_path=chart_path,
                normalize=True,
            )
            print("净值对比图已保存:", chart_path.resolve())
        except Exception as e:
            print("作图失败:", e)
        if benchmark_nav is not None and not benchmark_nav.empty:
            excess_chart_path = settings.output_dir / "excess_nav_compare.png"
            try:
                xnav = excess_nav_frame(nav_curves, benchmark_nav)
                if not xnav.empty:
                    plot_nav(
                        xnav,
                        title="相对股票池等权基准的超额净值（起点≈1）",
                        save_path=excess_chart_path,
                        normalize=True,
                    )
                    print("超额净值图已保存:", excess_chart_path.resolve())
            except Exception as e:
                print("超额净值作图失败:", e)


if __name__ == "__main__":
    main()
