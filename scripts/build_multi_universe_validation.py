#!/usr/bin/env python3
"""从多个股票池回测输出目录生成跨股票池验证表。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.multi_universe_validation import (
    collect_factor_universe_performance,
    collect_strategy_universe_performance,
    save_multi_universe_validation_outputs,
    summarize_factor_universe_robustness,
    summarize_strategy_universe_robustness,
)


def _parse_universe(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--universe 需要格式 name=output_dir")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("股票池名称不能为空")
    return name, Path(path).expanduser()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe",
        action="append",
        type=_parse_universe,
        required=True,
        help="股票池输出目录，格式 name=output_dir，可重复传入",
    )
    parser.add_argument(
        "--output-dir",
        default="output/multi_universe_validation",
        help="汇总结果输出目录",
    )
    args = parser.parse_args()

    universe_outputs = dict(args.universe)
    strategy_perf = collect_strategy_universe_performance(universe_outputs)
    strategy_robust = summarize_strategy_universe_robustness(strategy_perf)
    factor_perf = collect_factor_universe_performance(universe_outputs)
    factor_robust = summarize_factor_universe_robustness(factor_perf)
    paths = save_multi_universe_validation_outputs(
        args.output_dir,
        strategy_performance=strategy_perf,
        strategy_robustness=strategy_robust,
        factor_performance=factor_perf,
        factor_robustness=factor_robust,
    )

    print("多股票池验证已保存:")
    for name, path in paths.items():
        print("  %s=%s" % (name, path))
    if not strategy_robust.empty:
        print("\n策略稳健性摘要:")
        for rec in strategy_robust.head(10).to_dict("records"):
            print(
                "  %s  status=%s  avg_ann=%.4f  avg_excess=%.4f  pos_excess=%.2f%%"
                % (
                    rec["strategy"],
                    rec["status"],
                    rec["avg_ann_return"],
                    rec["avg_excess_ann_return"],
                    rec["positive_excess_rate"] * 100.0
                    if rec["positive_excess_rate"] == rec["positive_excess_rate"]
                    else float("nan"),
                )
            )
    if not factor_robust.empty:
        print("\n因子稳健性摘要:")
        for rec in factor_robust.head(10).to_dict("records"):
            print(
                "  %s  status=%s  avg_excess=%.4f  pos_excess=%.2f%%"
                % (
                    rec["factor"],
                    rec["status"],
                    rec["avg_excess_ann_return"],
                    rec["positive_excess_rate"] * 100.0
                    if rec["positive_excess_rate"] == rec["positive_excess_rate"]
                    else float("nan"),
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
