"""
日终纸面交易调度封装：记录一次调度运行的输入、输出和退出码。
"""
from __future__ import annotations

import contextlib
import io
import shlex
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from config import Settings
from live import daily_paper_cli


def scheduler_log_dir(settings: Settings) -> Path:
    return settings.output_dir / "scheduler_logs"


def _date_to_str(value: Any | None) -> str:
    if value is None or value == "":
        return datetime.now().strftime("%Y-%m-%d")
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _format_args(args: Sequence[str]) -> str:
    if not args:
        return "(default)"
    return " ".join(shlex.quote(str(x)) for x in args)


def run_scheduled_daily_paper(
    settings: Settings,
    *,
    daily_args: Sequence[str] | None = None,
    log_date: Any | None = None,
) -> dict[str, Any]:
    """
    运行一次日终纸面交易，并把 stdout/stderr 写入调度日志。

    这个函数不负责常驻定时；cron、launchd 或服务器调度器只需要每天调用
    `scripts/run_scheduled_daily_paper.py` 即可。
    """
    args = list(daily_args or [])
    log_dir = scheduler_log_dir(settings)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_date_s = _date_to_str(log_date)
    log_path = log_dir / ("%s.log" % run_date_s)

    started_at = datetime.now()
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    parsed_args = daily_paper_cli.build_arg_parser().parse_args(args)
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        try:
            exit_code = int(daily_paper_cli.run_daily_paper_from_args(settings, parsed_args))
        except Exception:
            traceback.print_exc()
            exit_code = 1
    finished_at = datetime.now()

    stdout = stdout_buf.getvalue().rstrip()
    stderr = stderr_buf.getvalue().rstrip()
    lines = [
        "=" * 72,
        "scheduled_daily_paper",
        "started_at=%s" % started_at.isoformat(timespec="seconds"),
        "finished_at=%s" % finished_at.isoformat(timespec="seconds"),
        "exit_code=%d" % exit_code,
        "daily_args=%s" % _format_args(args),
        "",
        "[stdout]",
        stdout if stdout else "(empty)",
        "",
        "[stderr]",
        stderr if stderr else "(empty)",
        "",
    ]
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")

    return {
        "exit_code": exit_code,
        "log_path": log_path,
        "stdout": stdout,
        "stderr": stderr,
        "started_at": started_at,
        "finished_at": finished_at,
    }
