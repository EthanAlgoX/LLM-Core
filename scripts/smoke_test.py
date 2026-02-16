#!/usr/bin/env python3
"""
自动化烟雾测试（面试模式）。

默认行为：
1) 所有模块做 `--help` 检查
2) 轻量模块做 `--toy` 实跑
3) LLaMA-Factory 重模块做启动检测（超时 + 产物检查）
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run as interview_run  # noqa: E402


HEAVY_STARTUP_MODULES = {
    "sft",
    "dpo",
    "ppo",
    "policy_gradient",
    "actor_critic",
    "rlhf",
}

STARTUP_ARTIFACTS = {
    "sft": "train_sft_auto.json",
    "dpo": "train_dpo_auto.json",
    "ppo": "train_ppo_auto.json",
    "policy_gradient": "train_policy_gradient_auto.json",
    "actor_critic": "train_actor_critic_auto.json",
    "rlhf": "train_rlhf_auto.json",
}


@dataclass
class CaseResult:
    phase: str
    module: str
    status: str
    elapsed_sec: float
    command: str
    detail: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke tests for interview learning flows.")
    parser.add_argument(
        "--modules",
        default="",
        help="仅测试指定模块，逗号分隔，如 sft,grpo,mdp",
    )
    parser.add_argument("--help-timeout", type=int, default=30)
    parser.add_argument("--toy-timeout", type=int, default=240)
    parser.add_argument("--startup-timeout", type=int, default=60)
    parser.add_argument("--allow-fail", action="store_true", help="即使失败也返回 0。")
    return parser.parse_args()


def iter_modules(selected: str) -> list[str]:
    all_modules = sorted(interview_run.MODULES.keys())
    if not selected.strip():
        return all_modules
    picked = [x.strip() for x in selected.split(",") if x.strip()]
    unknown = [x for x in picked if x not in interview_run.MODULES]
    if unknown:
        raise ValueError(f"Unknown modules: {unknown}")
    return picked


def module_output_dir(module: str) -> Path | None:
    spec = interview_run.MODULES[module]
    toy_args = list(spec.toy_args)
    if "--output-dir" not in toy_args:
        return None
    idx = toy_args.index("--output-dir")
    if idx + 1 >= len(toy_args):
        return None
    output_arg = toy_args[idx + 1]
    script_path = ROOT / spec.script
    module_dir = script_path.parent.parent
    return (module_dir / output_arg).resolve()


def kill_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        pass
    time.sleep(0.8)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def run_cmd(cmd: list[str], timeout: int) -> tuple[str, float, str, str]:
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        elapsed = round(time.time() - start, 2)
        status = "PASS" if proc.returncode == 0 else "FAIL"
        return status, elapsed, stdout, stderr
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        stdout, stderr = proc.communicate(timeout=3)
        elapsed = round(time.time() - start, 2)
        return "TIMEOUT", elapsed, stdout, stderr


def tail(text: str, n: int = 14) -> str:
    lines = [x for x in text.splitlines() if x.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-n:])


def run_help_phase(modules: Iterable[str], timeout: int) -> list[CaseResult]:
    results: list[CaseResult] = []
    for module in modules:
        spec = interview_run.MODULES[module]
        script = ROOT / spec.script
        cmd = [sys.executable, str(script), "--help"]
        status, elapsed, stdout, stderr = run_cmd(cmd, timeout=timeout)
        results.append(
            CaseResult(
                phase="help",
                module=module,
                status=status,
                elapsed_sec=elapsed,
                command=" ".join(cmd),
                stdout_tail=tail(stdout),
                stderr_tail=tail(stderr),
            )
        )
    return results


def run_toy_phase(modules: Iterable[str], toy_timeout: int, startup_timeout: int) -> list[CaseResult]:
    results: list[CaseResult] = []
    run_py = ROOT / "run.py"

    for module in modules:
        cmd = [sys.executable, str(run_py), "--module", module, "--toy"]
        timeout = startup_timeout if module in HEAVY_STARTUP_MODULES else toy_timeout
        status, elapsed, stdout, stderr = run_cmd(cmd, timeout=timeout)

        detail = ""
        if module in HEAVY_STARTUP_MODULES and status == "TIMEOUT":
            out_dir = module_output_dir(module)
            artifact = STARTUP_ARTIFACTS[module]
            artifact_path = out_dir / artifact if out_dir else None
            if artifact_path and artifact_path.exists():
                status = "STARTED"
                detail = f"startup artifact found: {artifact_path}"
            else:
                detail = f"timeout but startup artifact missing: {artifact_path}"

        results.append(
            CaseResult(
                phase="toy",
                module=module,
                status=status,
                elapsed_sec=elapsed,
                command=" ".join(cmd),
                detail=detail,
                stdout_tail=tail(stdout),
                stderr_tail=tail(stderr),
            )
        )
    return results


def save_report(results: list[CaseResult]) -> Path:
    out_dir = ROOT / "output" / "smoke_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"smoke_{ts}.json"
    report_path.write_text(
        json.dumps([asdict(x) for x in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    args = parse_args()
    modules = iter_modules(args.modules)

    print(f"[smoke] modules={len(modules)}")
    help_results = run_help_phase(modules=modules, timeout=args.help_timeout)
    toy_results = run_toy_phase(modules=modules, toy_timeout=args.toy_timeout, startup_timeout=args.startup_timeout)
    results = help_results + toy_results

    report_path = save_report(results)

    counts: dict[str, int] = {}
    for x in results:
        counts[x.status] = counts.get(x.status, 0) + 1

    print("\n[smoke] summary")
    for k in sorted(counts.keys()):
        print(f"- {k}: {counts[k]}")
    print(f"- report: {report_path}")

    failed = [x for x in results if x.status in {"FAIL", "TIMEOUT"}]
    if failed:
        print("\n[smoke] failed cases")
        for x in failed:
            print(f"- {x.phase}/{x.module}: {x.status} ({x.elapsed_sec}s)")
            if x.detail:
                print(f"  detail: {x.detail}")

    if failed and not args.allow_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
