#!/usr/bin/env python3
"""
LLaMA-Factory 的单模式 RLHF 流程（基于 PPO 近似实现）。

一、RLHF 原理（面向实现）
1) 先有可用的基础模型（通常来自预训练与 SFT）。
2) 奖励模型对模型回答进行打分，提供偏好信号。
3) 用强化学习更新策略模型，让高奖励回答的概率上升。
4) 在 LLM 场景中，常用 PPO 实现 RLHF 强化学习阶段。

二、代码框架（从入口到结果）
1) `parse_args`：读取训练与可视化参数。
2) `resolve_factory_dir`：定位 LLaMA-Factory。
3) `build_train_config`：构建最小可用 RLHF（PPO）配置。
4) `run_train`：执行训练（CLI 与模块入口双回退）。
5) `export_learning_artifacts`：导出 JSON/CSV/曲线图/summary。
6) `main`：串联完整流程并输出结果目录。

用法：
  python code/rlhf.py --reward-model <奖励模型路径或名称>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

DEFAULT_OUTPUT_DIR = "output"


def parse_args() -> argparse.Namespace:
    """解析命令行参数，返回 RLHF 训练与可视化所需配置。"""
    parser = argparse.ArgumentParser(description="Run RLHF (PPO backend) and export visualization artifacts.")

    # 模型与数据：策略模型 + 奖励模型是 RLHF 强化学习阶段的核心输入。
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--reward-model", required=True, help="奖励模型路径或名称（RLHF 必填）。")
    parser.add_argument(
        "--reward-model-type",
        default="full",
        choices=["full", "lora", "api"],
        help="奖励模型类型。",
    )
    parser.add_argument("--ref-model", default=None, help="可选：参考模型路径；不传则使用默认参考策略。")
    parser.add_argument("--template", default="qwen")
    parser.add_argument("--dataset", default="alpaca_en_demo")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    # 训练核心超参：保持最简但关键，便于教学与复现。
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=50)

    # PPO 参数：作为 RLHF 强化学习阶段的稳定实现。
    parser.add_argument("--ppo-buffer-size", type=int, default=1)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-target", type=float, default=6.0)
    parser.add_argument("--ppo-score-norm", action="store_true")
    parser.add_argument("--ppo-whiten-rewards", action="store_true")

    # 曲线平滑参数：用于观察趋势，降低抖动干扰。
    parser.add_argument("--ema-alpha", type=float, default=0.2, help="EMA smoothing factor for plotting curves.")
    return parser.parse_args()


def detect_device_and_precision() -> dict[str, Any]:
    """检测可用设备并选择推荐混合精度配置。"""
    if torch.cuda.is_available():
        return {"device": "cuda", "bf16": True, "fp16": False}
    if torch.backends.mps.is_available():
        return {"device": "mps", "bf16": False, "fp16": True}
    return {"device": "cpu", "bf16": False, "fp16": False}


def resolve_factory_dir(code_dir: Path) -> Path:
    """定位 LLaMA-Factory：优先 rlhf 本地，其次复用 sft 目录。"""
    candidates = [
        code_dir / "LLaMA-Factory",
        code_dir.parent.parent / "sft" / "LLaMA-Factory",
        code_dir.parent.parent / "sft" / "code" / "LLaMA-Factory",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "LLaMA-Factory directory not found. Expected one of:\n"
        f"- {candidates[0]}\n"
        f"- {candidates[1]}\n"
        f"- {candidates[2]}"
    )


def resolve_output_dir(base_dir: Path, output_dir: str) -> Path:
    """解析输出目录：相对路径按 rlhf 目录解析，绝对路径原样使用。"""
    out = Path(output_dir)
    if not out.is_absolute():
        out = (base_dir / out).resolve()
    return out


def ensure_layout_dirs(module_dir: Path, output_arg: str) -> dict[str, Path]:
    """创建并返回标准目录布局：code/data/models/output/checkpoints。"""
    output_dir = resolve_output_dir(module_dir, output_arg)
    layout = {
        "code": module_dir / "code",
        "data": module_dir / "data",
        "models": module_dir / "models",
        "output": output_dir,
        "checkpoints": module_dir / "checkpoints",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def build_train_config(args: argparse.Namespace, runtime: dict[str, Any], checkpoints_dir: Path) -> dict[str, Any]:
    """根据参数与运行时环境构造最小可用 RLHF（PPO 后端）配置。"""
    cfg = {
        # 说明：此脚本聚焦 RLHF 的强化学习阶段，对应 LLaMA-Factory 的 stage=ppo。
        "stage": "ppo",
        "do_train": True,
        "model_name_or_path": args.model_id,
        "reward_model": args.reward_model,
        "reward_model_type": args.reward_model_type,
        "dataset": args.dataset,
        "template": args.template,
        "finetuning_type": "lora",
        "lora_target": "all",
        "output_dir": str(checkpoints_dir),
        "overwrite_output_dir": True,
        "save_only_model": True,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "lr_scheduler_type": "cosine",
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "max_samples": args.max_samples,
        "max_grad_norm": 1.0,
        "report_to": "none",
        "ppo_buffer_size": args.ppo_buffer_size,
        "ppo_epochs": args.ppo_epochs,
        "ppo_target": args.ppo_target,
        "ppo_score_norm": args.ppo_score_norm,
        "ppo_whiten_rewards": args.ppo_whiten_rewards,
        "bf16": runtime["bf16"],
        "fp16": runtime["fp16"],
    }
    if args.ref_model:
        cfg["ref_model"] = args.ref_model
    return cfg


def run_train(factory_dir: Path, config_path: Path) -> None:
    """执行训练，优先使用 CLI，失败时回退到模块入口。"""
    env = os.environ.copy()
    env["FORCE_TORCHRUN"] = "1"
    src_dir = str(factory_dir / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

    commands: list[list[str]] = []
    if shutil.which("llamafactory-cli") is not None:
        commands.append(["llamafactory-cli", "train", str(config_path)])
    commands.append([sys.executable, "-m", "llamafactory.cli", "train", str(config_path)])

    last_error: Exception | None = None
    for cmd in commands:
        try:
            subprocess.run(cmd, cwd=str(factory_dir), check=True, env=env)
            return
        except Exception as exc:
            last_error = exc
            print(f"[WARN] Training command failed: {' '.join(cmd)}")

    raise RuntimeError(
        "Failed to start RLHF training via both CLI and module entrypoints. "
        "Please check LLaMA-Factory dependencies in your `finetune` environment."
    ) from last_error


def move_model_artifacts(checkpoints_dir: Path, models_dir: Path) -> None:
    """将最终模型文件从 checkpoints 根目录移动到 models 目录。"""
    artifact_names = [
        "README.md",
        "adapter_config.json",
        "adapter_model.safetensors",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "training_args.bin",
        "value_head.safetensors",
    ]
    for name in artifact_names:
        src = checkpoints_dir / name
        if not src.exists():
            continue
        dst = models_dir / name
        if dst.exists():
            if dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        shutil.move(str(src), str(dst))


def _to_float(v: Any) -> float | None:
    """将任意日志值转换为 float，无法转换时返回 None。"""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _ema(values: list[float], alpha: float) -> list[float]:
    """计算序列的指数滑动平均（EMA）并返回平滑后的新序列。"""
    out, prev = [], None
    for x in values:
        prev = x if prev is None else alpha * x + (1.0 - alpha) * prev
        out.append(prev)
    return out


def find_trainer_state(checkpoints_dir: Path) -> Path | None:
    """查找 trainer_state.json，优先根目录，其次最新 checkpoint。"""
    direct = checkpoints_dir / "trainer_state.json"
    if direct.exists():
        return direct

    checkpoints = []
    for d in checkpoints_dir.glob("checkpoint-*"):
        if d.is_dir():
            m = re.match(r"checkpoint-(\d+)$", d.name)
            if m:
                checkpoints.append((int(m.group(1)), d))
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    for _, ckpt in checkpoints:
        candidate = ckpt / "trainer_state.json"
        if candidate.exists():
            return candidate
    return None


def _extract_series(rows: list[dict[str, Any]], keys: list[str]) -> tuple[list[int], list[float]]:
    """按候选键顺序提取曲线数据，返回第一条可用序列。"""
    for key in keys:
        x, y = [], []
        for row in rows:
            if row.get(key) is None:
                continue
            x.append(int(row["step"]))
            y.append(float(row[key]))
        if y:
            return x, y
    return [], []


def export_learning_artifacts(checkpoints_dir: Path, output_dir: Path, ema_alpha: float) -> Path:
    """导出训练日志、CSV、曲线图与摘要，返回产物目录路径。"""
    state_path = find_trainer_state(checkpoints_dir)
    if state_path is None:
        raise FileNotFoundError(f"No trainer_state.json found under: {checkpoints_dir}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    log_history = state.get("log_history", [])
    if not log_history:
        raise RuntimeError(f"log_history is empty in: {state_path}")

    metrics_dir = output_dir / "rlhf_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / "log_history.json").write_text(
        json.dumps(log_history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    keys = [
        "step",
        "epoch",
        "loss",
        "reward",
        "kl",
        "objective/kl",
        "learning_rate",
        "ppo/loss/total",
        "ppo/learning_rate",
    ]

    rows = []
    for item in log_history:
        if "step" not in item:
            continue
        row = {"step": int(item.get("step", 0))}
        for key in keys:
            if key == "step":
                continue
            row[key] = _to_float(item.get(key))
        rows.append(row)

    with (metrics_dir / "training_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"`matplotlib` is required to generate visualization: {exc}") from exc

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    x, y = _extract_series(rows, ["loss", "ppo/loss/total"])
    axes[0, 0].plot(x, y, marker="o", alpha=0.45, label="loss(raw)")
    if y:
        axes[0, 0].plot(x, _ema(y, ema_alpha), linewidth=2, label=f"loss(ema={ema_alpha})")
    axes[0, 0].set_title("RLHF Loss")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    x, y = _extract_series(rows, ["reward"])
    axes[0, 1].plot(x, y, marker="o", alpha=0.45, label="reward(raw)")
    if y:
        axes[0, 1].plot(x, _ema(y, ema_alpha), linewidth=2, label=f"reward(ema={ema_alpha})")
    axes[0, 1].set_title("Reward")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    x, y = _extract_series(rows, ["kl", "objective/kl"])
    axes[1, 0].plot(x, y, marker="o", alpha=0.7, color="#ff7f0e", label="kl")
    axes[1, 0].set_title("KL Divergence")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    x, y = _extract_series(rows, ["learning_rate", "ppo/learning_rate"])
    axes[1, 1].plot(x, y, marker="o", alpha=0.7, color="#2ca02c", label="lr")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(metrics_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    summary = {
        "total_steps": len(rows),
        "final_step": rows[-1]["step"] if rows else None,
        "final_loss": next((r["loss"] for r in reversed(rows) if r["loss"] is not None), None),
        "final_reward": next((r["reward"] for r in reversed(rows) if r["reward"] is not None), None),
        "final_kl": next((r["kl"] for r in reversed(rows) if r["kl"] is not None), None),
        "best_reward": max((r["reward"] for r in rows if r["reward"] is not None), default=None),
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics_dir


def main() -> None:
    """主流程入口：生成配置、执行训练、导出可视化结果。"""
    args = parse_args()
    code_dir = Path(__file__).resolve().parent
    module_dir = code_dir.parent
    layout = ensure_layout_dirs(module_dir=module_dir, output_arg=args.output_dir)
    factory_dir = resolve_factory_dir(code_dir)
    config_path = layout["output"] / "train_rlhf_auto.json"

    runtime = detect_device_and_precision()
    print(f"Runtime: device={runtime['device']}, bf16={runtime['bf16']}, fp16={runtime['fp16']}")

    config = build_train_config(args=args, runtime=runtime, checkpoints_dir=layout["checkpoints"])
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Config written: {config_path}")

    run_train(factory_dir=factory_dir, config_path=config_path)
    move_model_artifacts(checkpoints_dir=layout["checkpoints"], models_dir=layout["models"])
    metrics_dir = export_learning_artifacts(
        checkpoints_dir=layout["checkpoints"],
        output_dir=layout["output"],
        ema_alpha=args.ema_alpha,
    )
    print(f"RLHF done. Visualization exported to: {metrics_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
