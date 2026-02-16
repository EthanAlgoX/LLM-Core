#!/usr/bin/env python3
"""
LLaMA-Factory 的单模式 DPO 流程。

一、DPO 原理（面向实现）
1) 每条样本包含 prompt、chosen（偏好回答）、rejected（非偏好回答）。
2) DPO 直接优化“chosen 相对 rejected 的偏好概率”，不显式训练奖励模型。
3) `pref_beta` 控制偏好强度，值越大通常更新更激进。
4) 通过 loss、reward margin、reward accuracy 等指标观察对齐效果。

二、代码框架（从入口到结果）
1) `build_default_args`：读取训练与可视化参数。
2) `detect_device_and_precision`：自动选择设备与混合精度。
3) `build_train_config`：构建最小可用 DPO 配置。
4) `run_train`：执行训练（含 CLI 与模块入口回退）。
5) `export_learning_artifacts`：导出 JSON/CSV/曲线图/summary。
6) `main`：串联完整流程并输出结果目录。

用法：
  python code/dpo.py
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
CORE_READING_ORDER = [
    ("build_default_args", "固定默认参数，直接 python 运行"),
    ("run_core_dpo_flow", "DPO 主流程：环境 -> 配置 -> 训练 -> 导出"),
    ("run_train", "调用 LLaMA-Factory 执行训练"),
    ("export_learning_artifacts", "导出 JSON/CSV/可视化曲线"),
    ("main", "程序入口"),
]


def build_default_args() -> argparse.Namespace:
    """解析命令行参数，返回 DPO 训练与可视化所需配置。"""
    parser = argparse.ArgumentParser(description="Run DPO and always export visualization artifacts.")
    # 模型与数据：尽量选用 LLaMA-Factory 自带 DPO 演示集，开箱可跑。
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--template", default="qwen")
    parser.add_argument("--dataset", default="dpo_zh_demo")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    # 训练核心超参：保留最关键参数，便于学习与快速实验。
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--pref-beta", type=float, default=0.1)
    parser.add_argument(
        "--pref-loss",
        default="sigmoid",
        choices=["sigmoid", "orpo", "simpo"],
        help="偏好损失类型。",
    )
    # 可视化参数：用于平滑 loss 曲线，降低训练噪声。
    parser.add_argument("--ema-alpha", type=float, default=0.2, help="EMA smoothing factor for plotting curves.")
    return parser.parse_known_args([])[0]


def print_core_learning_guide() -> None:
    """打印新手学习主线，帮助快速定位关键函数。"""
    print("=== DPO 新手学习主线（建议按顺序阅读）===", flush=True)
    for idx, (func, desc) in enumerate(CORE_READING_ORDER, start=1):
        print(f"{idx}. {func}: {desc}", flush=True)
    print("====================================", flush=True)


def detect_device_and_precision() -> dict[str, Any]:
    """检测可用设备并选择推荐混合精度配置。"""
    if torch.cuda.is_available():
        return {"device": "cuda", "bf16": True, "fp16": False}
    if torch.backends.mps.is_available():
        return {"device": "mps", "bf16": False, "fp16": True}
    return {"device": "cpu", "bf16": False, "fp16": False}


def resolve_factory_dir(base_dir: Path) -> Path:
    """定位 LLaMA-Factory 目录：优先 dpo 本地，其次复用 sft 目录。"""
    candidates = [
        base_dir / "LLaMA-Factory",
        base_dir.parent.parent / "sft" / "LLaMA-Factory",
        base_dir.parent.parent / "sft" / "code" / "LLaMA-Factory",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "LLaMA-Factory directory not found. Expected one of:\n"
        f"- {candidates[0]}\n"
        f"- {candidates[1]}"
    )


def resolve_output_dir(base_dir: Path, output_dir: str) -> Path:
    """解析输出目录：相对路径按 dpo 目录解析，绝对路径原样使用。"""
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


def build_train_config(args: argparse.Namespace, runtime: dict[str, Any], output_path: Path) -> dict[str, Any]:
    """根据参数与运行时环境构造最小可用的 LLaMA-Factory DPO 配置。"""
    return {
        "stage": "dpo",  # 训练阶段：直接偏好优化。
        "do_train": True,  # 启用训练流程。
        "model_name_or_path": args.model_id,  # 基座模型名称或路径。
        "dataset": args.dataset,  # 偏好数据集（包含 chosen/rejected）。
        "template": args.template,  # 对话模板。
        "finetuning_type": "lora",  # 参数高效微调。
        "lora_target": "all",  # 尽量覆盖可注入线性层。
        "pref_beta": args.pref_beta,  # DPO 偏好强度系数。
        "pref_loss": args.pref_loss,  # 偏好损失类型。
        "output_dir": str(output_path),  # 模型与日志输出目录。
        "overwrite_output_dir": True,  # 复用同一输出目录，避免目录膨胀。
        "per_device_train_batch_size": args.batch_size,  # 单设备 batch。
        "gradient_accumulation_steps": args.grad_accum,  # 梯度累积。
        "lr_scheduler_type": "cosine",  # 学习率调度策略。
        "logging_steps": args.logging_steps,  # 日志记录间隔。
        "save_steps": args.save_steps,  # checkpoint 保存间隔。
        "learning_rate": args.learning_rate,  # 初始学习率。
        "num_train_epochs": args.num_train_epochs,  # 训练轮数。
        "max_samples": args.max_samples,  # 样本上限（用于快速实验）。
        "max_grad_norm": 1.0,  # 梯度裁剪阈值。
        "report_to": "none",  # 关闭外部实验平台上报。
        "bf16": runtime["bf16"],  # 是否启用 bfloat16。
        "fp16": runtime["fp16"],  # 是否启用 float16。
    }


def run_train(factory_dir: Path, config_path: Path) -> None:
    """执行 DPO 训练，优先使用 CLI，失败时回退到模块入口。"""
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
        "Failed to start DPO training via both CLI and module entrypoints. "
        "Please check LLaMA-Factory dependencies in your `finetune` environment."
    ) from last_error


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


def export_learning_artifacts(checkpoints_dir: Path, output_dir: Path, ema_alpha: float) -> Path:
    """导出训练日志、CSV、曲线图与摘要，返回产物目录路径。"""
    state_path = find_trainer_state(checkpoints_dir)
    if state_path is None:
        raise FileNotFoundError(f"No trainer_state.json found under: {checkpoints_dir}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    log_history = state.get("log_history", [])
    if not log_history:
        raise RuntimeError(f"log_history is empty in: {state_path}")

    metrics_dir = output_dir / "dpo_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / "log_history.json").write_text(
        json.dumps(log_history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    keys = [
        "step",
        "epoch",
        "loss",
        "eval_loss",
        "learning_rate",
        "grad_norm",
        "rewards/chosen",
        "rewards/rejected",
        "rewards/accuracies",
        "rewards/margins",
        "logps/chosen",
        "logps/rejected",
    ]

    rows = []
    for item in log_history:
        if "step" not in item:
            continue
        row = {}
        for key in keys:
            if key == "step":
                row[key] = int(item.get(key, 0))
            else:
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

    def series(metric: str) -> tuple[list[int], list[float]]:
        x, y = [], []
        for r in rows:
            if r.get(metric) is None:
                continue
            x.append(int(r["step"]))
            y.append(float(r[metric]))
        return x, y

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    x, y = series("loss")
    axes[0, 0].plot(x, y, marker="o", alpha=0.45, label="loss(raw)")
    if y:
        axes[0, 0].plot(x, _ema(y, ema_alpha), linewidth=2, label=f"loss(ema={ema_alpha})")
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    x1, y1 = series("rewards/margins")
    x2, y2 = series("rewards/accuracies")
    axes[0, 1].plot(x1, y1, marker="o", alpha=0.7, label="reward_margin")
    axes[0, 1].plot(x2, y2, marker="o", alpha=0.7, label="reward_accuracy")
    axes[0, 1].set_title("Preference Rewards")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    x1, y1 = series("rewards/chosen")
    x2, y2 = series("rewards/rejected")
    axes[1, 0].plot(x1, y1, marker="o", alpha=0.7, label="chosen_reward")
    axes[1, 0].plot(x2, y2, marker="o", alpha=0.7, label="rejected_reward")
    axes[1, 0].set_title("Chosen vs Rejected")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    x, y = series("learning_rate")
    axes[1, 1].plot(x, y, marker="o", color="#2ca02c", label="lr")
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
        "final_reward_margin": next(
            (r["rewards/margins"] for r in reversed(rows) if r["rewards/margins"] is not None), None
        ),
        "final_reward_accuracy": next(
            (r["rewards/accuracies"] for r in reversed(rows) if r["rewards/accuracies"] is not None), None
        ),
        "best_reward_margin": max((r["rewards/margins"] for r in rows if r["rewards/margins"] is not None), default=None),
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics_dir


def run_core_dpo_flow(args: argparse.Namespace, layout: dict[str, Path], factory_dir: Path) -> Path:
    """DPO 主流程（新手重点）：环境检测 -> 配置生成 -> 训练 -> 导出结果。"""
    config_path = layout["output"] / "train_dpo_auto.json"

    print("[步骤 1/5] 检测运行环境与精度", flush=True)
    runtime = detect_device_and_precision()
    print(f"Runtime: device={runtime['device']}, bf16={runtime['bf16']}, fp16={runtime['fp16']}", flush=True)

    print("[步骤 2/5] 生成训练配置", flush=True)
    config = build_train_config(args=args, runtime=runtime, output_path=layout["checkpoints"])
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Config written: {config_path}", flush=True)

    print("[步骤 3/5] 启动 DPO 训练", flush=True)
    run_train(factory_dir=factory_dir, config_path=config_path)

    print("[步骤 4/5] 整理模型产物", flush=True)
    move_model_artifacts(checkpoints_dir=layout["checkpoints"], models_dir=layout["models"])

    print("[步骤 5/5] 导出学习指标与可视化", flush=True)
    return export_learning_artifacts(
        checkpoints_dir=layout["checkpoints"],
        output_dir=layout["output"],
        ema_alpha=args.ema_alpha,
    )


def main() -> None:
    """主流程入口：生成配置、执行训练、导出可视化结果。"""
    args = build_default_args()
    code_dir = Path(__file__).resolve().parent
    module_dir = code_dir.parent
    layout = ensure_layout_dirs(module_dir=module_dir, output_arg=args.output_dir)
    factory_dir = resolve_factory_dir(code_dir)

    metrics_dir = run_core_dpo_flow(args=args, layout=layout, factory_dir=factory_dir)
    print(f"DPO done. Visualization exported to: {metrics_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
