#!/usr/bin/env python3
"""
DPO 单文件学习脚本（主流程 + 可视化）。

结构约定：
1) `main`：主训练流程。
2) `export_dpo_visualization`：唯一可视化函数。
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch


DPO_CONFIG = {
    "model_id": "Qwen/Qwen3-0.6B",
    "template": "qwen",
    "dataset": "dpo_zh_demo",
    "output_dir": "output",
    "max_samples": 8,
    "num_train_epochs": 0.01,
    "learning_rate": 5e-6,
    "batch_size": 1,
    "grad_accum": 1,
    "logging_steps": 5,
    "save_steps": 500,
    "pref_beta": 0.1,
    "pref_loss": "sigmoid",
    "ema_alpha": 0.2,
}


def export_dpo_visualization(checkpoints_dir: Path, output_dir: Path) -> Path:
    """导出 DPO 可视化：JSON、CSV、曲线图、summary。"""
    print("5) 导出可视化结果", flush=True)

    state_path = checkpoints_dir / "trainer_state.json"
    if not state_path.exists():
        candidates: list[tuple[int, Path]] = []
        for d in checkpoints_dir.glob("checkpoint-*"):
            if not d.is_dir():
                continue
            m = re.match(r"checkpoint-(\d+)$", d.name)
            if m:
                candidates.append((int(m.group(1)), d / "trainer_state.json"))
        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, p in candidates:
            if p.exists():
                state_path = p
                break

    if not state_path.exists():
        raise FileNotFoundError(f"未找到 trainer_state.json：{checkpoints_dir}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    log_history = state.get("log_history", [])
    if not log_history:
        raise RuntimeError(f"log_history 为空：{state_path}")

    metrics_dir = output_dir / "dpo_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "log_history.json").write_text(json.dumps(log_history, ensure_ascii=False, indent=2), encoding="utf-8")

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

    rows: list[dict[str, float | int | None]] = []
    for item in log_history:
        if "step" not in item:
            continue
        row: dict[str, float | int | None] = {"step": int(item.get("step", 0))}
        for k in keys[1:]:
            v = item.get(k)
            if isinstance(v, (int, float)):
                row[k] = float(v)
            elif isinstance(v, str):
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = None
            else:
                row[k] = None
        rows.append(row)

    with (metrics_dir / "training_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: PERF203
        raise RuntimeError(f"缺少 matplotlib，无法导出曲线：{exc}") from exc

    def series(metric: str) -> tuple[list[int], list[float]]:
        x, y = [], []
        for r in rows:
            v = r.get(metric)
            if v is None:
                continue
            x.append(int(r["step"]))
            y.append(float(v))
        return x, y

    def ema(values: list[float], alpha: float) -> list[float]:
        out, prev = [], None
        for v in values:
            prev = v if prev is None else alpha * v + (1 - alpha) * prev
            out.append(prev)
        return out

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    x, y = series("loss")
    axes[0, 0].plot(x, y, marker="o", alpha=0.45, label="loss(raw)")
    if y:
        axes[0, 0].plot(x, ema(y, DPO_CONFIG["ema_alpha"]), linewidth=2, label=f"loss(ema={DPO_CONFIG['ema_alpha']})")
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
        "final_loss": next((r["loss"] for r in reversed(rows) if r.get("loss") is not None), None),
        "final_reward_margin": next((r["rewards/margins"] for r in reversed(rows) if r.get("rewards/margins") is not None), None),
        "final_reward_accuracy": next((r["rewards/accuracies"] for r in reversed(rows) if r.get("rewards/accuracies") is not None), None),
        "best_reward_margin": max((r["rewards/margins"] for r in rows if r.get("rewards/margins") is not None), default=None),
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics_dir


def main() -> None:
    """主训练流程：准备目录 -> 生成配置 -> 训练 -> 整理模型 -> 导出可视化。"""
    print("=== DPO 主流程（学习版）===", flush=True)
    print("1) 准备目录与运行环境", flush=True)

    code_dir = Path(__file__).resolve().parent
    module_dir = code_dir.parent
    output_dir = (module_dir / DPO_CONFIG["output_dir"]).resolve()
    checkpoints_dir = module_dir / "checkpoints"
    models_dir = module_dir / "models"
    for p in [module_dir / "code", module_dir / "data", models_dir, output_dir, checkpoints_dir]:
        p.mkdir(parents=True, exist_ok=True)

    runtime = {"device": "cpu", "bf16": False, "fp16": False}
    if torch.cuda.is_available():
        runtime = {"device": "cuda", "bf16": True, "fp16": False}
    elif torch.backends.mps.is_available():
        runtime = {"device": "mps", "bf16": False, "fp16": True}
    print(f"Runtime: device={runtime['device']}, bf16={runtime['bf16']}, fp16={runtime['fp16']}", flush=True)

    factory_dir = None
    for c in [
        code_dir / "LLaMA-Factory",
        module_dir / "LLaMA-Factory",
        module_dir.parent / "sft" / "LLaMA-Factory",
        module_dir.parent / "sft" / "code" / "LLaMA-Factory",
    ]:
        if c.exists():
            factory_dir = c
            break
    if factory_dir is None:
        raise FileNotFoundError("未找到 LLaMA-Factory，请检查 dpo/sft 目录结构。")

    print("2) 生成训练配置", flush=True)
    train_config = {
        "stage": "dpo",
        "do_train": True,
        "model_name_or_path": DPO_CONFIG["model_id"],
        "dataset": DPO_CONFIG["dataset"],
        "template": DPO_CONFIG["template"],
        "finetuning_type": "lora",
        "lora_target": "all",
        "pref_beta": DPO_CONFIG["pref_beta"],
        "pref_loss": DPO_CONFIG["pref_loss"],
        "output_dir": str(checkpoints_dir),
        "overwrite_output_dir": True,
        "per_device_train_batch_size": DPO_CONFIG["batch_size"],
        "gradient_accumulation_steps": DPO_CONFIG["grad_accum"],
        "lr_scheduler_type": "cosine",
        "logging_steps": DPO_CONFIG["logging_steps"],
        "save_steps": DPO_CONFIG["save_steps"],
        "learning_rate": DPO_CONFIG["learning_rate"],
        "num_train_epochs": DPO_CONFIG["num_train_epochs"],
        "max_samples": DPO_CONFIG["max_samples"],
        "max_grad_norm": 1.0,
        "report_to": "none",
        "bf16": runtime["bf16"],
        "fp16": runtime["fp16"],
    }
    config_path = output_dir / "train_dpo_auto.json"
    config_path.write_text(json.dumps(train_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Config written: {config_path}", flush=True)

    print("3) 启动 DPO 训练", flush=True)
    env = os.environ.copy()
    env["FORCE_TORCHRUN"] = "1"
    env["PYTHONPATH"] = str(factory_dir / "src") + os.pathsep + env.get("PYTHONPATH", "")

    commands: list[list[str]] = []
    if shutil.which("llamafactory-cli"):
        commands.append(["llamafactory-cli", "train", str(config_path)])
    commands.append([sys.executable, "-m", "llamafactory.cli", "train", str(config_path)])

    last_error: Exception | None = None
    for cmd in commands:
        try:
            subprocess.run(cmd, cwd=str(factory_dir), check=True, env=env)
            last_error = None
            break
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            print(f"[WARN] 训练入口失败：{' '.join(cmd)}", flush=True)
    if last_error is not None:
        raise RuntimeError("DPO 训练未能启动，请检查 finetune 环境依赖。") from last_error

    print("4) 整理模型产物", flush=True)
    for name in [
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
    ]:
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

    metrics_dir = export_dpo_visualization(checkpoints_dir=checkpoints_dir, output_dir=output_dir)
    print(f"DPO done. Visualization exported to: {metrics_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
