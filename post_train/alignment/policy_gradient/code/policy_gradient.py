#!/usr/bin/env python3
"""
Policy Gradient 单文件学习脚本（主流程 + 可视化，基于 PPO 近似实现）。

学习步骤（与终端输出 1~5 对应）：
1) 准备目录与环境。
2) 生成训练参数。
3) 启动训练（CLI/模块双入口）。
4) 整理模型产物。
5) 导出可视化（失败时生成占位结果）。
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


PG_CONFIG = {
    "model_id": "Qwen/Qwen3-0.6B",  # 策略模型。
    "reward_model": "Qwen/Qwen3-0.6B",  # 奖励模型。
    "reward_model_type": "full",  # 奖励模型类型。
    "template": "qwen",  # 模板。
    "dataset": "alpaca_en_demo",  # 数据集。
    "output_dir": "output",  # 输出目录。
    "max_samples": 8,  # 最大样本数。
    "num_train_epochs": 0.01,  # 训练轮数。
    "learning_rate": 1e-6,  # 学习率。
    "batch_size": 1,  # 单卡 batch。
    "grad_accum": 1,  # 梯度累积。
    "logging_steps": 5,  # 日志间隔。
    "save_steps": 50,  # checkpoint 间隔。
    "ppo_buffer_size": 1,  # PPO buffer 大小。
    "ppo_epochs": 4,  # PPO 更新次数。
    "ppo_target": 6.0,  # KL 目标。
    "ppo_score_norm": False,  # 是否归一化奖励。
    "ppo_whiten_rewards": False,  # 是否 whiten 奖励。
    "ema_alpha": 0.2,  # 曲线平滑系数。
}


def export_policy_gradient_visualization(checkpoints_dir: Path, output_dir: Path) -> Path:
    """导出 Policy Gradient 可视化，训练日志缺失时导出占位结果。"""
    print("5) 导出可视化结果", flush=True)

    metrics_dir = output_dir / "policy_gradient_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
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

    state_path = checkpoints_dir / "trainer_state.json"
    if not state_path.exists():
        for d in sorted(checkpoints_dir.glob("checkpoint-*"), reverse=True):
            cand = d / "trainer_state.json"
            if cand.exists():
                state_path = cand
                break

    def write_placeholder(note: str) -> None:
        (metrics_dir / "log_history.json").write_text("[]\n", encoding="utf-8")
        with (metrics_dir / "training_metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.axis("off")
            ax.text(0.02, 0.6, "Policy Gradient Placeholder", fontsize=16, weight="bold")
            ax.text(0.02, 0.35, note, fontsize=11)
            fig.tight_layout()
            fig.savefig(metrics_dir / "training_curves.png", dpi=160)
            plt.close(fig)
        except Exception:
            pass
        (metrics_dir / "summary.json").write_text(
            json.dumps(
                {
                    "total_steps": 0,
                    "final_step": None,
                    "final_loss": None,
                    "final_reward": None,
                    "final_kl": None,
                    "best_reward": None,
                    "status": "placeholder",
                    "note": note,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    if not state_path.exists():
        write_placeholder(f"未找到 trainer_state.json：{checkpoints_dir}")
        return metrics_dir

    state = json.loads(state_path.read_text(encoding="utf-8"))
    log_history = state.get("log_history", [])
    if not log_history:
        write_placeholder(f"log_history 为空：{state_path}")
        return metrics_dir

    (metrics_dir / "log_history.json").write_text(json.dumps(log_history, ensure_ascii=False, indent=2), encoding="utf-8")

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

    def series(candidates: list[str]) -> tuple[list[int], list[float]]:
        for metric in candidates:
            x, y = [], []
            for r in rows:
                v = r.get(metric)
                if v is None:
                    continue
                x.append(int(r["step"]))
                y.append(float(v))
            if y:
                return x, y
        return [], []

    def ema(values: list[float], alpha: float) -> list[float]:
        out, prev = [], None
        for v in values:
            prev = v if prev is None else alpha * v + (1 - alpha) * prev
            out.append(prev)
        return out

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    x, y = series(["loss", "ppo/loss/total"])
    axes[0, 0].plot(x, y, marker="o", alpha=0.45, label="loss(raw)")
    if y:
        axes[0, 0].plot(x, ema(y, PG_CONFIG["ema_alpha"]), linewidth=2, label=f"loss(ema={PG_CONFIG['ema_alpha']})")
    axes[0, 0].set_title("Policy Gradient Loss")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    x, y = series(["reward"])
    axes[0, 1].plot(x, y, marker="o", alpha=0.45, label="reward(raw)")
    if y:
        axes[0, 1].plot(x, ema(y, PG_CONFIG["ema_alpha"]), linewidth=2, label=f"reward(ema={PG_CONFIG['ema_alpha']})")
    axes[0, 1].set_title("Reward")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    x, y = series(["kl", "objective/kl"])
    axes[1, 0].plot(x, y, marker="o", alpha=0.7, color="#ff7f0e", label="kl")
    axes[1, 0].set_title("KL Divergence")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    x, y = series(["learning_rate", "ppo/learning_rate"])
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
        "final_loss": next((r["loss"] for r in reversed(rows) if r.get("loss") is not None), None),
        "final_reward": next((r["reward"] for r in reversed(rows) if r.get("reward") is not None), None),
        "final_kl": next((r["kl"] for r in reversed(rows) if r.get("kl") is not None), None),
        "best_reward": max((r["reward"] for r in rows if r.get("reward") is not None), default=None),
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics_dir


def main() -> None:
    """主训练流程：准备目录 -> 生成配置 -> 训练 -> 整理模型 -> 导出可视化。"""
    print("=== Policy Gradient 主流程（学习版）===", flush=True)

    # 步骤 1：准备目录与设备精度。
    print("1) 准备目录与运行环境", flush=True)

    code_dir = Path(__file__).resolve().parent
    module_dir = code_dir.parent
    output_dir = (module_dir / PG_CONFIG["output_dir"]).resolve()
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
        raise FileNotFoundError("未找到 LLaMA-Factory，请检查 policy_gradient/sft 目录结构。")

    # 步骤 2：写出训练配置。
    print("2) 生成训练配置", flush=True)
    train_config = {
        "stage": "ppo",  # 训练阶段：PPO（用于近似 PG）。
        "do_train": True,  # 执行训练。
        "model_name_or_path": PG_CONFIG["model_id"],  # 策略模型。
        "reward_model": PG_CONFIG["reward_model"],  # 奖励模型。
        "reward_model_type": PG_CONFIG["reward_model_type"],  # 奖励模型类型。
        "dataset": PG_CONFIG["dataset"],  # 数据集。
        "template": PG_CONFIG["template"],  # 模板。
        "finetuning_type": "lora",  # LoRA 微调。
        "lora_target": "all",  # LoRA 目标层。
        "output_dir": str(checkpoints_dir),  # checkpoint 目录。
        "overwrite_output_dir": True,  # 覆盖旧目录。
        "save_only_model": True,  # 仅保存模型权重。
        "per_device_train_batch_size": PG_CONFIG["batch_size"],  # 单卡 batch。
        "gradient_accumulation_steps": PG_CONFIG["grad_accum"],  # 梯度累积。
        "lr_scheduler_type": "cosine",  # 学习率调度器。
        "logging_steps": PG_CONFIG["logging_steps"],  # 日志间隔。
        "save_steps": PG_CONFIG["save_steps"],  # 保存间隔。
        "learning_rate": PG_CONFIG["learning_rate"],  # 学习率。
        "num_train_epochs": PG_CONFIG["num_train_epochs"],  # 训练轮数。
        "max_samples": PG_CONFIG["max_samples"],  # 最大样本数。
        "max_grad_norm": 1.0,  # 梯度裁剪。
        "report_to": "none",  # 关闭外部上报。
        "ppo_buffer_size": PG_CONFIG["ppo_buffer_size"],  # PPO buffer。
        "ppo_epochs": PG_CONFIG["ppo_epochs"],  # PPO 更新次数。
        "ppo_target": PG_CONFIG["ppo_target"],  # KL 目标。
        "ppo_score_norm": PG_CONFIG["ppo_score_norm"],  # 奖励归一化开关。
        "ppo_whiten_rewards": PG_CONFIG["ppo_whiten_rewards"],  # whiten 奖励开关。
        "bf16": runtime["bf16"],  # bf16 开关。
        "fp16": runtime["fp16"],  # fp16 开关。
    }
    config_path = output_dir / "train_policy_gradient_auto.json"
    config_path.write_text(json.dumps(train_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Config written: {config_path}", flush=True)

    # 步骤 3：执行训练，失败时回退第二入口。
    print("3) 启动 Policy Gradient 训练", flush=True)
    env = os.environ.copy()
    env["FORCE_TORCHRUN"] = "1"
    env.setdefault("DISABLE_VERSION_CHECK", "1")
    env["PYTHONPATH"] = str(factory_dir / "src") + os.pathsep + env.get("PYTHONPATH", "")

    train_ok = False
    commands: list[list[str]] = []
    if shutil.which("llamafactory-cli"):
        commands.append(["llamafactory-cli", "train", str(config_path)])
    commands.append([sys.executable, "-m", "llamafactory.cli", "train", str(config_path)])
    for cmd in commands:
        try:
            subprocess.run(cmd, cwd=str(factory_dir), check=True, env=env)
            train_ok = True
            break
        except Exception as exc:  # noqa: PERF203
            print(f"[WARN] 训练入口失败：{' '.join(cmd)} -> {exc}", flush=True)

    # 步骤 4：训练成功时整理模型产物。
    print("4) 整理模型产物", flush=True)
    if train_ok:
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
            "value_head.safetensors",
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

    # 步骤 5：导出可视化（包括失败占位场景）。
    metrics_dir = export_policy_gradient_visualization(checkpoints_dir=checkpoints_dir, output_dir=output_dir)
    print(f"Policy Gradient done. Visualization exported to: {metrics_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
