#!/usr/bin/env python3
"""
SFT 单文件学习脚本（极简版）。

结构约定：
1) `main`：主训练流程（准备环境 -> 训练 -> 整理模型）。
2) `export_sft_visualization`：唯一可视化函数（导出 JSON/CSV/曲线图/summary）。

新人阅读顺序（建议）：
1) 先看 `main` 里的训练参数段：理解“这次训练跑什么参数”。
2) 再看 `main`：理解“训练流程是如何串起来的”。
3) 最后看 `export_sft_visualization`：理解“如何把日志变成可视化”。

学习步骤（与终端输出 1~5 对应）：
1) 准备目录与运行环境（code/data/models/output/checkpoints + 设备精度）。
2) 生成训练配置（把教学参数写入 JSON，便于复现）。
3) 启动训练（优先 CLI，失败回退模块入口）。
4) 整理模型产物（把最终模型文件移动到 models）。
5) 导出可视化（loss/lr/grad 等曲线 + summary）。
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


def export_sft_visualization(checkpoints_dir: Path, output_dir: Path) -> Path:
    """导出学习可视化：log_history、CSV、4宫格曲线图、summary。"""
    print("5) 导出可视化结果", flush=True)

    # 查找 trainer_state.json：先找 checkpoints 根目录，再找最新 checkpoint-*。
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

    # trainer_state.json 是 HuggingFace Trainer 的核心状态文件：
    # 其中 log_history 记录了训练过程中每次日志打印的指标。
    state = json.loads(state_path.read_text(encoding="utf-8"))
    log_history = state.get("log_history", [])
    if not log_history:
        raise RuntimeError(f"log_history 为空：{state_path}")

    metrics_dir = output_dir / "sft_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "log_history.json").write_text(
        json.dumps(log_history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 这些字段是新手最值得先看的最小指标集：
    # step(训练步数), loss(训练损失), eval_loss(验证损失), learning_rate(学习率), grad_norm(梯度范数)。
    keys = ["step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm"]
    rows: list[dict[str, float | int | None]] = []
    for item in log_history:
        # 只保留含 step 的项：
        # 某些日志（如纯评估汇总）可能没有 step，直接跳过可以避免画图错位。
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
        # 取出指定指标的一维序列，作为画图输入。
        x, y = [], []
        for r in rows:
            v = r.get(metric)
            if v is None:
                continue
            x.append(int(r["step"]))
            y.append(float(v))
        return x, y

    def ema(values: list[float], alpha: float) -> list[float]:
        # EMA 用于平滑噪声，方便新人看到“整体趋势”。
        # alpha 越大越跟随最新值，越小越平滑。
        out, prev = [], None
        for v in values:
            prev = v if prev is None else alpha * v + (1 - alpha) * prev
            out.append(prev)
        return out

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    x, y = series("loss")
    # Loss 曲线：
    # 新手重点看“是否总体下降”，不必追求每个点都下降。
    axes[0, 0].plot(x, y, marker="o", alpha=0.45, label="loss(raw)")
    if y:
        axes[0, 0].plot(x, ema(y, 0.2), linewidth=2, label="loss(ema=0.2)")
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    x, y = series("eval_loss")
    # Eval loss 曲线：
    # 用来观察泛化趋势；若持续明显高于 train loss，通常说明过拟合风险在上升。
    axes[0, 1].plot(x, y, marker="o", alpha=0.7, color="#d62728", label="eval_loss")
    axes[0, 1].set_title("Eval Loss")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    x, y = series("learning_rate")
    # 学习率曲线：
    # 主要用于确认调度器是否按预期工作（本例默认 cosine）。
    axes[1, 0].plot(x, y, marker="o", color="#2ca02c", label="lr")
    axes[1, 0].set_title("Learning Rate")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    x, y = series("grad_norm")
    # 梯度范数曲线：
    # 主要看是否出现长期异常尖峰（可能意味着训练不稳定）。
    axes[1, 1].plot(x, y, marker="o", color="#9467bd", label="grad_norm")
    axes[1, 1].set_title("Gradient Norm")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(metrics_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    # summary 给出“最关键结论”，便于技术复盘时快速口述训练结果。
    summary = {
        "total_steps": len(rows),
        "final_step": rows[-1]["step"] if rows else None,
        "final_loss": next((r["loss"] for r in reversed(rows) if r.get("loss") is not None), None),
        "best_eval_loss": min((r["eval_loss"] for r in rows if r.get("eval_loss") is not None), default=None),
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics_dir


def main() -> None:
    """主训练流程：准备目录与环境 -> 训练 -> 整理模型 -> 导出可视化。"""
    print("=== SFT 主流程（学习版）===", flush=True)
    # 新手提示：每一步都会在终端打印编号（1~5），可直接对照本函数中的步骤注释阅读。

    # 步骤 1：准备目录结构与运行时设备配置。
    print("1) 准备目录与运行环境", flush=True)

    code_dir = Path(__file__).resolve().parent
    module_dir = code_dir.parent
    output_dir = (module_dir / "output").resolve()
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

    # LLaMA-Factory 是实际执行训练的底层框架目录。
    # 这里优先在当前模块内查找；找不到就报错，避免“静默失败”。
    factory_dir = None
    for c in [code_dir / "LLaMA-Factory", module_dir / "LLaMA-Factory"]:
        if c.exists():
            factory_dir = c
            break
    if factory_dir is None:
        raise FileNotFoundError("未找到 LLaMA-Factory，请检查 sft 目录结构。")

    # 步骤 2：把训练参数固化为 JSON，便于回看和复现。
    print("2) 生成训练配置", flush=True)
    train_config = {
        "stage": "sft",  # 训练阶段：监督微调。
        "do_train": True,  # 是否执行训练。
        "model_name_or_path": "Qwen/Qwen3-0.6B",  # 基座模型。
        "dataset": "identity,alpaca_en_demo",  # 训练集。
        "template": "qwen",  # 模板。
        "finetuning_type": "lora",  # 微调方式：LoRA。
        "lora_target": "all",  # LoRA 注入目标层。
        "output_dir": str(checkpoints_dir),  # checkpoint 输出目录。
        "overwrite_output_dir": True,  # 允许覆盖旧输出。
        "per_device_train_batch_size": 1,  # 单卡 batch。
        "gradient_accumulation_steps": 1,  # 梯度累积。
        "lr_scheduler_type": "cosine",  # 学习率调度器。
        "logging_steps": 5,  # 日志间隔。
        "save_steps": 1000,  # 保存间隔。
        "learning_rate": 5e-5,  # 学习率。
        "num_train_epochs": 0.01,  # 训练轮数。
        "max_samples": 8,  # 最大样本数。
        "max_grad_norm": 1.0,  # 梯度裁剪阈值。
        "report_to": "none",  # 关闭 wandb 等外部上报。
        "bf16": runtime["bf16"],  # 是否启用 bf16。
        "fp16": runtime["fp16"],  # 是否启用 fp16。
    }
    config_path = output_dir / "train_sft_auto.json"
    config_path.write_text(json.dumps(train_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Config written: {config_path}", flush=True)
    # 这个 JSON 是“可复现实验单据”：新人调参时可先改 train_config 默认值，再检查这里是否生效。

    # 步骤 3：执行训练命令（CLI -> 模块入口双回退）。
    print("3) 启动 SFT 训练", flush=True)
    env = os.environ.copy()
    env["FORCE_TORCHRUN"] = "1"
    env["PYTHONPATH"] = str(factory_dir / "src") + os.pathsep + env.get("PYTHONPATH", "")

    # 训练入口有两种：
    # 1) llamafactory-cli train ...
    # 2) python -m llamafactory.cli train ...
    # 这样可以减少“环境里没有安装 CLI”导致的阻塞。
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
        raise RuntimeError("SFT 训练未能启动，请检查 finetune 环境依赖。") from last_error

    # 步骤 4：把最终模型文件从 checkpoints 归档到 models。
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
        # 如果 models 下已有同名文件，先删除再移动，保证“当前结果”为最新训练产物。
        if dst.exists():
            if dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        shutil.move(str(src), str(dst))

    # 步骤 5：导出学习可视化产物，便于复盘训练过程。
    metrics_dir = export_sft_visualization(checkpoints_dir=checkpoints_dir, output_dir=output_dir)
    print(f"SFT done. Visualization exported to: {metrics_dir}", flush=True)
    # 新手建议的结果阅读顺序：
    # 1) 先看 summary.json（总览）；
    # 2) 再看 training_curves.png（趋势）；
    # 3) 最后看 training_metrics.csv（逐步细节）。


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
