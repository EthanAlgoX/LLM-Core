#!/usr/bin/env python3
"""
LLaMA-Factory 的单模式 SFT 流程：
训练 -> 导出指标 -> 绘制学习曲线。

用法：
  python sft.py
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


def parse_args() -> argparse.Namespace:
    """解析命令行参数，返回 SFT 训练与可视化所需配置。"""
    parser = argparse.ArgumentParser(description="Run SFT and always export visualization artifacts.")
    # 模型与数据：这三项决定“学什么、按什么模板学”。
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--template", default="qwen")
    parser.add_argument("--dataset", default="identity,alpaca_en_demo")
    parser.add_argument("--output-dir", default="qwen3_lora")

    # 训练核心超参：尽量保留少量最关键参数，便于学习和复现实验。
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=1000)
    # 可视化参数：用于平滑 loss 曲线，降低训练噪声，便于观察趋势。
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.2,
        help="EMA smoothing factor for plotting curves.",
    )
    return parser.parse_args()


def detect_device_and_precision() -> dict[str, Any]:
    """检测可用设备并选择推荐混合精度配置。"""
    # CUDA 优先：通常速度最快，并可用 bf16。
    if torch.cuda.is_available():
        return {"device": "cuda", "bf16": True, "fp16": False}
    # Apple Silicon 的 MPS 路径：经验上 fp16 更稳。
    if torch.backends.mps.is_available():
        # 在混合工具链中，MPS 使用 fp16 通常更稳定。
        return {"device": "mps", "bf16": False, "fp16": True}
    # CPU 兜底：不使用混合精度。
    return {"device": "cpu", "bf16": False, "fp16": False}


def build_train_config(args: argparse.Namespace, runtime: dict[str, Any], output_path: Path) -> dict[str, Any]:
    """根据参数与运行时环境构造最小可用的 LLaMA-Factory SFT 配置。"""
    # 保持配置最小化：只保留 SFT 学习/演示最关键参数。
    # 说明：
    # 1) stage 固定为 sft，突出单一学习目标；
    # 2) finetuning_type 选择 lora，降低显存占用；
    # 3) report_to 关闭外部上报，避免新手环境额外依赖。
    return {
        "stage": "sft",  # 训练阶段：监督微调（Supervised Fine-Tuning）。
        "do_train": True,  # 启用训练流程（而不是仅评估/预测）。
        "model_name_or_path": args.model_id,  # 基座模型名称或本地模型路径。
        "dataset": args.dataset,  # 训练数据集名称（可逗号分隔多个数据集）。
        "template": args.template,  # 对话模板名，控制提示词拼接与角色格式。
        "finetuning_type": "lora",  # 微调方式：LoRA（参数高效微调）。
        "lora_target": "all",  # LoRA 注入目标层；all 表示尽量覆盖可注入线性层。
        "output_dir": str(output_path),  # 训练输出目录（checkpoint、日志、状态文件）。
        "per_device_train_batch_size": args.batch_size,  # 单设备每步 batch size。
        "gradient_accumulation_steps": args.grad_accum,  # 梯度累积步数，用于等效增大总 batch。
        "lr_scheduler_type": "cosine",  # 学习率调度策略：余弦衰减。
        "logging_steps": args.logging_steps,  # 每隔多少步记录一次训练日志。
        "save_steps": args.save_steps,  # 每隔多少步保存一次 checkpoint。
        "learning_rate": args.learning_rate,  # 初始学习率。
        "num_train_epochs": args.num_train_epochs,  # 训练轮数（epoch）。
        "max_samples": args.max_samples,  # 训练样本上限（用于快速实验/教学演示）。
        "max_grad_norm": 1.0,  # 梯度裁剪阈值，抑制梯度爆炸。
        "report_to": "none",  # 不上报到外部实验平台（如 wandb）。
        "bf16": runtime["bf16"],  # 是否启用 bfloat16 混合精度。
        "fp16": runtime["fp16"],  # 是否启用 float16 混合精度。
    }


def run_train(factory_dir: Path, config_path: Path) -> None:
    """执行 SFT 训练，优先使用 CLI，失败时回退到模块入口。"""
    env = os.environ.copy()
    # 始终走 torchrun 流程，保持与 LLaMA-Factory 多卡/单卡入口一致。
    env["FORCE_TORCHRUN"] = "1"
    # 当环境未安装包时，确保可以直接从本地源码目录导入。
    src_dir = str(factory_dir / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

    # 先尝试官方 CLI；若失败则回退到模块入口（适配源码/可编辑安装场景）。
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
            # 记录失败但不立即退出：继续尝试下一个可用入口。
            print(f"[WARN] Training command failed: {' '.join(cmd)}")

    raise RuntimeError(
        "Failed to start SFT training via both CLI and module entrypoints. "
        "Please check LLaMA-Factory dependencies in your `finetune` environment."
    ) from last_error


def _to_float(v: Any) -> float | None:
    """将任意日志值转换为 float，无法转换时返回 None。"""
    # 训练日志里部分字段可能是字符串，这里统一转 float 便于 CSV/画图。
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
    # 指数滑动平均：y_t = alpha * x_t + (1-alpha) * y_{t-1}
    # 用于平滑 loss 抖动，观察长期收敛趋势。
    out, prev = [], None
    for x in values:
        prev = x if prev is None else alpha * x + (1.0 - alpha) * prev
        out.append(prev)
    return out


def find_trainer_state(output_dir: Path) -> Path | None:
    """查找 trainer_state.json，优先根目录，其次最新 checkpoint。"""
    # 优先读取 output_dir 根目录下的 trainer_state.json。
    direct = output_dir / "trainer_state.json"
    if direct.exists():
        return direct

    # 若根目录状态文件不存在，则使用最新 checkpoint 中的状态文件。
    # 某些训练中断/保存策略下，状态文件可能只存在于 checkpoint-* 目录。
    checkpoints = []
    for d in output_dir.glob("checkpoint-*"):
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


def export_learning_artifacts(output_dir: Path, ema_alpha: float) -> Path:
    """导出训练日志、CSV、曲线图与摘要，返回产物目录路径。"""
    # 从 trainer_state 读取 log_history，导出“原始日志 + 表格 + 曲线图 + 摘要”。
    state_path = find_trainer_state(output_dir)
    if state_path is None:
        raise FileNotFoundError(f"No trainer_state.json found under: {output_dir}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    log_history = state.get("log_history", [])
    if not log_history:
        raise RuntimeError(f"log_history is empty in: {state_path}")

    metrics_dir = output_dir / "sft_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / "log_history.json").write_text(
        json.dumps(log_history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    keys = ["step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm"]
    rows = []
    for item in log_history:
        if "step" not in item:
            # 跳过无 step 的事件记录（例如部分汇总日志）。
            continue
        # 将混合的数值/字符串日志规范化为可分析的数值表格。
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

    # 构建 x/y 序列，并自动跳过稀疏日志中的缺失值。
    def series(metric: str) -> tuple[list[int], list[float]]:
        # 不同 step 可能只记录部分字段（例如只有 train loss 没有 eval loss）。
        x, y = [], []
        for r in rows:
            if r.get(metric) is None:
                continue
            x.append(int(r["step"]))
            y.append(float(r[metric]))
        return x, y

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 左上：训练损失（原始 + EMA），用于判断是否总体下降。
    x, y = series("loss")
    axes[0, 0].plot(x, y, marker="o", alpha=0.45, label="loss(raw)")
    if y:
        # EMA 曲线可降低 step 级抖动，更容易观察整体趋势。
        axes[0, 0].plot(x, _ema(y, ema_alpha), linewidth=2, label=f"loss(ema={ema_alpha})")
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 右上：验证损失，用于观察泛化趋势（是否过拟合）。
    x, y = series("eval_loss")
    axes[0, 1].plot(x, y, marker="o", alpha=0.65, color="#d62728", label="eval_loss")
    axes[0, 1].set_title("Eval Loss")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 左下：学习率曲线，方便对齐调度器行为（如 cosine 衰减）。
    x, y = series("learning_rate")
    axes[1, 0].plot(x, y, marker="o", color="#2ca02c", label="lr")
    axes[1, 0].set_title("Learning Rate")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 右下：梯度范数，帮助排查梯度爆炸/不稳定训练。
    x, y = series("grad_norm")
    axes[1, 1].plot(x, y, marker="o", color="#9467bd", label="grad_norm")
    axes[1, 1].set_title("Gradient Norm")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(metrics_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    # 保存简要摘要，便于不打开完整 CSV/JSON 也能快速查看结果。
    # 说明：
    # - final_loss：最后一个可用训练 loss；
    # - best_eval_loss：全程最小验证 loss（若存在）；
    # - total_steps/final_step：反映训练长度。
    summary = {
        "total_steps": len(rows),
        "final_step": rows[-1]["step"] if rows else None,
        "final_loss": next((r["loss"] for r in reversed(rows) if r["loss"] is not None), None),
        "best_eval_loss": min((r["eval_loss"] for r in rows if r["eval_loss"] is not None), default=None),
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics_dir


def main() -> None:
    """主流程入口：生成配置、执行训练、导出可视化结果。"""
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    factory_dir = base_dir / "LLaMA-Factory"
    output_dir = factory_dir / args.output_dir
    config_path = factory_dir / "train_sft_auto.json"

    if not factory_dir.exists():
        raise FileNotFoundError(f"LLaMA-Factory directory not found: {factory_dir}")

    runtime = detect_device_and_precision()
    print(f"Runtime: device={runtime['device']}, bf16={runtime['bf16']}, fp16={runtime['fp16']}")

    config = build_train_config(args=args, runtime=runtime, output_path=output_dir)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Config written: {config_path}")

    # 单一流程：训练 -> 导出学习结果 -> 输出可视化目录。
    run_train(factory_dir=factory_dir, config_path=config_path)
    metrics_dir = export_learning_artifacts(output_dir=output_dir, ema_alpha=args.ema_alpha)
    print(f"SFT done. Visualization exported to: {metrics_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # 统一异常出口，方便脚本化调用时感知失败。
        print(f"[ERROR] {exc}")
        sys.exit(1)
