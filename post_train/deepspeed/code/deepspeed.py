#!/usr/bin/env python3
"""
DeepSpeed 最小可运行示例：Toy Regression + 可视化。

一、DeepSpeed 原理（面向实现）
1) DeepSpeed 是大模型训练加速框架，提供 ZeRO、混合精度、梯度累积等能力。
2) ZeRO 会分片优化器状态/梯度/参数，降低显存占用。
3) 训练流程本质仍是前向、反向、参数更新，只是由 DeepSpeed Engine 管理执行细节。

二、代码框架（从入口到结果）
1) `parse_args`：读取训练和 DeepSpeed 参数。
2) `build_deepspeed_config`：生成最小可用 DeepSpeed 配置。
3) `run_train_loop`：优先尝试 DeepSpeed，失败自动回退 Torch。
4) `export_learning_artifacts`：导出日志、CSV、曲线图、summary。
5) `main`：串联完整流程并落盘输出。

用法：
  python code/deepspeed.py
  python code/deepspeed.py --use-deepspeed
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_OUTPUT_DIR = "output"


def parse_args() -> argparse.Namespace:
    """解析命令行参数，返回训练与 DeepSpeed 配置。"""
    parser = argparse.ArgumentParser(description="Run DeepSpeed demo training and export visualization artifacts.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)

    # 训练参数。
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)

    # DeepSpeed 相关参数。
    parser.add_argument("--use-deepspeed", action="store_true", help="尝试使用 DeepSpeed 引擎训练。")
    parser.add_argument("--zero-stage", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """设置随机种子，增强复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> torch.device:
    """选择设备：CUDA > MPS > CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_output_dir(base_dir: Path, output_dir: str) -> Path:
    """解析输出目录：相对路径按 deepspeed 目录解析。"""
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
    for p in layout.values():
        p.mkdir(parents=True, exist_ok=True)
    return layout


class ToyRegressor(nn.Module):
    """简单 MLP 回归器：拟合 y = sin(3x)+噪声。"""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def sample_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """采样一批 toy 回归数据。"""
    x = torch.empty(batch_size, 1, device=device).uniform_(-math.pi, math.pi)
    y = torch.sin(3.0 * x) + 0.1 * torch.randn_like(x)
    return x, y


def build_deepspeed_config(args: argparse.Namespace) -> dict[str, Any]:
    """构建最小可用 DeepSpeed 配置。"""
    cfg: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "zero_optimization": {"stage": args.zero_stage},
        "optimizer": {
            "type": "Adam",
            "params": {"lr": args.learning_rate, "betas": [0.9, 0.999], "eps": 1e-8},
        },
    }
    if args.fp16:
        cfg["fp16"] = {"enabled": True}
    if args.bf16:
        cfg["bf16"] = {"enabled": True}
    return cfg


def try_init_deepspeed(
    model: nn.Module,
    ds_config: dict[str, Any],
) -> tuple[Any | None, str]:
    """尝试初始化 DeepSpeed，引擎不可用时返回 None 并附说明。"""
    try:
        import deepspeed  # type: ignore
    except Exception as exc:
        return None, f"deepspeed import failed: {exc}"

    try:
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
            dist_init_required=False,
        )
        return engine, "deepspeed initialized"
    except Exception as exc:
        return None, f"deepspeed initialize failed: {exc}"


def save_torch_checkpoint(
    checkpoints_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    log_item: dict[str, float],
) -> None:
    """保存 Torch 路径 checkpoint。"""
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "log": log_item,
        },
        checkpoints_dir / f"checkpoint-{step}.pt",
    )


def run_train_loop(
    model: ToyRegressor,
    args: argparse.Namespace,
    device: torch.device,
    ds_config: dict[str, Any],
    checkpoints_dir: Path,
) -> tuple[list[dict[str, float]], str]:
    """执行训练循环，优先使用 DeepSpeed，失败则回退 Torch。"""
    logs: list[dict[str, float]] = []
    backend = "torch"

    # 尝试 DeepSpeed。
    ds_engine = None
    if args.use_deepspeed:
        ds_engine, reason = try_init_deepspeed(model, ds_config)
        if ds_engine is not None:
            backend = "deepspeed"
            print(f"[INFO] {reason}")
        else:
            print(f"[WARN] {reason}. Fallback to Torch training.")

    if backend == "deepspeed":
        # DeepSpeed 路径。
        for step in range(1, args.steps + 1):
            x, y = sample_batch(args.batch_size, ds_engine.device)
            pred = ds_engine(x)
            loss = F.mse_loss(pred, y)
            ds_engine.backward(loss)
            ds_engine.step()

            # 简单验证 loss（同分布采样）。
            with torch.no_grad():
                xv, yv = sample_batch(args.batch_size, ds_engine.device)
                val_loss = float(F.mse_loss(ds_engine(xv), yv).item())

            log_item = {
                "step": float(step),
                "train_loss": float(loss.item()),
                "val_loss": val_loss,
                "learning_rate": float(ds_engine.get_lr()[0]) if hasattr(ds_engine, "get_lr") else args.learning_rate,
            }
            logs.append(log_item)

            if step % args.log_every == 0 or step == 1 or step == args.steps:
                print(
                    f"[Step {step:04d}] train_loss={log_item['train_loss']:.6f} "
                    f"val_loss={log_item['val_loss']:.6f}"
                )
            if step % args.save_every == 0 or step == args.steps:
                save_path = checkpoints_dir / f"checkpoint-{step}"
                ds_engine.save_checkpoint(str(save_path))
        return logs, backend

    # Torch 回退路径。
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    grad_accum = max(1, args.grad_accum_steps)
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, args.steps + 1):
        x, y = sample_batch(args.batch_size, device)
        pred = model(x)
        loss = F.mse_loss(pred, y) / grad_accum
        loss.backward()

        if step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            xv, yv = sample_batch(args.batch_size, device)
            val_loss = float(F.mse_loss(model(xv), yv).item())

        log_item = {
            "step": float(step),
            "train_loss": float(loss.item() * grad_accum),
            "val_loss": val_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        logs.append(log_item)

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            print(
                f"[Step {step:04d}] train_loss={log_item['train_loss']:.6f} "
                f"val_loss={log_item['val_loss']:.6f}"
            )
        if step % args.save_every == 0 or step == args.steps:
            save_torch_checkpoint(checkpoints_dir, step, model, optimizer, log_item)

    return logs, backend


def export_learning_artifacts(
    logs: list[dict[str, float]],
    output_dir: Path,
    backend: str,
    models_dir: Path,
    model: nn.Module,
    args: argparse.Namespace,
) -> Path:
    """导出日志、曲线图和 summary。"""
    metrics_dir = output_dir / "deepspeed_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / "training_log.json").write_text(
        json.dumps(logs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (metrics_dir / "training_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss", "learning_rate"])
        writer.writeheader()
        writer.writerows(logs)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"`matplotlib` is required for visualization: {exc}") from exc

    steps = [x["step"] for x in logs]
    train_loss = [x["train_loss"] for x in logs]
    val_loss = [x["val_loss"] for x in logs]
    lr = [x["learning_rate"] for x in logs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(steps, train_loss, label="train_loss")
    axes[0].plot(steps, val_loss, label="val_loss")
    axes[0].set_title(f"DeepSpeed Demo Loss ({backend})")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("mse")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, lr, color="#2ca02c")
    axes[1].set_title("Learning Rate")
    axes[1].set_xlabel("step")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(metrics_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    summary = {
        "backend": backend,
        "steps": len(logs),
        "final_train_loss": train_loss[-1] if train_loss else None,
        "final_val_loss": val_loss[-1] if val_loss else None,
        "best_val_loss": min(val_loss) if val_loss else None,
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 保存最终模型（Torch state dict）。
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, models_dir / "deepspeed_demo_model.pt")
    return metrics_dir


def main() -> None:
    """主流程入口：生成配置、训练并导出可视化。"""
    args = parse_args()
    set_seed(args.seed)
    device = detect_device()
    print(f"Runtime: device={device.type}")

    code_dir = Path(__file__).resolve().parent
    module_dir = code_dir.parent
    layout = ensure_layout_dirs(module_dir=module_dir, output_arg=args.output_dir)

    ds_config = build_deepspeed_config(args)
    (layout["output"] / "deepspeed_run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (layout["output"] / "deepspeed_config_auto.json").write_text(
        json.dumps(ds_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model = ToyRegressor(hidden_dim=args.hidden_dim)
    logs, backend = run_train_loop(
        model=model,
        args=args,
        device=device,
        ds_config=ds_config,
        checkpoints_dir=layout["checkpoints"],
    )
    metrics_dir = export_learning_artifacts(
        logs=logs,
        output_dir=layout["output"],
        backend=backend,
        models_dir=layout["models"],
        model=model,
        args=args,
    )
    print(f"DeepSpeed demo done. Visualization exported to: {metrics_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
