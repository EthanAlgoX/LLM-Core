#!/usr/bin/env python3
"""
混合精度训练最小可运行示例：AMP + GradScaler + 可视化。

一、混合精度原理（面向实现）
1) 混合精度用低精度（fp16/bf16）执行大部分算子，同时保留关键计算精度。
2) 目的：提升吞吐、降低显存占用。
3) fp16 常配合 GradScaler 防止梯度下溢；bf16 通常不需要 scaler。
4) 本示例支持 auto/no_amp/fp16/bf16，自动回退到稳定配置。

二、代码框架（从入口到结果）
1) `build_default_args`：读取训练与 AMP 参数。
2) `resolve_amp`：根据设备和参数确定最终 AMP 配置。
3) `run_train_loop`：执行训练，记录 loss/吞吐/缩放因子。
4) `export_artifacts`：导出 JSON/CSV/曲线图/summary。
5) `main`：串联流程并输出结果目录。

用法：
  python code/mixed_precision.py
  python code/mixed_precision.py --amp-mode fp16
  python code/mixed_precision.py --amp-mode bf16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_OUTPUT_DIR = "output"


def build_default_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run mixed-precision training demo and export artifacts.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)

    # 训练参数。
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)

    # 混合精度参数。
    parser.add_argument(
        "--amp-mode",
        default="auto",
        choices=["auto", "no_amp", "fp16", "bf16"],
        help="混合精度模式。",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser.parse_known_args([])[0]


def set_seed(seed: int) -> None:
    """设置随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    """选择设备：CUDA > MPS > CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_output_dir(base_dir: Path, output_dir: str) -> Path:
    """解析输出目录：相对路径按 mixed_precision 目录解析。"""
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


def resolve_amp(amp_mode: str, device: torch.device) -> dict[str, Any]:
    """根据设备与参数解析最终 AMP 配置。"""
    result = {
        "enabled": False,
        "dtype": None,
        "use_scaler": False,
        "reason": "no_amp",
    }

    if amp_mode == "no_amp":
        return result

    # 自动模式。
    if amp_mode == "auto":
        if device.type == "cuda":
            # CUDA 优先 bf16（若支持），否则 fp16。
            bf16_supported = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            if bf16_supported:
                result.update({"enabled": True, "dtype": torch.bfloat16, "use_scaler": False, "reason": "auto_cuda_bf16"})
            else:
                result.update({"enabled": True, "dtype": torch.float16, "use_scaler": True, "reason": "auto_cuda_fp16"})
        elif device.type == "mps":
            result.update({"enabled": True, "dtype": torch.float16, "use_scaler": False, "reason": "auto_mps_fp16"})
        else:
            result.update({"enabled": True, "dtype": torch.bfloat16, "use_scaler": False, "reason": "auto_cpu_bf16"})
        return result

    # 强制模式。
    if amp_mode == "fp16":
        if device.type in {"cuda", "mps"}:
            result.update(
                {
                    "enabled": True,
                    "dtype": torch.float16,
                    "use_scaler": device.type == "cuda",
                    "reason": f"forced_fp16_{device.type}",
                }
            )
        else:
            result.update({"enabled": False, "dtype": None, "use_scaler": False, "reason": "fp16_not_supported_on_cpu"})
        return result

    if amp_mode == "bf16":
        if device.type == "cuda":
            bf16_supported = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
            if bf16_supported:
                result.update({"enabled": True, "dtype": torch.bfloat16, "use_scaler": False, "reason": "forced_cuda_bf16"})
            else:
                result.update({"enabled": False, "dtype": None, "use_scaler": False, "reason": "cuda_bf16_not_supported"})
        elif device.type == "cpu":
            result.update({"enabled": True, "dtype": torch.bfloat16, "use_scaler": False, "reason": "forced_cpu_bf16"})
        else:
            # mps 对 bf16 支持不稳定，回退 no_amp。
            result.update({"enabled": False, "dtype": None, "use_scaler": False, "reason": "bf16_not_supported_on_mps"})
        return result

    return result


def autocast_context(device: torch.device, amp_cfg: dict[str, Any]):
    """返回 autocast 上下文管理器。"""
    if not amp_cfg["enabled"] or amp_cfg["dtype"] is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_cfg["dtype"], enabled=True)


def maybe_sync(device: torch.device) -> None:
    """在异步设备上同步，保证计时准确。"""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


class ToyRegressor(nn.Module):
    """简单 MLP 回归器。"""

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
    """采样 toy 回归数据。"""
    x = torch.empty(batch_size, 1, device=device).uniform_(-math.pi, math.pi)
    y = torch.sin(3.0 * x) + 0.1 * torch.randn_like(x)
    return x, y


def save_checkpoint(
    checkpoints_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    log_item: dict[str, float],
) -> None:
    """保存训练 checkpoint。"""
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "log": log_item,
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, checkpoints_dir / f"checkpoint-{step}.pt")


def run_train_loop(
    model: ToyRegressor,
    args: argparse.Namespace,
    device: torch.device,
    amp_cfg: dict[str, Any],
    checkpoints_dir: Path,
) -> tuple[list[dict[str, float]], Any]:
    """执行训练循环并返回日志与 scaler。"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    scaler = None
    if amp_cfg["use_scaler"]:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    logs: list[dict[str, float]] = []
    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        x, y = sample_batch(args.batch_size, device)

        with autocast_context(device, amp_cfg):
            pred = model(x)
            loss = F.mse_loss(pred, y)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scale_val = float(scaler.get_scale())
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scale_val = 1.0

        with torch.no_grad():
            xv, yv = sample_batch(args.batch_size, device)
            with autocast_context(device, amp_cfg):
                val_loss = float(F.mse_loss(model(xv), yv).item())

        maybe_sync(device)
        t1 = time.perf_counter()
        sec = max(t1 - t0, 1e-9)
        samples_per_sec = float(args.batch_size / sec)

        item = {
            "step": float(step),
            "train_loss": float(loss.item()),
            "val_loss": val_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "samples_per_sec": samples_per_sec,
            "scaler": scale_val,
        }
        logs.append(item)

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            print(
                f"[Step {step:04d}] train_loss={item['train_loss']:.6f} "
                f"val_loss={item['val_loss']:.6f} "
                f"samples/s={item['samples_per_sec']:.2f}"
            )
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(checkpoints_dir, step, model, optimizer, scaler, item)

    return logs, scaler


def export_artifacts(
    logs: list[dict[str, float]],
    output_dir: Path,
    models_dir: Path,
    model: ToyRegressor,
    args: argparse.Namespace,
    amp_cfg: dict[str, Any],
    device: torch.device,
) -> Path:
    """导出日志、CSV、曲线图和 summary。"""
    metrics_dir = output_dir / "mixed_precision_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / "training_log.json").write_text(
        json.dumps(logs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (metrics_dir / "training_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "train_loss", "val_loss", "learning_rate", "samples_per_sec", "scaler"],
        )
        writer.writeheader()
        writer.writerows(logs)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"`matplotlib` is required for visualization: {exc}") from exc

    steps = [x["step"] for x in logs]
    tr = [x["train_loss"] for x in logs]
    va = [x["val_loss"] for x in logs]
    sps = [x["samples_per_sec"] for x in logs]
    sc = [x["scaler"] for x in logs]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(steps, tr, label="train_loss")
    axes[0, 0].plot(steps, va, label="val_loss")
    axes[0, 0].set_title("Mixed Precision Loss")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].set_ylabel("mse")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(steps, sps, color="#2ca02c")
    axes[0, 1].set_title("Throughput (samples/s)")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(steps, sc, color="#ff7f0e")
    axes[1, 0].set_title("Grad Scaler")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.0,
        0.95,
        "\n".join(
            [
                f"device: {device.type}",
                f"amp_enabled: {amp_cfg['enabled']}",
                f"amp_dtype: {str(amp_cfg['dtype'])}",
                f"use_scaler: {amp_cfg['use_scaler']}",
                f"reason: {amp_cfg['reason']}",
            ]
        ),
        va="top",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(metrics_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, models_dir / "mixed_precision_model.pt")

    summary = {
        "device": device.type,
        "amp_mode_input": args.amp_mode,
        "amp_enabled": amp_cfg["enabled"],
        "amp_dtype": str(amp_cfg["dtype"]),
        "use_scaler": amp_cfg["use_scaler"],
        "reason": amp_cfg["reason"],
        "steps": len(logs),
        "final_train_loss": tr[-1] if tr else None,
        "final_val_loss": va[-1] if va else None,
        "best_val_loss": min(va) if va else None,
        "avg_samples_per_sec": float(np.mean(sps)) if sps else None,
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics_dir


def main() -> None:
    """主流程入口。"""
    args = build_default_args()
    set_seed(args.seed)
    device = choose_device()
    amp_cfg = resolve_amp(args.amp_mode, device)

    print(
        f"Runtime: device={device.type}, amp_enabled={amp_cfg['enabled']}, "
        f"dtype={amp_cfg['dtype']}, scaler={amp_cfg['use_scaler']}"
    )

    code_dir = Path(__file__).resolve().parent
    module_dir = code_dir.parent
    layout = ensure_layout_dirs(module_dir=module_dir, output_arg=args.output_dir)

    (layout["output"] / "mixed_precision_run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (layout["output"] / "mixed_precision_resolved_amp.json").write_text(
        json.dumps(
            {
                "enabled": amp_cfg["enabled"],
                "dtype": str(amp_cfg["dtype"]),
                "use_scaler": amp_cfg["use_scaler"],
                "reason": amp_cfg["reason"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    model = ToyRegressor(hidden_dim=args.hidden_dim)
    logs, _ = run_train_loop(
        model=model,
        args=args,
        device=device,
        amp_cfg=amp_cfg,
        checkpoints_dir=layout["checkpoints"],
    )
    metrics_dir = export_artifacts(
        logs=logs,
        output_dir=layout["output"],
        models_dir=layout["models"],
        model=model,
        args=args,
        amp_cfg=amp_cfg,
        device=device,
    )
    print(f"Mixed precision demo done. Visualization exported to: {metrics_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
