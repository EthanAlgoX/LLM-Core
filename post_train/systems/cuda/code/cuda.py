#!/usr/bin/env python3
"""
CUDA 学习示例：环境检测 + 基准测试 + 简单训练曲线。

一、CUDA 原理（面向实现）
1) CUDA 是 NVIDIA GPU 并行计算平台，可显著加速张量运算与深度学习训练。
2) 训练中常见关键指标：算子耗时、吞吐、显存占用、数值稳定性。
3) 本示例聚焦三件事：
   - 检查 CUDA 可用性与设备信息；
   - 比较不同矩阵尺寸的 matmul 速度；
   - 在当前设备上跑一个可视化的 toy 训练过程。

二、代码框架（从入口到结果）
1) `parse_args`：读取测试参数。
2) `collect_cuda_info`：采集 CUDA 设备信息。
3) `run_matmul_benchmark`：执行矩阵乘 benchmark。
4) `run_toy_train`：执行简化训练并记录 loss。
5) `export_artifacts`：导出 JSON/CSV/曲线图/summary。
6) `main`：串联完整流程。

用法：
  python code/cuda.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_OUTPUT_DIR = "output"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run CUDA diagnostics and export visualization artifacts.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)

    # 基准参数。
    parser.add_argument("--benchmark-iters", type=int, default=30)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--matmul-sizes", default="256,512,1024")

    # toy 训练参数。
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """设置随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_output_dir(base_dir: Path, output_dir: str) -> Path:
    """解析输出目录：相对路径按 cuda 目录解析。"""
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


def collect_cuda_info() -> dict[str, Any]:
    """收集 CUDA 与设备信息。"""
    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_available": bool(torch.backends.cudnn.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "devices": [],
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            info["devices"].append(
                {
                    "index": i,
                    "name": prop.name,
                    "total_memory_gb": round(prop.total_memory / (1024**3), 3),
                    "multi_processor_count": int(prop.multi_processor_count),
                    "capability": f"{prop.major}.{prop.minor}",
                }
            )
    return info


def choose_device() -> torch.device:
    """选择运算设备：CUDA > MPS > CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_sizes(text: str) -> list[int]:
    """解析矩阵尺寸列表字符串。"""
    out: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise ValueError("--matmul-sizes cannot be empty")
    return out


def synchronize_if_needed(device: torch.device) -> None:
    """在异步设备上同步计时。"""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def run_matmul_benchmark(
    device: torch.device,
    sizes: list[int],
    warmup_iters: int,
    benchmark_iters: int,
) -> list[dict[str, float]]:
    """执行矩阵乘基准测试。"""
    results: list[dict[str, float]] = []
    for n in sizes:
        a = torch.randn(n, n, device=device)
        b = torch.randn(n, n, device=device)

        for _ in range(warmup_iters):
            _ = a @ b
        synchronize_if_needed(device)

        t0 = time.perf_counter()
        for _ in range(benchmark_iters):
            _ = a @ b
        synchronize_if_needed(device)
        t1 = time.perf_counter()

        avg_ms = (t1 - t0) * 1000.0 / benchmark_iters
        flops = 2.0 * (n**3)
        tflops = (flops / (avg_ms / 1000.0)) / 1e12
        results.append({"size": float(n), "avg_ms": float(avg_ms), "tflops": float(tflops)})
    return results


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
    """采样 toy 回归数据：y = sin(3x) + 噪声。"""
    x = torch.empty(batch_size, 1, device=device).uniform_(-math.pi, math.pi)
    y = torch.sin(3.0 * x) + 0.1 * torch.randn_like(x)
    return x, y


def run_toy_train(
    device: torch.device,
    train_steps: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    log_every: int,
) -> tuple[list[dict[str, float]], ToyRegressor]:
    """执行 toy 训练并记录损失曲线。"""
    model = ToyRegressor(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    logs: list[dict[str, float]] = []
    for step in range(1, train_steps + 1):
        x, y = sample_batch(batch_size, device)
        pred = model(x)
        loss = F.mse_loss(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            xv, yv = sample_batch(batch_size, device)
            val_loss = float(F.mse_loss(model(xv), yv).item())

        item = {
            "step": float(step),
            "train_loss": float(loss.item()),
            "val_loss": val_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        logs.append(item)
        if step % log_every == 0 or step == 1 or step == train_steps:
            print(
                f"[Step {step:04d}] train_loss={item['train_loss']:.6f} "
                f"val_loss={item['val_loss']:.6f}"
            )
    return logs, model


def export_artifacts(
    cuda_info: dict[str, Any],
    benchmark: list[dict[str, float]],
    train_logs: list[dict[str, float]],
    output_dir: Path,
    models_dir: Path,
    model: ToyRegressor,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    """导出 JSON/CSV/曲线图/summary。"""
    metrics_dir = output_dir / "cuda_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / "cuda_info.json").write_text(json.dumps(cuda_info, ensure_ascii=False, indent=2), encoding="utf-8")
    (metrics_dir / "benchmark.json").write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
    (metrics_dir / "training_log.json").write_text(json.dumps(train_logs, ensure_ascii=False, indent=2), encoding="utf-8")

    with (metrics_dir / "benchmark.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["size", "avg_ms", "tflops"])
        writer.writeheader()
        writer.writerows(benchmark)

    with (metrics_dir / "training_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss", "learning_rate"])
        writer.writeheader()
        writer.writerows(train_logs)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"`matplotlib` is required for visualization: {exc}") from exc

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sizes = [x["size"] for x in benchmark]
    avg_ms = [x["avg_ms"] for x in benchmark]
    tflops = [x["tflops"] for x in benchmark]
    axes[0].plot(sizes, avg_ms, marker="o", label="avg_ms")
    axes[0].set_title(f"Matmul Latency ({device.type})")
    axes[0].set_xlabel("matrix size")
    axes[0].set_ylabel("ms")
    axes[0].grid(True, alpha=0.3)

    ax2 = axes[0].twinx()
    ax2.plot(sizes, tflops, marker="s", color="#ff7f0e", label="TFLOPS")
    ax2.set_ylabel("TFLOPS")

    steps = [x["step"] for x in train_logs]
    tr = [x["train_loss"] for x in train_logs]
    va = [x["val_loss"] for x in train_logs]
    axes[1].plot(steps, tr, label="train_loss")
    axes[1].plot(steps, va, label="val_loss")
    axes[1].set_title("Toy Training Curve")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("mse")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(metrics_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, models_dir / "cuda_toy_model.pt")

    summary = {
        "runtime_device": device.type,
        "cuda_available": cuda_info["cuda_available"],
        "num_cuda_devices": cuda_info["device_count"],
        "final_train_loss": tr[-1] if tr else None,
        "final_val_loss": va[-1] if va else None,
        "best_val_loss": min(va) if va else None,
        "max_benchmark_tflops": max(tflops) if tflops else None,
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics_dir


def main() -> None:
    """主流程入口。"""
    args = parse_args()
    set_seed(args.seed)

    code_dir = Path(__file__).resolve().parent
    module_dir = code_dir.parent
    layout = ensure_layout_dirs(module_dir=module_dir, output_arg=args.output_dir)
    (layout["output"] / "cuda_run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cuda_info = collect_cuda_info()
    device = choose_device()
    print(
        f"Runtime: device={device.type}, "
        f"cuda_available={cuda_info['cuda_available']}, "
        f"cuda_devices={cuda_info['device_count']}"
    )

    sizes = parse_sizes(args.matmul_sizes)
    benchmark = run_matmul_benchmark(
        device=device,
        sizes=sizes,
        warmup_iters=args.warmup_iters,
        benchmark_iters=args.benchmark_iters,
    )
    train_logs, model = run_toy_train(
        device=device,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        log_every=args.log_every,
    )

    metrics_dir = export_artifacts(
        cuda_info=cuda_info,
        benchmark=benchmark,
        train_logs=train_logs,
        output_dir=layout["output"],
        models_dir=layout["models"],
        model=model,
        args=args,
        device=device,
    )
    print(f"CUDA demo done. Visualization exported to: {metrics_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
