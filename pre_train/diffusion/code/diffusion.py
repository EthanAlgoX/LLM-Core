#!/usr/bin/env python3
"""
轻量级 Diffusion 训练示例（2D Toy Data）。

一、Diffusion 原理（面向实现）
1) 前向扩散：逐步向真实样本加入高斯噪声，直到接近纯噪声。
2) 反向去噪：训练模型预测噪声，从纯噪声逐步还原样本。
3) 训练目标：最小化“真实噪声 vs 预测噪声”的 MSE。
4) 采样阶段：从随机噪声出发，按时间步反向迭代得到生成样本。

二、代码框架（从入口到结果）
1) `parse_args`：读取训练与可视化参数。
2) `build_noise_schedule`：构造 beta/alpha/alpha_bar 噪声调度。
3) `train_loop`：执行训练并保存 checkpoint。
4) `sample_reverse_process`：执行反向去噪采样。
5) `export_artifacts`：导出 CSV/JSON/曲线图/采样图。
6) `main`：串联完整流程。

用法：
  python code/diffusion.py
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

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_OUTPUT_DIR = "output"


def parse_args() -> argparse.Namespace:
    """解析命令行参数，返回 Diffusion 训练与可视化配置。"""
    parser = argparse.ArgumentParser(description="Run toy diffusion training and export visualization artifacts.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)

    # 模型与扩散参数：保持最简、便于学习。
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--time-dim", type=int, default=64)

    # 训练参数：默认可在 CPU 上快速跑通。
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-every-epochs", type=int, default=10)

    # 可视化参数。
    parser.add_argument("--num-vis-samples", type=int, default=2000)
    return parser.parse_args()


def detect_device() -> torch.device:
    """选择训练设备：CUDA > MPS > CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """设置随机种子，增强实验可复现性。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_output_dir(base_dir: Path, output_dir: str) -> Path:
    """解析输出目录：相对路径按 diffusion 目录解析，绝对路径原样使用。"""
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


def build_noise_schedule(
    timesteps: int,
    beta_start: float,
    beta_end: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """构建线性 beta 调度及其派生项。"""
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": torch.sqrt(alpha_bars),
        "sqrt_one_minus_alpha_bars": torch.sqrt(1.0 - alpha_bars),
    }


def sample_toy_data(batch_size: int, device: torch.device) -> torch.Tensor:
    """采样 2D 八高斯环形分布，作为 toy 训练数据。"""
    centers = torch.tensor(
        [
            [1.0, 0.0],
            [0.707, 0.707],
            [0.0, 1.0],
            [-0.707, 0.707],
            [-1.0, 0.0],
            [-0.707, -0.707],
            [0.0, -1.0],
            [0.707, -0.707],
        ],
        device=device,
        dtype=torch.float32,
    ) * 2.0
    indices = torch.randint(0, centers.shape[0], (batch_size,), device=device)
    noise = 0.08 * torch.randn(batch_size, 2, device=device)
    return centers[indices] + noise


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    sqrt_alpha_bars: torch.Tensor,
    sqrt_one_minus_alpha_bars: torch.Tensor,
) -> torch.Tensor:
    """前向扩散：给定 x0 和时间步 t，得到带噪样本 x_t。"""
    a = sqrt_alpha_bars[t].unsqueeze(1)
    b = sqrt_one_minus_alpha_bars[t].unsqueeze(1)
    return a * x0 + b * noise


class SinusoidalTimeEmbedding(nn.Module):
    """将离散时间步 t 编码为连续向量表示。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        angles = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class DenoiseMLP(nn.Module):
    """简单噪声预测网络：输入 x_t 与 t，输出预测噪声 epsilon。"""

    def __init__(self, hidden_dim: int, time_dim: int) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x_t, t_emb], dim=1))


def save_checkpoint(
    checkpoints_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    avg_loss: float,
) -> None:
    """保存训练 checkpoint（模型、优化器和当前 epoch 指标）。"""
    ckpt_dir = checkpoints_dir / f"checkpoint-{epoch}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_dir / "state.pt",
    )


def train_loop(
    model: DenoiseMLP,
    optimizer: torch.optim.Optimizer,
    schedule: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    checkpoints_dir: Path,
) -> list[dict[str, Any]]:
    """执行训练循环并返回日志历史。"""
    model.train()
    logs: list[dict[str, Any]] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_losses: list[float] = []
        for _ in range(args.steps_per_epoch):
            x0 = sample_toy_data(args.batch_size, device)
            t = torch.randint(0, args.timesteps, (args.batch_size,), device=device)
            noise = torch.randn_like(x0)
            x_t = q_sample(
                x0=x0,
                t=t,
                noise=noise,
                sqrt_alpha_bars=schedule["sqrt_alpha_bars"],
                sqrt_one_minus_alpha_bars=schedule["sqrt_one_minus_alpha_bars"],
            )

            pred_noise = model(x_t, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            loss_val = float(loss.item())
            epoch_losses.append(loss_val)

            if global_step % args.logging_steps == 0:
                logs.append(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": loss_val,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    }
                )

        avg_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
        print(f"[Epoch {epoch:03d}] avg_loss={avg_loss:.6f}")

        if epoch % args.save_every_epochs == 0 or epoch == args.epochs:
            save_checkpoint(
                checkpoints_dir=checkpoints_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                avg_loss=avg_loss,
            )

    return logs


@torch.no_grad()
def sample_reverse_process(
    model: DenoiseMLP,
    schedule: dict[str, torch.Tensor],
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """从高斯噪声出发执行反向扩散，生成样本点。"""
    model.eval()
    x = torch.randn(num_samples, 2, device=device)
    betas = schedule["betas"]
    alphas = schedule["alphas"]
    alpha_bars = schedule["alpha_bars"]

    for i in reversed(range(betas.shape[0])):
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)
        eps = model(x, t)
        alpha_t = alphas[i]
        beta_t = betas[i]
        alpha_bar_t = alpha_bars[i]

        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
        z = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
        x = coef1 * (x - coef2 * eps) + torch.sqrt(beta_t) * z

    return x.cpu()


def export_artifacts(
    output_dir: Path,
    logs: list[dict[str, Any]],
    generated_samples: torch.Tensor,
    target_samples: torch.Tensor,
) -> Path:
    """导出日志、CSV、曲线图、采样图和 summary。"""
    metrics_dir = output_dir / "diffusion_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / "log_history.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")

    csv_path = metrics_dir / "training_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "epoch", "loss", "learning_rate"])
        writer.writeheader()
        writer.writerows(logs)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"`matplotlib` is required to generate visualization: {exc}") from exc

    steps = [r["step"] for r in logs]
    losses = [r["loss"] for r in logs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(steps, losses, marker="o", alpha=0.7)
    axes[0].set_title("Diffusion Training Loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("mse loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(target_samples[:, 0], target_samples[:, 1], s=8, alpha=0.35, label="target")
    axes[1].scatter(generated_samples[:, 0], generated_samples[:, 1], s=8, alpha=0.35, label="generated")
    axes[1].set_title("Target vs Generated Samples")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(metrics_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    torch.save(generated_samples, metrics_dir / "generated_samples.pt")
    torch.save(target_samples, metrics_dir / "target_samples.pt")

    summary = {
        "total_logged_steps": len(logs),
        "final_step": logs[-1]["step"] if logs else None,
        "final_loss": logs[-1]["loss"] if logs else None,
        "best_loss": min((r["loss"] for r in logs), default=None),
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics_dir


def main() -> None:
    """主流程入口：训练 toy diffusion 并导出可视化结果。"""
    args = parse_args()
    code_dir = Path(__file__).resolve().parent
    module_dir = code_dir.parent
    layout = ensure_layout_dirs(module_dir=module_dir, output_arg=args.output_dir)

    set_seed(args.seed)
    device = detect_device()
    print(f"Runtime: device={device.type}")

    schedule = build_noise_schedule(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )

    model = DenoiseMLP(hidden_dim=args.hidden_dim, time_dim=args.time_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    logs = train_loop(
        model=model,
        optimizer=optimizer,
        schedule=schedule,
        args=args,
        device=device,
        checkpoints_dir=layout["checkpoints"],
    )

    model_path = layout["models"] / "diffusion_mlp_final.pt"
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, model_path)
    print(f"Model saved: {model_path}")

    generated = sample_reverse_process(
        model=model,
        schedule=schedule,
        num_samples=args.num_vis_samples,
        device=device,
    )
    target = sample_toy_data(args.num_vis_samples, device).cpu()
    metrics_dir = export_artifacts(
        output_dir=layout["output"],
        logs=logs,
        generated_samples=generated,
        target_samples=target,
    )
    print(f"Diffusion done. Visualization exported to: {metrics_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
