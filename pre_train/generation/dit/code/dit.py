#!/usr/bin/env python3
"""
轻量级 DiT（Diffusion Transformer）训练示例（Toy Gray Images）。

一、DiT 原理（面向实现）
1) 前向扩散：对真实图像逐步加噪，得到 x_t。
2) DiT 模型：用 Transformer 在 patch token 上预测噪声 epsilon。
3) 训练目标：最小化“真实噪声 vs 预测噪声”的 MSE。
4) 反向采样：从高斯噪声出发，逐步去噪生成图像。

二、代码框架（从入口到结果）
1) `build_default_args`：读取训练与可视化参数。
2) `build_noise_schedule`：构造扩散噪声调度。
3) `ToyDiT`：最小可运行的 DiT 网络（patchify + Transformer）。
4) `train_loop`：执行训练并保存 checkpoint。
5) `sample_reverse_process`：执行反向扩散采样。
6) `export_artifacts`：导出 CSV/JSON/曲线图/采样图/summary。

用法：
  python code/dit.py
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


def build_default_args() -> argparse.Namespace:
    """解析命令行参数，返回 DiT 训练与可视化配置。"""
    parser = argparse.ArgumentParser(description="Run toy DiT training and export visualization artifacts.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)

    # 图像与扩散参数。
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)

    # DiT 模型参数。
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    # 训练参数：默认可在 CPU/MPS 快速跑通。
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-every-epochs", type=int, default=10)

    # 可视化参数。
    parser.add_argument("--num-vis-samples", type=int, default=16)
    return parser.parse_known_args([])[0]


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
    """解析输出目录：相对路径按 dit 目录解析，绝对路径原样使用。"""
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
    """构建线性 beta 调度及其派生量。"""
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


def sample_toy_images(batch_size: int, image_size: int, device: torch.device) -> torch.Tensor:
    """采样 toy 灰度图：随机高斯 blob 的组合，范围归一化到 [-1, 1]。"""
    h = w = image_size
    ys = torch.linspace(-1.0, 1.0, h, device=device)
    xs = torch.linspace(-1.0, 1.0, w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1).view(1, 1, h, w, 2)  # [1,1,H,W,2]

    num_blobs = 2
    centers = torch.empty(batch_size, num_blobs, 2, device=device).uniform_(-0.8, 0.8)
    sigmas = torch.empty(batch_size, num_blobs, 1, 1, device=device).uniform_(0.08, 0.28)
    amps = torch.empty(batch_size, num_blobs, 1, 1, device=device).uniform_(0.6, 1.0)

    diff = grid - centers.view(batch_size, num_blobs, 1, 1, 2)
    dist2 = (diff**2).sum(dim=-1)
    blobs = amps * torch.exp(-dist2 / (2.0 * sigmas**2))
    images = blobs.sum(dim=1, keepdim=True)  # [B,1,H,W]

    mins = images.amin(dim=(2, 3), keepdim=True)
    maxs = images.amax(dim=(2, 3), keepdim=True)
    images = (images - mins) / (maxs - mins + 1e-6)
    return images * 2.0 - 1.0


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    sqrt_alpha_bars: torch.Tensor,
    sqrt_one_minus_alpha_bars: torch.Tensor,
) -> torch.Tensor:
    """前向扩散：给定 x0 和 t，生成 x_t。"""
    a = sqrt_alpha_bars[t].view(-1, 1, 1, 1)
    b = sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
    return a * x0 + b * noise


class SinusoidalTimeEmbedding(nn.Module):
    """将离散时间步 t 编码为连续向量。"""

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


class ToyDiT(nn.Module):
    """最小可运行 DiT：Patch Embedding + Transformer + 噪声预测头。"""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        time_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.patch_dim = patch_size * patch_size

        self.patch_embed = nn.Linear(self.patch_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        ff_dim = int(hidden_dim * mlp_ratio)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.patch_dim)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """把图像切成 patch token。"""
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.view(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # [B,gh,gw,C,p,p]
        return x.view(b, self.num_patches, c * p * p)

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """把 patch token 还原成图像。"""
        b = tokens.shape[0]
        p = self.patch_size
        g = self.grid_size
        x = tokens.view(b, g, g, 1, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # [B,1,gh,p,gw,p]
        return x.view(b, 1, self.image_size, self.image_size)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """输入带噪图像 x_t 和时间步 t，输出预测噪声。"""
        tokens = self.patch_embed(self.patchify(x_t))
        t_emb = self.time_embed(t).unsqueeze(1)
        h = self.transformer(tokens + self.pos_embed + t_emb)
        pred_tokens = self.out_proj(self.norm(h))
        return self.unpatchify(pred_tokens)


def save_checkpoint(
    checkpoints_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    avg_loss: float,
) -> None:
    """保存训练 checkpoint。"""
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
    model: ToyDiT,
    optimizer: torch.optim.Optimizer,
    schedule: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    checkpoints_dir: Path,
) -> list[dict[str, Any]]:
    """执行 DiT 训练循环并返回日志历史。"""
    model.train()
    logs: list[dict[str, Any]] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_losses: list[float] = []
        for _ in range(args.steps_per_epoch):
            x0 = sample_toy_images(args.batch_size, args.image_size, device)
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
    model: ToyDiT,
    schedule: dict[str, torch.Tensor],
    num_samples: int,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    """执行反向扩散采样，返回生成图像。"""
    model.eval()
    x = torch.randn(num_samples, 1, image_size, image_size, device=device)
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

    return x.clamp(-1.0, 1.0).cpu()


def build_image_grid(images: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """将 [N,1,H,W] 图像拼接为单张网格图。"""
    n, _, h, w = images.shape
    total = min(n, rows * cols)
    canvas = torch.ones(rows * h, cols * w)
    normed = (images[:total, 0] + 1.0) * 0.5  # [-1,1] -> [0,1]
    for i in range(total):
        r = i // cols
        c = i % cols
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = normed[i]
    return canvas


def export_artifacts(
    output_dir: Path,
    logs: list[dict[str, Any]],
    generated_samples: torch.Tensor,
    target_samples: torch.Tensor,
) -> Path:
    """导出日志、CSV、曲线图、采样图和 summary。"""
    metrics_dir = output_dir / "dit_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / "log_history.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")

    with (metrics_dir / "training_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "epoch", "loss", "learning_rate"])
        writer.writeheader()
        writer.writerows(logs)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"`matplotlib` is required to generate visualization: {exc}") from exc

    steps = [item["step"] for item in logs]
    losses = [item["loss"] for item in logs]
    rows = cols = int(math.sqrt(min(generated_samples.shape[0], 16)))
    rows = max(rows, 1)
    cols = rows

    gen_grid = build_image_grid(generated_samples, rows=rows, cols=cols)
    tgt_grid = build_image_grid(target_samples, rows=rows, cols=cols)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(steps, losses, marker="o", alpha=0.7)
    axes[0].set_title("DiT Training Loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("mse loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].imshow(tgt_grid.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Target Samples")
    axes[1].axis("off")

    axes[2].imshow(gen_grid.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title("Generated Samples")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(metrics_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    torch.save(generated_samples, metrics_dir / "generated_samples.pt")
    torch.save(target_samples, metrics_dir / "target_samples.pt")

    summary = {
        "total_logged_steps": len(logs),
        "final_step": logs[-1]["step"] if logs else None,
        "final_loss": logs[-1]["loss"] if logs else None,
        "best_loss": min((item["loss"] for item in logs), default=None),
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics_dir


def main() -> None:
    """主流程入口：训练 toy DiT 并导出可视化结果。"""
    args = build_default_args()
    if args.image_size % args.patch_size != 0:
        raise ValueError("--image-size must be divisible by --patch-size")

    code_dir = Path(__file__).resolve().parent
    module_dir = code_dir.parent
    layout = ensure_layout_dirs(module_dir=module_dir, output_arg=args.output_dir)
    (layout["output"] / "train_dit_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    set_seed(args.seed)
    device = detect_device()
    print(f"Runtime: device={device.type}")

    schedule = build_noise_schedule(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )

    model = ToyDiT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        hidden_dim=args.hidden_dim,
        time_dim=args.time_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    logs = train_loop(
        model=model,
        optimizer=optimizer,
        schedule=schedule,
        args=args,
        device=device,
        checkpoints_dir=layout["checkpoints"],
    )

    model_path = layout["models"] / "dit_final.pt"
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, model_path)
    print(f"Model saved: {model_path}")

    generated = sample_reverse_process(
        model=model,
        schedule=schedule,
        num_samples=max(args.num_vis_samples, 1),
        image_size=args.image_size,
        device=device,
    )
    target = sample_toy_images(max(args.num_vis_samples, 1), args.image_size, device).cpu()
    metrics_dir = export_artifacts(
        output_dir=layout["output"],
        logs=logs,
        generated_samples=generated,
        target_samples=target,
    )
    print(f"DiT done. Visualization exported to: {metrics_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
