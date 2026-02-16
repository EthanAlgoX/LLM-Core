#!/usr/bin/env python3
"""
Megatron 学习示例：并行配置 + Toy Causal LM 训练 + 可视化。

一、Megatron 原理（面向实现）
1) Megatron-LM 通过张量并行（TP）、流水线并行（PP）和数据并行协同扩展训练。
2) TP：将单层算子拆到多卡；PP：将网络层拆到多段；DP：复制模型并行处理不同 batch。
3) 本示例聚焦“配置理解 + 可运行训练”：
   - 生成 Megatron 风格并行配置；
   - 训练一个最小 Causal LM；
   - 导出训练曲线，便于学习并行配置与训练过程的关系。

二、代码框架（从入口到结果）
1) `build_default_args`：读取训练与并行参数。
2) `build_megatron_config`：生成并行配置快照。
3) `run_train_loop`：优先尝试 Megatron 依赖检查，训练路径默认 Torch。
4) `export_artifacts`：导出 JSON/CSV/曲线图/summary。
5) `main`：串联完整流程。

用法：
  python code/megatron.py
  python code/megatron.py --use-megatron --tensor-model-parallel-size 2 --pipeline-model-parallel-size 2
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


def build_default_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run Megatron demo training and export visualization artifacts.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)

    # 训练参数。
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)

    # Megatron 并行配置参数（用于学习与配置记录）。
    parser.add_argument("--use-megatron", action="store_true", help="尝试检测 Megatron 依赖并记录状态。")
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--global-batch-size", type=int, default=32)
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
    """解析输出目录：相对路径按 megatron 目录解析。"""
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


def build_megatron_config(args: argparse.Namespace) -> dict[str, Any]:
    """生成 Megatron 风格配置快照。"""
    world_size = (
        args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.data_parallel_size
    )
    grad_accum = max(1, args.global_batch_size // max(1, args.micro_batch_size * args.data_parallel_size))
    return {
        "tensor_model_parallel_size": args.tensor_model_parallel_size,
        "pipeline_model_parallel_size": args.pipeline_model_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.global_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "world_size_required": world_size,
        "note": "This demo uses torch training path; config is exported for Megatron parallelism learning.",
    }


def try_check_megatron() -> tuple[bool, str]:
    """检查 Megatron 依赖是否可导入。"""
    try:
        import megatron  # type: ignore  # noqa: F401
    except Exception as exc:
        return False, f"megatron import failed: {exc}"
    return True, "megatron import success"


def build_toy_corpus_ids(data_dir: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    """构建 toy 文本语料并编码为 token id。"""
    corpus = (
        "megatron enables scalable llm pretraining. "
        "tensor parallelism and pipeline parallelism improve efficiency. "
        "this project is for learning llm and vlm training systems. "
    )
    corpus = (corpus * 400).strip()
    vocab = sorted(set(corpus))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    ids = torch.tensor([stoi[ch] for ch in corpus], dtype=torch.long)

    info = {
        "corpus_length": int(len(corpus)),
        "vocab_size": int(len(vocab)),
        "vocab": "".join(vocab),
    }
    (data_dir / "corpus.txt").write_text(corpus, encoding="utf-8")
    (data_dir / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return ids, info


def sample_lm_batch(data_ids: torch.Tensor, seq_len: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """采样 next-token 训练 batch。"""
    n = data_ids.size(0)
    max_start = n - seq_len - 1
    idx = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data_ids[i : i + seq_len] for i in idx], dim=0).to(device)
    y = torch.stack([data_ids[i + 1 : i + seq_len + 1] for i in idx], dim=0).to(device)
    return x, y


class TinyCausalLM(nn.Module):
    """最小 Causal LM：token embedding + transformer encoder + lm head。"""

    def __init__(self, vocab_size: int, seq_len: int, hidden_dim: int, n_layers: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        if t > self.seq_len:
            raise ValueError(f"sequence length {t} exceeds model limit {self.seq_len}")
        h = self.token_emb(x) + self.pos_emb[:, :t, :]
        # Causal mask: upper triangle masked.
        mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        h = self.tr(h, mask=mask)
        h = self.ln(h)
        return self.head(h)


def save_checkpoint(
    checkpoints_dir: Path,
    step: int,
    model: TinyCausalLM,
    optimizer: torch.optim.Optimizer,
    log_item: dict[str, float],
) -> None:
    """保存训练 checkpoint。"""
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
    model: TinyCausalLM,
    data_ids: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    checkpoints_dir: Path,
) -> list[dict[str, float]]:
    """运行 toy Causal LM 训练。"""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    logs: list[dict[str, float]] = []
    for step in range(1, args.steps + 1):
        x, y = sample_lm_batch(data_ids, args.seq_len, args.batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # 简单验证损失。
        with torch.no_grad():
            xv, yv = sample_lm_batch(data_ids, args.seq_len, args.batch_size, device)
            lv = model(xv)
            val_loss = float(F.cross_entropy(lv.reshape(-1, lv.size(-1)), yv.reshape(-1)).item())

        item = {
            "step": float(step),
            "train_loss": float(loss.item()),
            "val_loss": val_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        logs.append(item)

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            print(
                f"[Step {step:04d}] train_loss={item['train_loss']:.6f} "
                f"val_loss={item['val_loss']:.6f}"
            )
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(checkpoints_dir, step, model, optimizer, item)
    return logs


def export_artifacts(
    logs: list[dict[str, float]],
    output_dir: Path,
    models_dir: Path,
    model: TinyCausalLM,
    args: argparse.Namespace,
    backend_note: str,
) -> Path:
    """导出日志、CSV、曲线图、summary。"""
    metrics_dir = output_dir / "megatron_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / "training_log.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    with (metrics_dir / "training_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss", "learning_rate"])
        writer.writeheader()
        writer.writerows(logs)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"`matplotlib` is required for visualization: {exc}") from exc

    steps = [x["step"] for x in logs]
    tr = [x["train_loss"] for x in logs]
    va = [x["val_loss"] for x in logs]
    lr = [x["learning_rate"] for x in logs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(steps, tr, label="train_loss")
    axes[0].plot(steps, va, label="val_loss")
    axes[0].set_title("Toy LM Loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("cross-entropy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, lr, color="#2ca02c")
    axes[1].set_title("Learning Rate")
    axes[1].set_xlabel("step")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(metrics_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, models_dir / "megatron_toy_lm.pt")

    summary = {
        "backend": backend_note,
        "steps": len(logs),
        "final_train_loss": tr[-1] if tr else None,
        "final_val_loss": va[-1] if va else None,
        "best_val_loss": min(va) if va else None,
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics_dir


def main() -> None:
    """主流程入口。"""
    args = build_default_args()
    set_seed(args.seed)
    device = choose_device()
    print(f"Runtime: device={device.type}")

    code_dir = Path(__file__).resolve().parent
    module_dir = code_dir.parent
    layout = ensure_layout_dirs(module_dir=module_dir, output_arg=args.output_dir)

    megatron_cfg = build_megatron_config(args)
    (layout["output"] / "megatron_run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (layout["output"] / "megatron_config_auto.json").write_text(
        json.dumps(megatron_cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    backend_note = "torch"
    if args.use_megatron:
        ok, msg = try_check_megatron()
        backend_note = "megatron_available(torch_demo)" if ok else "megatron_unavailable(torch_fallback)"
        print(f"[INFO] {msg}")

    data_ids, data_info = build_toy_corpus_ids(layout["data"])
    model = TinyCausalLM(
        vocab_size=data_info["vocab_size"],
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )

    logs = run_train_loop(
        model=model,
        data_ids=data_ids,
        args=args,
        device=device,
        checkpoints_dir=layout["checkpoints"],
    )
    metrics_dir = export_artifacts(
        logs=logs,
        output_dir=layout["output"],
        models_dir=layout["models"],
        model=model,
        args=args,
        backend_note=backend_note,
    )
    print(f"Megatron demo done. Visualization exported to: {metrics_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
