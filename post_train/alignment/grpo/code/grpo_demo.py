#!/usr/bin/env python3
"""
GRPO 单文件学习脚本（主流程 + 可视化）。

结构约定：
1) `main`：主训练流程（准备数据 -> 训练 -> 保存模型）。
2) `export_grpo_visualization`：唯一可视化函数（导出 JSON/CSV/曲线图/summary）。

新人阅读顺序（建议）：
1) 先看 `GRPO_CONFIG`：理解采样、奖励和训练超参。
2) 再看 `main`：理解“合成数据 -> 奖励函数 -> 训练器 -> 训练”主链路。
3) 最后看 `export_grpo_visualization`：理解如何分析奖励与 loss 曲线。

学习步骤（与终端输出 1~5 对应）：
1) 准备目录与合成训练数据。
2) 构建奖励函数与 GRPO 参数。
3) 构建 GRPOTrainer（兼容不同 TRL 接口）。
4) 执行训练并保存模型。
5) 导出可视化与摘要。
"""

from __future__ import annotations

import csv
import inspect
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer


GRPO_CONFIG = {
    "model_name": "Qwen/Qwen3-0.6B",  # 策略模型名称或路径。
    "output_dir": "output",  # 输出目录。
    "seed": 42,  # 随机种子，保证复现。
    "train_size": 4,  # 合成训练样本数量。
    "learning_rate": 5e-7,  # 学习率。
    "num_train_epochs": 0.05,  # 训练轮数。
    "per_device_train_batch_size": 1,  # 单卡 batch。
    "gradient_accumulation_steps": 1,  # 梯度累积步数。
    "num_generations": 2,  # 每个 prompt 采样候选数。
    "generation_batch_size": 2,  # 生成阶段 batch。
    "max_prompt_length": 384,  # prompt 最大长度。
    "max_completion_length": 56,  # completion 最大长度。
    "logging_steps": 1,  # 日志间隔。
    "save_steps": 100,  # 保存间隔。
    "temperature": 1.1,  # 采样温度。
    "top_p": 0.95,  # nucleus 采样阈值。
    "reward_weights": [1.0, 0.3, 0.2, 0.05],  # 奖励权重 [correctness,distance,format,compact]。
    "ema_alpha": 0.25,  # 可视化曲线平滑系数。
}


def export_grpo_visualization(trainer: GRPOTrainer, output_dir: Path, train_metrics: dict[str, Any]) -> Path:
    """导出 GRPO 可视化：JSON、CSV、曲线图与 summary。"""
    print("5) 导出可视化结果", flush=True)

    metrics_dir = output_dir / "grpo_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    log_history = trainer.state.log_history
    (metrics_dir / "log_history.json").write_text(
        json.dumps(log_history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (metrics_dir / "train_summary.json").write_text(
        json.dumps(train_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 新手优先关注：
    # loss/reward/reward_std 先判断学习是否有效，再看各 reward 分量贡献。
    keys = [
        "step",
        "epoch",
        "loss",
        "learning_rate",
        "grad_norm",
        "reward",
        "reward_std",
        "rewards/correctness_reward/mean",
        "rewards/distance_reward/mean",
        "rewards/format_reward/mean",
        "rewards/compact_output_reward/mean",
        "entropy",
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
        axes[0, 0].plot(x, ema(y, GRPO_CONFIG["ema_alpha"]), linewidth=2, label=f"loss(ema={GRPO_CONFIG['ema_alpha']})")
    axes[0, 0].set_title("GRPO Loss")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    x, y = series("reward")
    axes[0, 1].plot(x, y, marker="o", alpha=0.45, label="reward(raw)")
    if y:
        axes[0, 1].plot(
            x,
            ema(y, GRPO_CONFIG["ema_alpha"]),
            linewidth=2,
            label=f"reward(ema={GRPO_CONFIG['ema_alpha']})",
        )
    axes[0, 1].set_title("Total Reward")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    x1, y1 = series("rewards/correctness_reward/mean")
    x2, y2 = series("rewards/distance_reward/mean")
    x3, y3 = series("rewards/format_reward/mean")
    x4, y4 = series("rewards/compact_output_reward/mean")
    axes[1, 0].plot(x1, y1, marker="o", label="correctness")
    axes[1, 0].plot(x2, y2, marker="o", label="distance")
    axes[1, 0].plot(x3, y3, marker="o", label="format")
    axes[1, 0].plot(x4, y4, marker="o", label="compact")
    axes[1, 0].set_title("Reward Components")
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
        "final_reward": next((r["reward"] for r in reversed(rows) if r.get("reward") is not None), None),
        "best_reward": max((r["reward"] for r in rows if r.get("reward") is not None), default=None),
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics_dir


def main() -> None:
    """主训练流程：准备目录和数据 -> 构建配置 -> 执行训练 -> 导出可视化。"""
    print("=== GRPO 主流程（学习版）===", flush=True)
    # 新手提示：终端步骤号（1~5）可直接对应本函数注释。

    # 步骤 1：准备目录、随机种子与教学用合成数据。
    print("1) 准备目录与训练数据", flush=True)

    code_dir = Path(__file__).resolve().parent
    module_dir = code_dir.parent
    output_dir = (module_dir / GRPO_CONFIG["output_dir"]).resolve()
    checkpoints_dir = module_dir / "checkpoints"
    models_dir = module_dir / "models"
    data_dir = module_dir / "data"
    for p in [module_dir / "code", data_dir, models_dir, output_dir, checkpoints_dir]:
        p.mkdir(parents=True, exist_ok=True)

    set_seed(GRPO_CONFIG["seed"])
    rng = random.Random(GRPO_CONFIG["seed"])

    # 内部小工具：仅在主流程里使用，减少新手在文件内来回跳转。
    strict_format_pattern = re.compile(r"^\s*<reasoning>.*?</reasoning>\s*<answer>\s*[-+]?\d+\s*</answer>\s*$", re.S)
    answer_pattern = re.compile(r"<answer>\s*([-+]?\d+)\s*</answer>", re.S)

    def completion_to_text(completion: Any) -> str:
        if isinstance(completion, str):
            return completion
        if isinstance(completion, dict):
            return str(completion.get("content", ""))
        if isinstance(completion, list):
            parts = []
            for item in completion:
                if isinstance(item, dict):
                    parts.append(str(item.get("content", "")))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(completion)

    def extract_answer(text: str) -> str | None:
        match = answer_pattern.search(text)
        if match:
            return match.group(1)
        nums = re.findall(r"[-+]?\d+", text)
        return nums[-1] if nums else None

    def make_sample() -> dict[str, str]:
        op = rng.choice(["+", "-", "*", "/"])
        if op == "+":
            a, b = rng.randint(1, 40), rng.randint(1, 40)
            ans = a + b
        elif op == "-":
            a, b = rng.randint(1, 60), rng.randint(1, 60)
            if a < b:
                a, b = b, a
            ans = a - b
        elif op == "*":
            a, b = rng.randint(1, 12), rng.randint(1, 12)
            ans = a * b
        else:
            b = rng.randint(1, 12)
            ans = rng.randint(1, 20)
            a = b * ans
        return {"prompt": f"{a} {op} {b}", "answer": str(ans)}

    format_example = (
        "示例:\n"
        "题目: 2 + 3\n"
        "输出:\n"
        "<reasoning>2+3=5</reasoning>\n"
        "<answer>5</answer>\n\n"
    )
    instruction = (
        "你是数学助手。请只输出两行，不要额外解释:\n"
        "<reasoning>简短计算过程</reasoning>\n"
        "<answer>整数答案</answer>\n\n"
    )

    samples = [make_sample() for _ in range(GRPO_CONFIG["train_size"])]
    train_dataset = Dataset.from_list(
        [
            {
                "prompt": f"{instruction}{format_example}题目: {item['prompt']}\n输出:",
                "answer": item["answer"],
            }
            for item in samples
        ]
    )

    (data_dir / "train_demo.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

    print("2) 构建奖励函数与 GRPO 配置", flush=True)

    def correctness_reward(completions, answer, **kwargs):
        rewards = []
        for completion, gold in zip(completions, answer):
            pred = extract_answer(completion_to_text(completion))
            rewards.append(1.0 if pred == str(gold) else 0.0)
        return rewards

    def distance_reward(completions, answer, **kwargs):
        rewards = []
        for completion, gold in zip(completions, answer):
            pred_text = extract_answer(completion_to_text(completion))
            if pred_text is None:
                rewards.append(-0.2)
                continue
            try:
                pred = int(pred_text)
                target = int(gold)
            except ValueError:
                rewards.append(-0.2)
                continue
            rewards.append(max(-0.2, 1.0 - abs(pred - target) / 10.0))
        return rewards

    def format_reward(completions, **kwargs):
        rewards = []
        for completion in completions:
            text = completion_to_text(completion)
            score = 0.0
            if "<reasoning>" in text and "</reasoning>" in text:
                score += 0.05
            if "<answer>" in text and "</answer>" in text:
                score += 0.05
            if strict_format_pattern.match(text):
                score += 0.10
            rewards.append(score)
        return rewards

    def compact_output_reward(completions, **kwargs):
        rewards = []
        for completion in completions:
            text = completion_to_text(completion).strip()
            rewards.append(0.03 if len(text) <= 120 else -0.03)
        return rewards

    reward_funcs = [correctness_reward, distance_reward, format_reward, compact_output_reward]
    reward_weights = GRPO_CONFIG["reward_weights"]

    if GRPO_CONFIG["generation_batch_size"] % GRPO_CONFIG["num_generations"] != 0:
        raise ValueError(
            "generation_batch_size 必须能被 num_generations 整除："
            f"{GRPO_CONFIG['generation_batch_size']} / {GRPO_CONFIG['num_generations']}"
        )

    desired = {
        "output_dir": str(checkpoints_dir),  # checkpoint 输出目录。
        "overwrite_output_dir": True,  # 覆盖旧输出。
        "learning_rate": GRPO_CONFIG["learning_rate"],  # 学习率。
        "per_device_train_batch_size": GRPO_CONFIG["per_device_train_batch_size"],  # 单卡 batch。
        "gradient_accumulation_steps": GRPO_CONFIG["gradient_accumulation_steps"],  # 梯度累积。
        "num_generations": GRPO_CONFIG["num_generations"],  # 每 prompt 采样数。
        "generation_batch_size": GRPO_CONFIG["generation_batch_size"],  # 生成 batch。
        "max_prompt_length": GRPO_CONFIG["max_prompt_length"],  # prompt 最大长度。
        "max_completion_length": GRPO_CONFIG["max_completion_length"],  # completion 最大长度。
        "num_train_epochs": GRPO_CONFIG["num_train_epochs"],  # 训练轮数。
        "logging_steps": GRPO_CONFIG["logging_steps"],  # 日志间隔。
        "save_steps": GRPO_CONFIG["save_steps"],  # 保存间隔。
        "lr_scheduler_type": "cosine",  # 学习率调度器。
        "warmup_steps": 1,  # 预热步数。
        "temperature": GRPO_CONFIG["temperature"],  # 采样温度。
        "top_p": GRPO_CONFIG["top_p"],  # top-p 采样。
        "reward_weights": reward_weights,  # 多奖励权重。
        "scale_rewards": "group",  # 组内奖励标准化，降低方差。
        "report_to": "none",  # 关闭外部上报。
    }
    supported = set(inspect.signature(GRPOConfig.__init__).parameters)
    grpo_args = GRPOConfig(**{k: v for k, v in desired.items() if k in supported})

    # 步骤 3：构建训练器，兼容不同 TRL 版本参数名。
    print("3) 构建 GRPOTrainer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(GRPO_CONFIG["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    init_params = inspect.signature(GRPOTrainer.__init__).parameters
    trainer_kwargs: dict[str, Any] = {
        "model": GRPO_CONFIG["model_name"],
        "args": grpo_args,
        "train_dataset": train_dataset,
    }
    if "processing_class" in init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in init_params:
        trainer_kwargs["tokenizer"] = tokenizer

    if "reward_funcs" in init_params:
        trainer_kwargs["reward_funcs"] = reward_funcs
    elif "reward_func" in init_params:
        trainer_kwargs["reward_func"] = correctness_reward
    else:
        raise RuntimeError("当前 TRL 版本未找到奖励函数参数（reward_funcs/reward_func）。")

    trainer = GRPOTrainer(**trainer_kwargs)

    # 步骤 4：执行训练并保存模型权重。
    print("4) 执行 GRPO 训练并保存模型", flush=True)
    train_output = trainer.train()
    trainer.save_model(str(models_dir))

    (output_dir / "run_config.json").write_text(
        json.dumps({**GRPO_CONFIG, "reward_weights": reward_weights}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # run_config.json 用于记录本次实验超参，便于复现实验与参数对比。

    # 步骤 5：导出学习指标、曲线图和 summary。
    metrics_dir = export_grpo_visualization(
        trainer=trainer,
        output_dir=output_dir,
        train_metrics=dict(train_output.metrics),
    )
    print(f"GRPO done. Visualization exported to: {metrics_dir}", flush=True)
    # 建议的结果阅读顺序：summary.json -> training_curves.png -> training_metrics.csv。


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
