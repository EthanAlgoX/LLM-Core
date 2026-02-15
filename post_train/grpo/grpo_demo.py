#!/usr/bin/env python3
"""
GRPO demo (TRL): improved rewards + smoother loss visualization.
"""

import argparse
import csv
import inspect
import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer


STRICT_FORMAT_PATTERN = re.compile(
    r"^\s*<reasoning>.*?</reasoning>\s*<answer>\s*[-+]?\d+\s*</answer>\s*$",
    re.S,
)
ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*([-+]?\d+)\s*</answer>", re.S)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO toy training demo.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default="qwen3_grpo_out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=24, help="Number of synthetic math samples.")

    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--max-completion-length", type=int, default=56)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--reward-weights",
        default="1.0,0.3,0.2,0.05",
        help="Comma-separated weights for [correctness,distance,format,compact].",
    )
    parser.add_argument("--ema-alpha", type=float, default=0.25, help="EMA alpha for plotted curves.")
    return parser.parse_args()


def _make_math_sample(rng: random.Random) -> dict[str, str]:
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


def build_dataset(train_size: int, seed: int) -> Dataset:
    rng = random.Random(seed)
    samples = [_make_math_sample(rng) for _ in range(train_size)]

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

    wrapped = []
    for item in samples:
        prompt = (
            f"{instruction}"
            f"{format_example}"
            f"题目: {item['prompt']}\n"
            "输出:"
        )
        wrapped.append({"prompt": prompt, "answer": item["answer"]})
    return Dataset.from_list(wrapped)


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        parts = []
        for x in completion:
            if isinstance(x, dict):
                parts.append(str(x.get("content", "")))
            else:
                parts.append(str(x))
        return "".join(parts)
    return str(completion)


def extract_answer(text: str) -> str | None:
    match = ANSWER_TAG_PATTERN.search(text)
    if match:
        return match.group(1)
    nums = re.findall(r"[-+]?\d+", text)
    return nums[-1] if nums else None


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

        err = abs(pred - target)
        rewards.append(max(-0.2, 1.0 - err / 10.0))
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
        if STRICT_FORMAT_PATTERN.match(text):
            score += 0.10
        rewards.append(score)
    return rewards


def compact_output_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = completion_to_text(completion).strip()
        rewards.append(0.03 if len(text) <= 120 else -0.03)
    return rewards


def parse_reward_weights(text: str, n: int) -> list[float]:
    weights = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(weights) != n:
        raise ValueError(f"reward_weights expects {n} values, got {len(weights)}: {text}")
    return weights


def build_grpo_config(args: argparse.Namespace, reward_weights: list[float]) -> GRPOConfig:
    if args.generation_batch_size % args.num_generations != 0:
        raise ValueError(
            "generation_batch_size must be divisible by num_generations. "
            f"Got {args.generation_batch_size} and {args.num_generations}."
        )

    desired = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "generation_batch_size": args.generation_batch_size,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 1,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "reward_weights": reward_weights,
        "scale_rewards": "group",
        "report_to": "none",
    }

    supported = set(inspect.signature(GRPOConfig.__init__).parameters)
    filtered = {k: v for k, v in desired.items() if k in supported}
    return GRPOConfig(**filtered)


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_series(rows: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for row in rows:
        if row.get(key) is None:
            continue
        xs.append(int(row["step"]))
        ys.append(float(row[key]))
    return xs, ys


def _ema(values: list[float], alpha: float) -> list[float]:
    out = []
    prev = None
    for v in values:
        prev = v if prev is None else alpha * v + (1.0 - alpha) * prev
        out.append(prev)
    return out


def export_training_metrics(
    trainer: GRPOTrainer,
    output_dir: str,
    train_metrics: dict[str, Any],
    ema_alpha: float,
) -> None:
    out_dir = Path(output_dir)
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    log_history = trainer.state.log_history
    with (metrics_dir / "log_history.json").open("w", encoding="utf-8") as f:
        json.dump(log_history, f, ensure_ascii=False, indent=2)

    with (metrics_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(train_metrics, f, ensure_ascii=False, indent=2)

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

    rows = []
    for item in log_history:
        if "step" not in item:
            continue
        row = {}
        for key in keys:
            if key in item:
                row[key] = int(item[key]) if key == "step" else _to_float(item[key])
            else:
                row[key] = None
        rows.append(row)

    with (metrics_dir / "training_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable, skip plot generation: {exc}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    x, y = _extract_series(rows, "loss")
    axes[0, 0].plot(x, y, marker="o", alpha=0.45, label="raw")
    if y:
        axes[0, 0].plot(x, _ema(y, ema_alpha), linewidth=2.0, label=f"ema({ema_alpha})")
    axes[0, 0].set_title("GRPO Loss")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    x, y = _extract_series(rows, "reward")
    axes[0, 1].plot(x, y, marker="o", alpha=0.45, label="raw")
    if y:
        axes[0, 1].plot(x, _ema(y, ema_alpha), linewidth=2.0, label=f"ema({ema_alpha})")
    axes[0, 1].set_title("Total Reward")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].set_ylabel("reward")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    x1, y1 = _extract_series(rows, "rewards/correctness_reward/mean")
    x2, y2 = _extract_series(rows, "rewards/distance_reward/mean")
    x3, y3 = _extract_series(rows, "rewards/format_reward/mean")
    x4, y4 = _extract_series(rows, "rewards/compact_output_reward/mean")
    axes[1, 0].plot(x1, y1, marker="o", label="correctness")
    axes[1, 0].plot(x2, y2, marker="o", label="distance")
    axes[1, 0].plot(x3, y3, marker="o", label="format")
    axes[1, 0].plot(x4, y4, marker="o", label="compact")
    axes[1, 0].set_title("Reward Components")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].set_ylabel("reward")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    x, y = _extract_series(rows, "learning_rate")
    axes[1, 1].plot(x, y, marker="o")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].set_ylabel("lr")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(metrics_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_dataset = build_dataset(train_size=args.train_size, seed=args.seed)
    reward_funcs = [correctness_reward, distance_reward, format_reward, compact_output_reward]
    reward_weights = parse_reward_weights(args.reward_weights, len(reward_funcs))
    grpo_args = build_grpo_config(args, reward_weights)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    init_params = inspect.signature(GRPOTrainer.__init__).parameters
    trainer_kwargs = {
        "model": args.model_name,
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
        raise RuntimeError("Cannot find reward function argument in this TRL version.")

    trainer = GRPOTrainer(**trainer_kwargs)
    train_output = trainer.train()
    trainer.save_model(grpo_args.output_dir)
    export_training_metrics(trainer, grpo_args.output_dir, train_output.metrics, args.ema_alpha)

    print(f"GRPO demo done. Model saved to: {grpo_args.output_dir}")
    print(f"Metrics exported to: {Path(grpo_args.output_dir) / 'metrics'}")
    print(f"Train summary: {train_output.metrics}")


if __name__ == "__main__":
    main()
