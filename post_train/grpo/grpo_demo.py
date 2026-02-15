#!/usr/bin/env python3
"""
GRPO minimal demo (TRL) for /post_train/grpo.

Usage:
    python grpo_demo.py
"""

import inspect
import json
import csv
import re
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


MODEL_NAME = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "qwen3_grpo_out"


def build_dataset() -> Dataset:
    # 演示用的极小数据集：prompt + 标准答案。
    samples = [
        {"prompt": "计算: 3 + 5 = ? 请用标签输出。", "answer": "8"},
        {"prompt": "计算: 12 - 7 = ? 请用标签输出。", "answer": "5"},
        {"prompt": "计算: 6 * 4 = ? 请用标签输出。", "answer": "24"},
        {"prompt": "计算: 15 / 3 = ? 请用标签输出。", "answer": "5"},
    ]

    # 将任务包装为统一输出模板，方便后续用格式奖励约束。
    wrapped = []
    for item in samples:
        prompt = (
            "你是一个数学助手。请严格使用以下格式回答:\n"
            "<reasoning>...</reasoning>\n"
            "<answer>最终数字答案</answer>\n\n"
            f"题目: {item['prompt']}"
        )
        wrapped.append({"prompt": prompt, "answer": item["answer"]})
    return Dataset.from_list(wrapped)


def completion_to_text(completion) -> str:
    # TRL 不同版本/后端可能返回 str、dict、list，这里统一转成纯文本。
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


def extract_answer(text: str):
    # 优先从 <answer> 标签提取；若无标签则回退到最后一个整数。
    tag_match = re.search(r"<answer>\s*([-+]?\d+)\s*</answer>", text)
    if tag_match:
        return tag_match.group(1)
    numbers = re.findall(r"[-+]?\d+", text)
    return numbers[-1] if numbers else None


def correctness_reward(completions, answer, **kwargs):
    # 正确性奖励：答案匹配给 1.0，否则 0.0。
    rewards = []
    for completion, gold in zip(completions, answer):
        pred = extract_answer(completion_to_text(completion))
        rewards.append(1.0 if pred == str(gold) else 0.0)
    return rewards


def format_reward(completions, **kwargs):
    # 格式奖励：满足指定 XML 风格输出时给一个小奖励，鼓励结构化回答。
    pattern = re.compile(
        r"^\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$", re.S
    )
    return [0.2 if pattern.match(completion_to_text(c)) else 0.0 for c in completions]


def build_grpo_config() -> GRPOConfig:
    # 这里放“期望参数集合”，稍后会根据当前 TRL 版本自动过滤。
    desired = {
        "output_dir": OUTPUT_DIR,
        "learning_rate": 1e-6,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        # GRPO 需要 >=2 个 generation 才能计算优势；并保持 batch 可整除。
        "num_generations": 2,
        "generation_batch_size": 2,
        "max_prompt_length": 256,
        "max_completion_length": 96,
        "num_train_epochs": 1,
        "logging_steps": 1,
        "save_steps": 20,
        "report_to": "none",
    }
    # 兼容不同版本 TRL：只传构造函数支持的字段，避免参数名变化导致报错。
    supported = set(inspect.signature(GRPOConfig.__init__).parameters)
    filtered = {k: v for k, v in desired.items() if k in supported}
    return GRPOConfig(**filtered)


def _to_float(value):
    # 统一把训练日志中的字符串/数字转换为 float，便于画图与落盘。
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def export_training_metrics(trainer: GRPOTrainer, output_dir: str, train_metrics: dict) -> None:
    # 导出训练日志与关键指标，方便后续分析。
    out_dir = Path(output_dir)
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    log_history = trainer.state.log_history

    # 1) 原始日志完整保存，便于回溯所有键值。
    with (metrics_dir / "log_history.json").open("w", encoding="utf-8") as f:
        json.dump(log_history, f, ensure_ascii=False, indent=2)

    # 2) 训练最终汇总指标。
    with (metrics_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(train_metrics, f, ensure_ascii=False, indent=2)

    # 3) 关键曲线数据（CSV）。
    keys = [
        "step",
        "epoch",
        "loss",
        "learning_rate",
        "grad_norm",
        "reward",
        "reward_std",
        "rewards/correctness_reward/mean",
        "rewards/format_reward/mean",
        "entropy",
    ]
    rows = []
    for item in log_history:
        if "step" not in item:
            continue
        row = {}
        for k in keys:
            if k in item:
                v = item[k]
                row[k] = _to_float(v) if k != "step" else int(v)
            else:
                row[k] = None
        rows.append(row)

    with (metrics_dir / "training_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    # 4) 尝试生成可视化图（loss/reward/lr）。
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable, skip plot generation: {exc}")
        return

    steps = [r["step"] for r in rows if r["step"] is not None]
    loss = [r["loss"] for r in rows]
    reward = [r["reward"] for r in rows]
    corr_reward = [r["rewards/correctness_reward/mean"] for r in rows]
    fmt_reward = [r["rewards/format_reward/mean"] for r in rows]
    lr = [r["learning_rate"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(steps, loss, marker="o")
    axes[0, 0].set_title("GRPO Loss")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, reward, marker="o")
    axes[0, 1].set_title("Total Reward")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].set_ylabel("reward")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(steps, corr_reward, marker="o", label="correctness")
    axes[1, 0].plot(steps, fmt_reward, marker="o", label="format")
    axes[1, 0].set_title("Reward Components")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].set_ylabel("reward")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(steps, lr, marker="o")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].set_ylabel("lr")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(metrics_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def main():
    # 1) 构造数据和训练配置。
    train_dataset = build_dataset()
    grpo_args = build_grpo_config()

    # 2) 准备 tokenizer；无 pad_token 时回退到 eos_token。
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) 按 GRPOTrainer 当前签名组装参数，做版本兼容。
    init_params = inspect.signature(GRPOTrainer.__init__).parameters
    trainer_kwargs = {
        "model": MODEL_NAME,
        "args": grpo_args,
        "train_dataset": train_dataset,
    }
    if "processing_class" in init_params:
        trainer_kwargs["processing_class"] = tokenizer

    if "reward_funcs" in init_params:
        trainer_kwargs["reward_funcs"] = [correctness_reward, format_reward]
    elif "reward_func" in init_params:
        trainer_kwargs["reward_func"] = correctness_reward
    else:
        raise RuntimeError("Cannot find reward function argument in this TRL version.")

    # 4) 启动训练并保存模型到 output_dir。
    trainer = GRPOTrainer(**trainer_kwargs)
    train_output = trainer.train()
    trainer.save_model(grpo_args.output_dir)
    export_training_metrics(trainer, grpo_args.output_dir, train_output.metrics)
    print(f"GRPO demo done. Model saved to: {grpo_args.output_dir}")
    print(f"Metrics exported to: {Path(grpo_args.output_dir) / 'metrics'}")


if __name__ == "__main__":
    main()
