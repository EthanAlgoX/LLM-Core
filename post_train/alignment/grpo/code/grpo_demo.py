#!/usr/bin/env python3
"""
GRPO 最简训练脚本（学习版）。

一、GRPO 原理（面向实现）
1) 对每个 prompt 采样多条候选回答（num_generations）。
2) 用多个奖励函数对候选回答打分（如正确性、格式、简洁性）。
3) 将奖励按组做标准化（scale_rewards="group"），降低样本间方差。
4) 用标准化后的相对优势信号更新策略，使高奖励回答概率上升。

二、代码框架（从入口到结果）
1) `parse_args`：读取训练与可视化参数。
2) `build_dataset`：构造带格式约束的合成数学数据。
3) `build_grpo_config`：组装 GRPOConfig（含 reward 权重与采样超参）。
4) `build_trainer`：兼容不同 TRL 版本创建 GRPOTrainer。
5) `trainer.train()`：执行训练并保存模型。
6) `export_training_metrics`：导出 JSON/CSV/曲线图/summary 便于学习分析。
"""

from __future__ import annotations

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
DEFAULT_OUTPUT_DIR = "output"


def parse_args() -> argparse.Namespace:
    """解析命令行参数，返回 GRPO 训练与可视化配置。"""
    parser = argparse.ArgumentParser(description="Run GRPO and always export visualization artifacts.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=24, help="合成训练样本数量。")

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
        help="奖励权重，顺序为 [correctness,distance,format,compact]。",
    )
    parser.add_argument("--ema-alpha", type=float, default=0.25, help="曲线平滑 EMA 系数。")
    return parser.parse_args()


def make_math_sample(rng: random.Random) -> dict[str, str]:
    """随机生成一道四则运算样本，返回 prompt 与标准答案。"""
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
    """构造带模板约束的训练集，要求模型按标签输出推理与答案。"""
    rng = random.Random(seed)
    samples = [make_math_sample(rng) for _ in range(train_size)]

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
        prompt = f"{instruction}{format_example}题目: {item['prompt']}\n输出:"
        wrapped.append({"prompt": prompt, "answer": item["answer"]})
    return Dataset.from_list(wrapped)


def completion_to_text(completion: Any) -> str:
    """将不同结构的 completion 统一转为纯文本。"""
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
    """优先从 <answer> 标签抽取答案；若无标签则回退为最后一个整数。"""
    match = ANSWER_TAG_PATTERN.search(text)
    if match:
        return match.group(1)
    nums = re.findall(r"[-+]?\d+", text)
    return nums[-1] if nums else None


def correctness_reward(completions, answer, **kwargs):
    """正确性奖励：答案完全一致记 1，否则记 0。"""
    rewards = []
    for completion, gold in zip(completions, answer):
        pred = extract_answer(completion_to_text(completion))
        rewards.append(1.0 if pred == str(gold) else 0.0)
    return rewards


def distance_reward(completions, answer, **kwargs):
    """距离奖励：按预测值与真值距离衰减，最小惩罚为 -0.2。"""
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
    """格式奖励：鼓励输出包含 reasoning/answer 标签并满足严格结构。"""
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
    """简洁奖励：鼓励短输出，避免无关冗长文本。"""
    rewards = []
    for completion in completions:
        text = completion_to_text(completion).strip()
        rewards.append(0.03 if len(text) <= 120 else -0.03)
    return rewards


def parse_reward_weights(text: str, n: int) -> list[float]:
    """解析奖励权重字符串，并校验数量与奖励函数个数一致。"""
    weights = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(weights) != n:
        raise ValueError(f"reward_weights 需要 {n} 个值，实际得到 {len(weights)} 个: {text}")
    return weights


def resolve_output_dir(base_dir: Path, output_dir: str) -> Path:
    """解析输出目录：相对路径按 grpo 目录解析，绝对路径原样使用。"""
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


def build_grpo_config(args: argparse.Namespace, reward_weights: list[float], checkpoints_dir: Path) -> GRPOConfig:
    """构造 GRPO 配置，并按当前 TRL 版本过滤不支持字段。"""
    if args.generation_batch_size % args.num_generations != 0:
        raise ValueError(
            "generation_batch_size 必须能被 num_generations 整除。"
            f"当前为 {args.generation_batch_size} 和 {args.num_generations}。"
        )

    desired = {
        "output_dir": str(checkpoints_dir),  # 训练输出目录（仅 checkpoint）。
        "overwrite_output_dir": True,  # 允许复用统一 checkpoint 目录。
        "learning_rate": args.learning_rate,  # 学习率。
        "per_device_train_batch_size": args.per_device_train_batch_size,  # 单设备 batch。
        "gradient_accumulation_steps": args.gradient_accumulation_steps,  # 梯度累积。
        "num_generations": args.num_generations,  # 每个 prompt 生成候选数量。
        "generation_batch_size": args.generation_batch_size,  # 生成阶段 batch。
        "max_prompt_length": args.max_prompt_length,  # prompt 最大 token。
        "max_completion_length": args.max_completion_length,  # completion 最大 token。
        "num_train_epochs": args.num_train_epochs,  # 训练 epoch。
        "logging_steps": args.logging_steps,  # 日志间隔。
        "save_steps": args.save_steps,  # checkpoint 保存间隔。
        "lr_scheduler_type": "cosine",  # 学习率调度策略。
        "warmup_steps": 1,  # 预热步数。
        "temperature": args.temperature,  # 采样温度。
        "top_p": args.top_p,  # nucleus 采样阈值。
        "reward_weights": reward_weights,  # 多奖励权重。
        "scale_rewards": "group",  # 按组标准化奖励，减小方差。
        "report_to": "none",  # 关闭外部平台上报。
    }

    supported = set(inspect.signature(GRPOConfig.__init__).parameters)
    filtered = {k: v for k, v in desired.items() if k in supported}
    return GRPOConfig(**filtered)


def build_trainer(
    args: argparse.Namespace, grpo_args: GRPOConfig, train_dataset: Dataset, reward_funcs: list[Any]
) -> GRPOTrainer:
    """按 TRL 版本差异构造 GRPOTrainer（兼容 tokenizer/processing_class 参数）。"""
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
        raise RuntimeError("当前 TRL 版本未找到奖励函数参数（reward_funcs/reward_func）。")

    return GRPOTrainer(**trainer_kwargs)


def _to_float(value: Any) -> float | None:
    """将日志字段安全转换为 float。"""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_series(rows: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    """从表格行中提取指定指标的 step 序列与数值序列。"""
    xs, ys = [], []
    for row in rows:
        if row.get(key) is None:
            continue
        xs.append(int(row["step"]))
        ys.append(float(row[key]))
    return xs, ys


def _ema(values: list[float], alpha: float) -> list[float]:
    """计算指数滑动平均，用于平滑震荡曲线。"""
    out = []
    prev = None
    for value in values:
        prev = value if prev is None else alpha * value + (1.0 - alpha) * prev
        out.append(prev)
    return out


def export_training_metrics(
    trainer: GRPOTrainer,
    output_dir: str,
    train_metrics: dict[str, Any],
    ema_alpha: float,
) -> Path:
    """导出 GRPO 训练日志、CSV、可视化与摘要，返回 metrics 目录。"""
    out_dir = Path(output_dir)
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    log_history = trainer.state.log_history
    (metrics_dir / "log_history.json").write_text(
        json.dumps(log_history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (metrics_dir / "train_summary.json").write_text(
        json.dumps(train_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

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
        raise RuntimeError(f"`matplotlib` 不可用，无法绘制训练曲线：{exc}") from exc

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

    summary = {
        "total_steps": len(rows),
        "final_step": rows[-1]["step"] if rows else None,
        "final_loss": next((r["loss"] for r in reversed(rows) if r["loss"] is not None), None),
        "final_reward": next((r["reward"] for r in reversed(rows) if r["reward"] is not None), None),
        "best_reward": max((r["reward"] for r in rows if r["reward"] is not None), default=None),
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics_dir


def save_run_config(output_dir: Path, args: argparse.Namespace, reward_weights: list[float]) -> None:
    """保存本次训练配置快照，便于复现。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = vars(args).copy()
    payload["reward_weights"] = reward_weights
    (output_dir / "run_config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """主流程：构建数据 -> 构建训练器 -> 训练 -> 导出可视化结果。"""
    args = parse_args()
    code_dir = Path(__file__).resolve().parent
    module_dir = code_dir.parent
    layout = ensure_layout_dirs(module_dir=module_dir, output_arg=args.output_dir)
    args.output_dir = str(layout["output"])
    set_seed(args.seed)

    train_dataset = build_dataset(train_size=args.train_size, seed=args.seed)
    reward_funcs = [correctness_reward, distance_reward, format_reward, compact_output_reward]
    reward_weights = parse_reward_weights(args.reward_weights, len(reward_funcs))
    grpo_args = build_grpo_config(args, reward_weights, checkpoints_dir=layout["checkpoints"])
    save_run_config(layout["output"], args, reward_weights)

    trainer = build_trainer(args=args, grpo_args=grpo_args, train_dataset=train_dataset, reward_funcs=reward_funcs)
    train_output = trainer.train()
    trainer.save_model(str(layout["models"]))
    metrics_dir = export_training_metrics(trainer, str(layout["output"]), train_output.metrics, args.ema_alpha)

    print(f"GRPO 训练完成，模型目录: {layout['models']}")
    print(f"Checkpoint 目录: {layout['checkpoints']}")
    print(f"指标目录: {metrics_dir}")
    print(f"训练摘要: {train_output.metrics}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise
