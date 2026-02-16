#!/usr/bin/env python3
"""
根据模块运行结果自动生成面试口述稿。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run as interview_run  # noqa: E402


NOTES: dict[str, dict[str, str]] = {
    "sft": {
        "principle": "用监督数据做 next-token 预测，优化交叉熵损失，让模型先学会稳定跟随指令。",
        "compare": "相比 DPO，SFT 不利用 chosen/rejected 相对偏好信息。",
        "engineering": "优先关注 loss 趋势、学习率调度与梯度范数稳定性。",
    },
    "dpo": {
        "principle": "直接优化偏好对中 chosen 相对 rejected 的概率，不显式训练奖励模型。",
        "compare": "相比 PPO，DPO 不做在线 rollout，工程链路更短。",
        "engineering": "关注偏好数据质量，坏标注会直接把优化方向拉偏。",
    },
    "grpo": {
        "principle": "同一 prompt 采样多候选并做组内相对奖励比较，再更新策略。",
        "compare": "相比 PPO，GRPO 更强调组内相对信号，减少对绝对分值尺度的敏感。",
        "engineering": "奖励函数设计决定上限，建议拆分 correctness/format 等分量可视化。",
    },
    "ppo": {
        "principle": "通过策略梯度提高奖励，同时用 clip/KL 约束避免策略单步漂移过大。",
        "compare": "相比 SFT，PPO 优化目标是奖励而不是监督标签拟合。",
        "engineering": "重点看 reward 与 loss 是否同步改善，以及是否出现 KL 失控。",
    },
    "rlhf": {
        "principle": "RLHF 是流程：SFT 打底 -> 奖励建模 -> PPO 优化策略。",
        "compare": "相比 DPO，RLHF 通常显式包含奖励模型与在线强化学习阶段。",
        "engineering": "警惕 reward hacking，必须用独立样本验证真实回答质量。",
    },
    "mdp": {
        "principle": "MDP 用状态、动作、转移概率与奖励定义决策过程，值迭代通过 Bellman 备份逼近最优值函数。",
        "compare": "相比 TD Learning，MDP 值迭代通常假设已知环境模型（可直接做规划）。",
        "engineering": "看 final_delta 是否足够小、迭代轮数是否合理，避免过早停止导致策略不稳定。",
    },
    "td_learning": {
        "principle": "TD Learning 用一步 bootstrap 估计更新价值，不需要完整回报就能在线学习。",
        "compare": "相比 Monte Carlo，TD 更新方差更低、收敛更快，但会引入偏差。",
        "engineering": "重点看学习率与探索策略，二者直接决定收敛速度和稳定性。",
    },
    "gae": {
        "principle": "GAE 通过加权多步 TD 残差构造优势函数，在偏差和方差之间做可控折中。",
        "compare": "相比单步优势估计，GAE 通常更平滑，策略梯度训练更稳定。",
        "engineering": "lambda 过大可能高方差，lambda 过小可能高偏差，建议配合曲线一起调参。",
    },
    "advantage": {
        "principle": "Advantage 衡量动作相对状态基线的增益，核心是减少策略梯度估计方差。",
        "compare": "相比直接用 return，优势函数更容易让优化步骤稳定。",
        "engineering": "检查优势归一化与基线估计是否正确，否则会出现训练震荡。",
    },
    "blip2": {
        "principle": "通过 Q-Former 把视觉特征压缩成语言模型可用表示，实现图文对齐。",
        "compare": "相比 LLaVA，BLIP2 更强调桥接模块（Q-Former）设计。",
        "engineering": "先用 dry-run 打通链路，再上真实图像避免依赖与显存坑。",
    },
    "llava": {
        "principle": "把图像编码后投影到 LLM 输入空间，配合视觉指令数据完成图文对话能力。",
        "compare": "相比 BLIP2，LLaVA 更偏 projector + 指令微调范式。",
        "engineering": "提示词模板和图像预处理会显著影响推理表现。",
    },
    "flamingo": {
        "principle": "在语言模型层中插入跨注意力层，周期性注入视觉条件。",
        "compare": "相比 LLaVA，Flamingo 的视觉信息融合更深层、持续性更强。",
        "engineering": "跨注意力计算开销高，部署时要注意吞吐与显存折中。",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate interview-ready oral brief from run artifacts.")
    parser.add_argument("--module", required=True, choices=sorted(interview_run.MODULES.keys()))
    parser.add_argument("--summary", default="", help="可选：指定 summary/result JSON 路径。")
    parser.add_argument("--save", default="", help="可选：把生成内容保存到文件。")
    return parser.parse_args()


def module_output_dir(module: str) -> Path | None:
    spec = interview_run.MODULES[module]
    toy_args = list(spec.toy_args)
    if "--output-dir" not in toy_args:
        return None
    idx = toy_args.index("--output-dir")
    if idx + 1 >= len(toy_args):
        return None
    output_arg = toy_args[idx + 1]
    script_path = ROOT / spec.script
    module_dir = script_path.parent.parent
    return (module_dir / output_arg).resolve()


def discover_summary(module: str) -> tuple[Path | None, dict[str, Any]]:
    out_dir = module_output_dir(module)
    if out_dir is None or not out_dir.exists():
        return None, {}

    candidates: list[Path] = []
    for name in ("summary.json", "train_summary.json", "result.json"):
        candidates.extend(out_dir.rglob(name))
    if not candidates:
        return None, {}

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    path = candidates[0]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    return path, data


def load_summary(module: str, summary_arg: str) -> tuple[Path | None, dict[str, Any]]:
    if summary_arg.strip():
        path = Path(summary_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"summary file not found: {path}")
        return path, json.loads(path.read_text(encoding="utf-8"))
    return discover_summary(module)


def build_metric_interpretation(data: dict[str, Any]) -> str:
    if not data:
        return "未找到可解析的 summary/result JSON。建议先执行对应模块的 `--toy` 运行。"

    parts: list[str] = []
    priority_keys = [
        "final_loss",
        "best_eval_loss",
        "train_loss",
        "final_reward",
        "best_reward",
        "iterations",
        "final_delta",
        "total_steps",
        "num_states",
    ]
    for key in priority_keys:
        if key in data:
            parts.append(f"{key}={data.get(key)}")

    for key, value in data.items():
        if key in priority_keys:
            continue
        if isinstance(value, (int, float, str, bool)):
            parts.append(f"{key}={value}")
        if len(parts) >= 8:
            break

    if "response" in data:
        text = str(data.get("response", ""))
        parts.append(f"sample_response={text[:80]}")

    if not parts:
        keys = ", ".join(sorted(list(data.keys()))[:8])
        return f"已读取结果，但字段风格非标准。可用字段示例：{keys}"
    return "，".join(parts)


def build_brief(module: str, summary_path: Path | None, data: dict[str, Any]) -> str:
    spec = interview_run.MODULES[module]
    note = NOTES.get(
        module,
        {
            "principle": "该模块用于演示该方法的最小可运行流程。",
            "compare": "请重点和相邻方法对比：目标函数、数据依赖、在线/离线训练方式。",
            "engineering": "请结合本次输出曲线解释训练是否稳定，以及下一步调参方向。",
        },
    )

    metric_line = build_metric_interpretation(data)
    source_line = str(summary_path) if summary_path else "自动发现失败（建议先运行 toy）"

    return "\n".join(
        [
            f"# {module} 面试口述稿",
            "",
            "## 30秒定义",
            f"{module} 在本项目中的定位是：{spec.summary}。",
            "",
            "## 60秒原理",
            note["principle"],
            "",
            "## 60秒结果解释",
            f"本次读取结果文件：{source_line}",
            f"关键指标：{metric_line}",
            "解释方式：先说趋势（是否收敛/是否稳定），再说原因（目标函数与奖励设计），最后说改进方向。",
            "",
            "## 30秒工程经验",
            note["engineering"],
            "",
            "## 对比回答（加分项）",
            note["compare"],
            "",
            "## 面试追问自测",
            "1. 这个方法最怕的数据问题是什么？",
            "2. 如果指标变好但样例变差，你先查哪三个点？",
            "3. 和上一个方法相比，你为什么要切换到它？",
        ]
    )


def main() -> None:
    args = parse_args()
    summary_path, data = load_summary(args.module, args.summary)
    text = build_brief(args.module, summary_path, data)

    if args.save.strip():
        out_path = Path(args.save).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"saved: {out_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
