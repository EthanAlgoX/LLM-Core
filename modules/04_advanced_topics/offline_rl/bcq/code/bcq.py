#!/usr/bin/env python3
"""
BCQ（Batch-Constrained Q-learning）最小可运行示例：LineWorld Offline RL。

一、BCQ 原理（面向实现）
1) BCQ 是离线强化学习方法，只用固定数据集训练策略。
2) 核心思想：动作选择要受“行为策略分布”约束，避免分布外动作。
3) 离散版 BCQ（dBCQ）常见做法：
   - 训练 Q 网络估计 Q(s,a)
   - 训练行为模型 g(s) 估计数据中动作分布 p(a|s)
   - 选择动作时仅保留高概率动作集合，再从中选 Q 最大动作
4) 这样可减小离线数据覆盖不足导致的过估计。

新人阅读顺序（建议）
1) 先看 `build_default_args`：明确可调参数和默认值。
2) 再看 `main`：把握执行主链路（准备 -> 训练/推理 -> 导出）。
3) 最后看可视化导出函数（如 `export_artifacts`）理解输出文件。

二、代码框架（从入口到结果）
1) `build_default_args`：读取环境、数据和训练参数。
2) `collect_offline_dataset`：构建离线行为数据集。
3) `train_bcq`：训练 Q 网络与行为模型。
4) `evaluate_policy`：评估 BCQ 约束下的贪心策略。
5) `export_artifacts`：导出 CSV/JSON/曲线图/summary。
6) `main`：串联完整流程。

用法：
  python code/bcq.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_OUTPUT_DIR = "output"
ACTION_TEXT = {0: "L", 1: "R"}


def build_default_args() -> argparse.Namespace:
    """解析命令行参数，返回 BCQ 配置。"""
    parser = argparse.ArgumentParser(description="Run BCQ demo training and export visualization artifacts.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)

    # 环境参数。
    parser.add_argument("--line-size", type=int, default=9)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--step-penalty", type=float, default=-0.01)
    parser.add_argument("--goal-reward", type=float, default=1.0)
    parser.add_argument("--fail-reward", type=float, default=-1.0)
    parser.add_argument("--slip-prob", type=float, default=0.05)

    # 离线数据参数。
    parser.add_argument("--dataset-episodes", type=int, default=500)
    parser.add_argument("--behavior-epsilon", type=float, default=0.35)
    parser.add_argument("--behavior-right-prob", type=float, default=0.65)

    # BCQ 训练参数。
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--q-learning-rate", type=float, default=3e-3)
    parser.add_argument("--imit-learning-rate", type=float, default=3e-3)
    parser.add_argument("--bcq-threshold", type=float, default=0.3, help="动作过滤阈值（相对最大概率）。")
    parser.add_argument("--imit-logit-reg", type=float, default=1e-2)
    parser.add_argument("--target-update-tau", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=200)
    return parser.parse_known_args([])[0]


def detect_device() -> torch.device:
    """选择设备：CUDA > MPS > CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """设置随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_output_dir(base_dir: Path, output_dir: str) -> Path:
    """解析输出目录：相对路径按 bcq 目录解析。"""
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


@dataclass
class StepResult:
    """环境一步交互结果。"""

    state: int
    reward: float
    done: bool
    success: bool


class LineWorld:
    """简单一维环境。"""

    def __init__(self, args: argparse.Namespace) -> None:
        if args.line_size < 3:
            raise ValueError("--line-size must be >= 3")
        self.n = args.line_size
        self.max_steps = args.max_steps
        self.step_penalty = args.step_penalty
        self.goal_reward = args.goal_reward
        self.fail_reward = args.fail_reward
        self.slip_prob = args.slip_prob
        self.reset()

    def reset(self) -> int:
        self.s = self.n // 2
        self.t = 0
        return self.s

    def _terminal(self, s: int) -> bool:
        return s == 0 or s == self.n - 1

    def step(self, action: int) -> StepResult:
        if action not in (0, 1):
            raise ValueError(f"invalid action: {action}")
        if self._terminal(self.s):
            return StepResult(self.s, 0.0, True, self.s == self.n - 1)

        self.t += 1
        if random.random() < self.slip_prob:
            action = 1 - action

        self.s += -1 if action == 0 else 1
        self.s = max(0, min(self.n - 1, self.s))

        if self.s == self.n - 1:
            return StepResult(self.s, self.goal_reward, True, True)
        if self.s == 0:
            return StepResult(self.s, self.fail_reward, True, False)
        if self.t >= self.max_steps:
            return StepResult(self.s, self.step_penalty, True, False)
        return StepResult(self.s, self.step_penalty, False, False)

    def state_to_feature(self, s: int) -> float:
        """离散状态映射为归一化标量特征。"""
        return (float(s) / float(self.n - 1)) * 2.0 - 1.0


class QNet(nn.Module):
    """Q 网络：输入状态，输出两个动作 Q 值。"""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImitationNet(nn.Module):
    """行为模型：估计数据中动作分布 p(a|s)。"""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def collect_offline_dataset(env: LineWorld, args: argparse.Namespace, data_dir: Path) -> dict[str, np.ndarray]:
    """用行为策略采样离线数据集。"""
    states, actions, rewards, next_states, dones = [], [], [], [], []
    success_count = 0
    action_count = {0: 0, 1: 0}

    for _ in range(args.dataset_episodes):
        s = env.reset()
        done = False
        while not done:
            if random.random() < args.behavior_epsilon:
                a = random.randint(0, 1)
            else:
                a = 1 if random.random() < args.behavior_right_prob else 0

            step = env.step(a)
            states.append(env.state_to_feature(s))
            actions.append(a)
            rewards.append(step.reward)
            next_states.append(env.state_to_feature(step.state))
            dones.append(1.0 if step.done else 0.0)
            action_count[a] += 1

            if step.done and step.success:
                success_count += 1
            s = step.state
            done = step.done

    ds = {
        "states": np.array(states, dtype=np.float32).reshape(-1, 1),
        "actions": np.array(actions, dtype=np.int64),
        "rewards": np.array(rewards, dtype=np.float32),
        "next_states": np.array(next_states, dtype=np.float32).reshape(-1, 1),
        "dones": np.array(dones, dtype=np.float32),
    }
    np.savez(data_dir / "offline_dataset.npz", **ds)

    n_trans = int(ds["states"].shape[0])
    stats = {
        "num_transitions": n_trans,
        "dataset_episodes": int(args.dataset_episodes),
        "behavior_success_rate": float(success_count / max(args.dataset_episodes, 1)),
        "action_left_ratio": float(action_count[0] / max(n_trans, 1)),
        "action_right_ratio": float(action_count[1] / max(n_trans, 1)),
        "reward_mean": float(np.mean(ds["rewards"])) if n_trans > 0 else 0.0,
        "reward_std": float(np.std(ds["rewards"])) if n_trans > 0 else 0.0,
    }
    (data_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(
        f"Offline dataset ready: transitions={stats['num_transitions']}, "
        f"success_rate={stats['behavior_success_rate']:.3f}"
    )
    return ds


def sample_batch(dataset: dict[str, np.ndarray], batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """从离线数据集中采样一个 batch。"""
    n = dataset["states"].shape[0]
    idx = np.random.randint(0, n, size=batch_size)
    return {
        "states": torch.tensor(dataset["states"][idx], dtype=torch.float32, device=device),
        "actions": torch.tensor(dataset["actions"][idx], dtype=torch.long, device=device),
        "rewards": torch.tensor(dataset["rewards"][idx], dtype=torch.float32, device=device),
        "next_states": torch.tensor(dataset["next_states"][idx], dtype=torch.float32, device=device),
        "dones": torch.tensor(dataset["dones"][idx], dtype=torch.float32, device=device),
    }


def _select_actions_from_q_and_bcq_mask(
    q_values: torch.Tensor,
    bc_logits: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """根据行为模型阈值掩码选择动作，返回动作索引和可行动作比例。"""
    probs = torch.softmax(bc_logits, dim=1)
    max_prob = probs.max(dim=1, keepdim=True).values
    mask = (probs / (max_prob + 1e-8) > threshold).float()

    masked_q = q_values + (mask - 1.0) * 1e9
    act = torch.argmax(masked_q, dim=1)
    fallback = torch.argmax(probs, dim=1)
    no_valid = mask.sum(dim=1) < 0.5
    act = torch.where(no_valid, fallback, act)
    valid_ratio = float(mask.mean().item())
    return act, torch.tensor(valid_ratio, dtype=torch.float32, device=q_values.device)


@torch.no_grad()
def evaluate_policy(
    env: LineWorld,
    qnet: QNet,
    imit_net: ImitationNet,
    threshold: float,
    episodes: int,
    device: torch.device,
) -> tuple[float, float, float]:
    """评估 BCQ 贪心策略。"""
    qnet.eval()
    imit_net.eval()
    returns = []
    succ = 0
    for _ in range(episodes):
        s = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            st = torch.tensor([[env.state_to_feature(s)]], dtype=torch.float32, device=device)
            q = qnet(st)
            bc_logits = imit_net(st)
            action, _ = _select_actions_from_q_and_bcq_mask(q, bc_logits, threshold)
            a = int(action.item())
            step = env.step(a)
            ep_ret += step.reward
            if step.done and step.success:
                succ += 1
            s = step.state
            done = step.done
        returns.append(ep_ret)
    return float(np.mean(returns)), float(np.std(returns)), float(succ / max(episodes, 1))


def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """Polyak 平滑更新。"""
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)


def save_checkpoint(
    checkpoints_dir: Path,
    step: int,
    qnet: QNet,
    target_qnet: QNet,
    imit_net: ImitationNet,
    target_imit_net: ImitationNet,
    q_optimizer: torch.optim.Optimizer,
    imit_optimizer: torch.optim.Optimizer,
    log_item: dict[str, float],
) -> None:
    """保存训练 checkpoint。"""
    torch.save(
        {
            "step": step,
            "qnet_state_dict": qnet.state_dict(),
            "target_qnet_state_dict": target_qnet.state_dict(),
            "imit_net_state_dict": imit_net.state_dict(),
            "target_imit_net_state_dict": target_imit_net.state_dict(),
            "q_optimizer_state_dict": q_optimizer.state_dict(),
            "imit_optimizer_state_dict": imit_optimizer.state_dict(),
            "log": log_item,
        },
        checkpoints_dir / f"checkpoint-{step}.pt",
    )


def train_bcq(
    env: LineWorld,
    dataset: dict[str, np.ndarray],
    qnet: QNet,
    target_qnet: QNet,
    imit_net: ImitationNet,
    target_imit_net: ImitationNet,
    q_optimizer: torch.optim.Optimizer,
    imit_optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    checkpoints_dir: Path,
) -> list[dict[str, float]]:
    """执行 BCQ 训练并返回日志。"""
    logs: list[dict[str, float]] = []
    for step in range(1, args.updates + 1):
        batch = sample_batch(dataset, args.batch_size, device)

        # Q 网络更新：使用 target 网络 + BCQ 动作过滤产生目标动作。
        with torch.no_grad():
            next_q = target_qnet(batch["next_states"])
            next_bc_logits = target_imit_net(batch["next_states"])
            next_action, valid_ratio = _select_actions_from_q_and_bcq_mask(
                next_q, next_bc_logits, args.bcq_threshold
            )
            next_q_selected = next_q.gather(1, next_action.unsqueeze(1)).squeeze(1)
            td_target = batch["rewards"] + args.gamma * (1.0 - batch["dones"]) * next_q_selected

        q = qnet(batch["states"])
        q_data = q.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        q_loss = F.mse_loss(q_data, td_target)

        q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(qnet.parameters(), args.max_grad_norm)
        q_optimizer.step()

        # 行为模型更新：监督学习拟合数据动作分布。
        bc_logits = imit_net(batch["states"])
        bc_loss = F.cross_entropy(bc_logits, batch["actions"])
        reg_loss = args.imit_logit_reg * torch.mean(bc_logits**2)
        imit_loss = bc_loss + reg_loss

        imit_optimizer.zero_grad(set_to_none=True)
        imit_loss.backward()
        torch.nn.utils.clip_grad_norm_(imit_net.parameters(), args.max_grad_norm)
        imit_optimizer.step()

        soft_update(target_qnet, qnet, args.target_update_tau)
        soft_update(target_imit_net, imit_net, args.target_update_tau)

        eval_return, eval_std, eval_success = np.nan, np.nan, np.nan
        if step % args.eval_every == 0 or step == 1 or step == args.updates:
            eval_return, eval_std, eval_success = evaluate_policy(
                env=env,
                qnet=qnet,
                imit_net=imit_net,
                threshold=args.bcq_threshold,
                episodes=args.eval_episodes,
                device=device,
            )

        log_item = {
            "step": float(step),
            "q_loss": float(q_loss.item()),
            "bc_loss": float(bc_loss.item()),
            "imit_reg_loss": float(reg_loss.item()),
            "imit_total_loss": float(imit_loss.item()),
            "q_data_mean": float(q_data.mean().item()),
            "valid_action_ratio": float(valid_ratio.item()),
            "eval_return": float(eval_return),
            "eval_return_std": float(eval_std),
            "eval_success_rate": float(eval_success),
            "q_learning_rate": float(q_optimizer.param_groups[0]["lr"]),
            "imit_learning_rate": float(imit_optimizer.param_groups[0]["lr"]),
        }
        logs.append(log_item)

        if step % args.log_every == 0 or step == 1 or step == args.updates:
            print(
                f"[Step {step:04d}] q_loss={log_item['q_loss']:.6f} "
                f"bc_loss={log_item['bc_loss']:.6f} "
                f"eval_return={log_item['eval_return']:.4f}"
            )
        if step % args.save_every == 0 or step == args.updates:
            save_checkpoint(
                checkpoints_dir,
                step,
                qnet,
                target_qnet,
                imit_net,
                target_imit_net,
                q_optimizer,
                imit_optimizer,
                log_item,
            )

    return logs


def export_artifacts(
    logs: list[dict[str, float]],
    dataset_stats: dict[str, Any],
    output_dir: Path,
    models_dir: Path,
    qnet: QNet,
    imit_net: ImitationNet,
    env: LineWorld,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    """导出日志、模型、策略和可视化。"""
    metrics_dir = output_dir / "bcq_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / "training_log.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    (metrics_dir / "dataset_stats.json").write_text(
        json.dumps(dataset_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fields = [
        "step",
        "q_loss",
        "bc_loss",
        "imit_reg_loss",
        "imit_total_loss",
        "q_data_mean",
        "valid_action_ratio",
        "eval_return",
        "eval_return_std",
        "eval_success_rate",
        "q_learning_rate",
        "imit_learning_rate",
    ]
    with (metrics_dir / "training_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(logs)

    torch.save({"model_state_dict": qnet.state_dict(), "args": vars(args)}, models_dir / "bcq_qnet.pt")
    torch.save({"model_state_dict": imit_net.state_dict(), "args": vars(args)}, models_dir / "bcq_imit_net.pt")

    policy_payload = {}
    qmax_payload = {}
    behavior_prob_payload = {}
    with torch.no_grad():
        for s in range(env.n):
            feat = torch.tensor([[env.state_to_feature(s)]], dtype=torch.float32, device=device)
            qv = qnet(feat)
            logits = imit_net(feat)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            action, _ = _select_actions_from_q_and_bcq_mask(qv, logits, args.bcq_threshold)
            a = int(action.item())
            policy_payload[str(s)] = ACTION_TEXT[a]
            qmax_payload[str(s)] = float(torch.max(qv).item())
            behavior_prob_payload[str(s)] = {"L": float(probs[0]), "R": float(probs[1])}

    (metrics_dir / "policy.json").write_text(json.dumps(policy_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (metrics_dir / "qmax_by_state.json").write_text(
        json.dumps(qmax_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (metrics_dir / "behavior_prob_by_state.json").write_text(
        json.dumps(behavior_prob_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"`matplotlib` is required for visualization: {exc}") from exc

    steps = [x["step"] for x in logs]
    q_loss = [x["q_loss"] for x in logs]
    bc_loss = [x["bc_loss"] for x in logs]
    imit_total = [x["imit_total_loss"] for x in logs]
    valid_ratio = [x["valid_action_ratio"] for x in logs]

    eval_steps = [x["step"] for x in logs if not np.isnan(x["eval_return"])]
    eval_return = [x["eval_return"] for x in logs if not np.isnan(x["eval_return"])]
    eval_success = [x["eval_success_rate"] for x in logs if not np.isnan(x["eval_success_rate"])]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(steps, q_loss, label="q_loss")
    axes[0, 0].plot(steps, bc_loss, label="bc_loss")
    axes[0, 0].plot(steps, imit_total, label="imit_total")
    axes[0, 0].set_title("BCQ Losses")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(eval_steps, eval_return, marker="o", label="eval_return")
    axes[0, 1].plot(eval_steps, eval_success, marker="o", label="eval_success_rate")
    axes[0, 1].set_title("Offline Policy Evaluation")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(steps, valid_ratio, color="#2ca02c", label="valid_action_ratio")
    axes[1, 0].set_title("BCQ Action Constraint Strength")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].set_ylabel("avg allowed-action ratio")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    states = list(range(env.n))
    qvals = [qmax_payload[str(s)] for s in states]
    axes[1, 1].plot(states, qvals, marker="o", label="max Q(s,.)")
    for s in states:
        axes[1, 1].text(s, qvals[s], policy_payload[str(s)], ha="center", va="bottom")
    axes[1, 1].set_title("Final BCQ Policy by State")
    axes[1, 1].set_xlabel("state")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(metrics_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    summary = {
        "updates": len(logs),
        "bcq_threshold": args.bcq_threshold,
        "final_q_loss": q_loss[-1] if q_loss else None,
        "final_bc_loss": bc_loss[-1] if bc_loss else None,
        "final_eval_return": eval_return[-1] if eval_return else None,
        "best_eval_return": max(eval_return) if eval_return else None,
        "final_eval_success_rate": eval_success[-1] if eval_success else None,
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics_dir


def main() -> None:
    """主流程入口：离线数据构建 + BCQ 训练 + 可视化导出。"""
    print("=== BCQ 主流程（学习版）===", flush=True)

    # 步骤 1：读取参数、设置随机种子、选择设备。
    args = build_default_args()
    set_seed(args.seed)
    device = detect_device()
    print(f"Runtime: device={device.type}")

    # 步骤 2：创建标准目录并保存运行配置，便于复现实验。
    code_dir = Path(__file__).resolve().parent
    module_dir = code_dir.parent
    layout = ensure_layout_dirs(module_dir=module_dir, output_arg=args.output_dir)
    (layout["output"] / "bcq_run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 步骤 3：构建离线环境并采集/加载离线数据集。
    env = LineWorld(args)
    dataset = collect_offline_dataset(env, args, layout["data"])
    dataset_stats = json.loads((layout["data"] / "dataset_stats.json").read_text(encoding="utf-8"))

    # 步骤 4：初始化 BCQ 所需网络（Q 网络 + 行为克隆网络）及其目标网络。
    qnet = QNet().to(device)
    target_qnet = QNet().to(device)
    target_qnet.load_state_dict(qnet.state_dict())

    imit_net = ImitationNet().to(device)
    target_imit_net = ImitationNet().to(device)
    target_imit_net.load_state_dict(imit_net.state_dict())

    q_optimizer = torch.optim.Adam(qnet.parameters(), lr=args.q_learning_rate)
    imit_optimizer = torch.optim.Adam(imit_net.parameters(), lr=args.imit_learning_rate)

    # 步骤 5：执行 BCQ 训练，日志和 checkpoint 会持续写入磁盘。
    logs = train_bcq(
        env=env,
        dataset=dataset,
        qnet=qnet,
        target_qnet=target_qnet,
        imit_net=imit_net,
        target_imit_net=target_imit_net,
        q_optimizer=q_optimizer,
        imit_optimizer=imit_optimizer,
        args=args,
        device=device,
        checkpoints_dir=layout["checkpoints"],
    )

    # 步骤 6：导出学习结果（曲线/指标/策略），用于复盘与核心要点讲解。
    metrics_dir = export_artifacts(
        logs=logs,
        dataset_stats=dataset_stats,
        output_dir=layout["output"],
        models_dir=layout["models"],
        qnet=qnet,
        imit_net=imit_net,
        env=env,
        args=args,
        device=device,
    )
    print(f"BCQ done. Visualization exported to: {metrics_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
