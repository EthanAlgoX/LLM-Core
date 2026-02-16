#!/usr/bin/env python3
"""
CQL（Conservative Q-Learning）最小可运行示例：LineWorld Offline RL。

一、CQL 原理（面向实现）
1) CQL 属于离线强化学习：只用固定数据集训练，不与环境在线交互更新。
2) 标准 Q-learning 容易对“数据分布外动作”过度乐观。
3) CQL 在 TD 损失外加入保守正则：
   L_cql = E_s[logsumexp_a Q(s,a)] - E_(s,a~D)[Q(s,a)]
4) 该项会压低未见动作 Q 值，减小分布外过估计风险。

二、代码框架（从入口到结果）
1) `parse_args`：读取环境、离线数据和训练参数。
2) `collect_offline_dataset`：构建行为策略数据集。
3) `train_cql`：执行 CQL 训练。
4) `evaluate_policy`：周期评估贪心策略性能。
5) `export_artifacts`：导出 CSV/JSON/曲线图/summary。
6) `main`：串联完整流程。

用法：
  python code/cql.py
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


def parse_args() -> argparse.Namespace:
    """解析命令行参数，返回 CQL 配置。"""
    parser = argparse.ArgumentParser(description="Run CQL demo training and export visualization artifacts.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)

    # 环境参数：一维环境，左端失败，右端成功。
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

    # CQL 训练参数。
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--cql-alpha", type=float, default=1.0)
    parser.add_argument("--target-update-tau", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=200)
    return parser.parse_args()


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
    """解析输出目录：相对路径按 cql 目录解析。"""
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
        # 轻微 slip：小概率翻转动作。
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
        """离散状态映射为归一化特征。"""
        return (float(s) / float(self.n - 1)) * 2.0 - 1.0


class QNet(nn.Module):
    """轻量 Q 网络：输入状态标量，输出两个动作 Q 值。"""

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


def collect_offline_dataset(
    env: LineWorld,
    args: argparse.Namespace,
    data_dir: Path,
) -> dict[str, np.ndarray]:
    """用行为策略采样离线数据集。"""
    states, actions, rewards, next_states, dones = [], [], [], [], []
    success_count = 0

    for _ in range(args.dataset_episodes):
        s = env.reset()
        done = False
        while not done:
            # 行为策略：epsilon 随机 + 右偏动作。
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

    stats = {
        "num_transitions": int(ds["states"].shape[0]),
        "dataset_episodes": int(args.dataset_episodes),
        "behavior_success_rate": float(success_count / max(args.dataset_episodes, 1)),
        "reward_mean": float(np.mean(ds["rewards"])) if ds["rewards"].size > 0 else 0.0,
        "reward_std": float(np.std(ds["rewards"])) if ds["rewards"].size > 0 else 0.0,
    }
    (data_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(
        f"Offline dataset ready: transitions={stats['num_transitions']}, "
        f"success_rate={stats['behavior_success_rate']:.3f}"
    )
    return ds


def sample_batch(dataset: dict[str, np.ndarray], batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    """从离线数据集中随机采样一个 batch。"""
    n = dataset["states"].shape[0]
    idx = np.random.randint(0, n, size=batch_size)
    return {
        "states": torch.tensor(dataset["states"][idx], dtype=torch.float32, device=device),
        "actions": torch.tensor(dataset["actions"][idx], dtype=torch.long, device=device),
        "rewards": torch.tensor(dataset["rewards"][idx], dtype=torch.float32, device=device),
        "next_states": torch.tensor(dataset["next_states"][idx], dtype=torch.float32, device=device),
        "dones": torch.tensor(dataset["dones"][idx], dtype=torch.float32, device=device),
    }


@torch.no_grad()
def evaluate_policy(env: LineWorld, qnet: QNet, episodes: int, device: torch.device) -> tuple[float, float, float]:
    """评估贪心策略，返回平均回报、回报标准差、成功率。"""
    qnet.eval()
    returns, succ = [], 0
    for _ in range(episodes):
        s = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            st = torch.tensor([[env.state_to_feature(s)]], dtype=torch.float32, device=device)
            q = qnet(st)
            a = int(torch.argmax(q, dim=-1).item())
            step = env.step(a)
            ep_ret += step.reward
            if step.done and step.success:
                succ += 1
            s = step.state
            done = step.done
        returns.append(ep_ret)
    return float(np.mean(returns)), float(np.std(returns)), float(succ / max(episodes, 1))


def soft_update(target: QNet, online: QNet, tau: float) -> None:
    """Polyak 平滑更新 target 网络。"""
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)


def save_checkpoint(
    checkpoints_dir: Path,
    step: int,
    qnet: QNet,
    target_qnet: QNet,
    optimizer: torch.optim.Optimizer,
    log_item: dict[str, float],
) -> None:
    """保存训练 checkpoint。"""
    torch.save(
        {
            "step": step,
            "qnet_state_dict": qnet.state_dict(),
            "target_qnet_state_dict": target_qnet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "log": log_item,
        },
        checkpoints_dir / f"checkpoint-{step}.pt",
    )


def train_cql(
    env: LineWorld,
    dataset: dict[str, np.ndarray],
    qnet: QNet,
    target_qnet: QNet,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    checkpoints_dir: Path,
) -> list[dict[str, float]]:
    """执行 CQL 训练并返回日志。"""
    logs: list[dict[str, float]] = []
    for step in range(1, args.updates + 1):
        batch = sample_batch(dataset, args.batch_size, device)

        q = qnet(batch["states"])
        q_data = q.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = target_qnet(batch["next_states"])
            next_q_max = torch.max(next_q, dim=1).values
            td_target = batch["rewards"] + args.gamma * (1.0 - batch["dones"]) * next_q_max

        td_loss = F.mse_loss(q_data, td_target)

        q_logsumexp = torch.logsumexp(q, dim=1)
        cql_loss = (q_logsumexp - q_data).mean()
        total_loss = td_loss + args.cql_alpha * cql_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(qnet.parameters(), args.max_grad_norm)
        optimizer.step()
        soft_update(target_qnet, qnet, args.target_update_tau)

        eval_return, eval_std, eval_success = np.nan, np.nan, np.nan
        if step % args.eval_every == 0 or step == 1 or step == args.updates:
            eval_return, eval_std, eval_success = evaluate_policy(env, qnet, args.eval_episodes, device)

        log_item = {
            "step": float(step),
            "td_loss": float(td_loss.item()),
            "cql_loss": float(cql_loss.item()),
            "total_loss": float(total_loss.item()),
            "q_data_mean": float(q_data.mean().item()),
            "q_logsumexp_mean": float(q_logsumexp.mean().item()),
            "conservative_gap": float((q_logsumexp.mean() - q_data.mean()).item()),
            "eval_return": float(eval_return),
            "eval_return_std": float(eval_std),
            "eval_success_rate": float(eval_success),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        logs.append(log_item)

        if step % args.log_every == 0 or step == 1 or step == args.updates:
            print(
                f"[Step {step:04d}] total={log_item['total_loss']:.6f} "
                f"td={log_item['td_loss']:.6f} cql={log_item['cql_loss']:.6f} "
                f"eval_return={log_item['eval_return']:.4f}"
            )
        if step % args.save_every == 0 or step == args.updates:
            save_checkpoint(checkpoints_dir, step, qnet, target_qnet, optimizer, log_item)

    return logs


def export_artifacts(
    logs: list[dict[str, float]],
    dataset_stats: dict[str, Any],
    output_dir: Path,
    models_dir: Path,
    qnet: QNet,
    env: LineWorld,
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    """导出日志、模型、策略和可视化结果。"""
    metrics_dir = output_dir / "cql_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / "training_log.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    (metrics_dir / "dataset_stats.json").write_text(json.dumps(dataset_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = [
        "step",
        "td_loss",
        "cql_loss",
        "total_loss",
        "q_data_mean",
        "q_logsumexp_mean",
        "conservative_gap",
        "eval_return",
        "eval_return_std",
        "eval_success_rate",
        "learning_rate",
    ]
    with (metrics_dir / "training_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(logs)

    qnet_path = models_dir / "cql_qnet.pt"
    torch.save({"model_state_dict": qnet.state_dict(), "args": vars(args)}, qnet_path)

    # 导出最终贪心策略。
    policy_payload = {}
    qmax_payload = {}
    with torch.no_grad():
        for s in range(env.n):
            feat = torch.tensor([[env.state_to_feature(s)]], dtype=torch.float32, device=device)
            qv = qnet(feat).squeeze(0).cpu().numpy()
            action = int(np.argmax(qv))
            policy_payload[str(s)] = ACTION_TEXT[action]
            qmax_payload[str(s)] = float(np.max(qv))
    (metrics_dir / "policy.json").write_text(json.dumps(policy_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (metrics_dir / "qmax_by_state.json").write_text(json.dumps(qmax_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"`matplotlib` is required for visualization: {exc}") from exc

    steps = [x["step"] for x in logs]
    td_loss = [x["td_loss"] for x in logs]
    cql_loss = [x["cql_loss"] for x in logs]
    total_loss = [x["total_loss"] for x in logs]
    q_data = [x["q_data_mean"] for x in logs]
    q_lse = [x["q_logsumexp_mean"] for x in logs]
    gap = [x["conservative_gap"] for x in logs]

    eval_steps = [x["step"] for x in logs if not np.isnan(x["eval_return"])]
    eval_return = [x["eval_return"] for x in logs if not np.isnan(x["eval_return"])]
    eval_success = [x["eval_success_rate"] for x in logs if not np.isnan(x["eval_success_rate"])]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(steps, total_loss, label="total_loss")
    axes[0, 0].plot(steps, td_loss, label="td_loss")
    axes[0, 0].plot(steps, cql_loss, label="cql_loss")
    axes[0, 0].set_title("CQL Losses")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(steps, q_data, label="E[Q(s,a_data)]")
    axes[0, 1].plot(steps, q_lse, label="E[logsumexp(Q)]")
    axes[0, 1].plot(steps, gap, label="conservative_gap")
    axes[0, 1].set_title("Conservative Regularization")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(eval_steps, eval_return, marker="o", label="eval_return")
    axes[1, 0].plot(eval_steps, eval_success, marker="o", label="eval_success_rate")
    axes[1, 0].set_title("Offline Policy Evaluation")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    state_ids = list(range(env.n))
    qvals = [qmax_payload[str(s)] for s in state_ids]
    axes[1, 1].plot(state_ids, qvals, marker="o", label="max Q(s,.)")
    for s in state_ids:
        axes[1, 1].text(s, qvals[s], policy_payload[str(s)], ha="center", va="bottom")
    axes[1, 1].set_title("Final Policy by State")
    axes[1, 1].set_xlabel("state")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(metrics_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    summary = {
        "updates": len(logs),
        "final_total_loss": total_loss[-1] if total_loss else None,
        "final_eval_return": eval_return[-1] if eval_return else None,
        "best_eval_return": max(eval_return) if eval_return else None,
        "final_eval_success_rate": eval_success[-1] if eval_success else None,
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics_dir


def main() -> None:
    """主流程入口：离线数据构建 + CQL 训练 + 可视化导出。"""
    args = parse_args()
    set_seed(args.seed)
    device = detect_device()
    print(f"Runtime: device={device.type}")

    code_dir = Path(__file__).resolve().parent
    module_dir = code_dir.parent
    layout = ensure_layout_dirs(module_dir=module_dir, output_arg=args.output_dir)
    (layout["output"] / "cql_run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    env = LineWorld(args)
    dataset = collect_offline_dataset(env, args, layout["data"])
    dataset_stats = json.loads((layout["data"] / "dataset_stats.json").read_text(encoding="utf-8"))

    qnet = QNet().to(device)
    target_qnet = QNet().to(device)
    target_qnet.load_state_dict(qnet.state_dict())
    optimizer = torch.optim.Adam(qnet.parameters(), lr=args.learning_rate)

    logs = train_cql(
        env=env,
        dataset=dataset,
        qnet=qnet,
        target_qnet=target_qnet,
        optimizer=optimizer,
        args=args,
        device=device,
        checkpoints_dir=layout["checkpoints"],
    )

    metrics_dir = export_artifacts(
        logs=logs,
        dataset_stats=dataset_stats,
        output_dir=layout["output"],
        models_dir=layout["models"],
        qnet=qnet,
        env=env,
        args=args,
        device=device,
    )
    print(f"CQL done. Visualization exported to: {metrics_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
