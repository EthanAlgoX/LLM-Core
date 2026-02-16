#!/usr/bin/env python3
"""
Advantage（优势函数）最小可运行示例：MC / TD / GAE 三种估计方式。

一、Advantage 原理（面向实现）
1) 优势函数定义：A(s,a) = Q(s,a) - V(s)，表示动作相对基线的“好坏”。
2) 在策略梯度中，用 advantage 代替 return 可显著降低方差。
3) 常见估计方式：
   - MC: A_t = G_t - V(s_t)
   - TD(1-step): A_t = r_t + gamma*V(s_{t+1}) - V(s_t)
   - GAE: 对多步 TD 残差做指数加权和。

二、代码框架（从入口到结果）
1) `build_default_args`：读取环境与训练参数。
2) `LineWorld`：轻量环境。
3) `compute_advantages_*`：三种优势估计实现。
4) `train_loop`：Actor-Critic 训练（可切换 advantage 方式）。
5) `export_artifacts`：导出 CSV/JSON/曲线图/对比图。
6) `main`：串联流程。

用法：
  python code/advantage.py
  python code/advantage.py --advantage-method td
  python code/advantage.py --advantage-method gae
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


def build_default_args() -> argparse.Namespace:
    """解析命令行参数，返回 Advantage 训练配置。"""
    parser = argparse.ArgumentParser(description="Run advantage-estimation demo and export visualization artifacts.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)

    # 环境参数。
    parser.add_argument("--line-size", type=int, default=9)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--step-penalty", type=float, default=-0.01)
    parser.add_argument("--goal-reward", type=float, default=1.0)
    parser.add_argument("--fail-reward", type=float, default=-1.0)

    # 训练参数。
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--episodes-per-iter", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--advantage-method", default="gae", choices=["mc", "td", "gae"])
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=50)
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
    """解析输出目录：相对路径按 advantage 目录解析。"""
    out = Path(output_dir)
    if not out.is_absolute():
        out = (base_dir / out).resolve()
    return out


def ensure_layout_dirs(module_dir: Path, output_arg: str) -> dict[str, Path]:
    """创建并返回标准目录布局。"""
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


@dataclass
class StepResult:
    """环境一步交互结果。"""

    state: int
    reward: float
    done: bool


class LineWorld:
    """简单一维环境：左端失败，右端成功。"""

    def __init__(self, args: argparse.Namespace) -> None:
        if args.line_size < 3:
            raise ValueError("--line-size must be >= 3")
        self.n = args.line_size
        self.max_steps = args.max_steps
        self.step_penalty = args.step_penalty
        self.goal_reward = args.goal_reward
        self.fail_reward = args.fail_reward
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
            return StepResult(self.s, 0.0, True)

        self.t += 1
        self.s += -1 if action == 0 else 1
        self.s = max(0, min(self.n - 1, self.s))

        if self.s == self.n - 1:
            return StepResult(self.s, self.goal_reward, True)
        if self.s == 0:
            return StepResult(self.s, self.fail_reward, True)
        if self.t >= self.max_steps:
            return StepResult(self.s, self.step_penalty, True)
        return StepResult(self.s, self.step_penalty, False)

    def state_tensor(self, s: int, device: torch.device) -> torch.Tensor:
        x = (float(s) / float(self.n - 1)) * 2.0 - 1.0
        return torch.tensor([[x]], dtype=torch.float32, device=device)


class TinyActorCritic(nn.Module):
    """轻量 Actor-Critic 网络。"""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden_dim, 2)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy(h), self.value(h)


def compute_returns_mc(rewards: list[float], gamma: float) -> list[float]:
    """蒙特卡洛回报 G_t。"""
    returns = [0.0 for _ in rewards]
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        returns[t] = g
    return returns


def compute_advantages_mc(rewards: list[float], values: list[float], gamma: float) -> tuple[list[float], list[float]]:
    """MC 方式优势：A_t = G_t - V_t。"""
    returns = compute_returns_mc(rewards, gamma)
    advantages = [returns[t] - values[t] for t in range(len(rewards))]
    return advantages, returns


def compute_advantages_td(
    rewards: list[float],
    values: list[float],
    dones: list[float],
    next_value: float,
    gamma: float,
) -> tuple[list[float], list[float]]:
    """一步 TD 方式优势。"""
    advantages = [0.0 for _ in rewards]
    returns = [0.0 for _ in rewards]
    for t in range(len(rewards)):
        nv = next_value if t == len(rewards) - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * non_terminal - values[t]
        advantages[t] = delta
        returns[t] = advantages[t] + values[t]
    return advantages, returns


def compute_advantages_gae(
    rewards: list[float],
    values: list[float],
    dones: list[float],
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    """GAE 方式优势。"""
    advantages = [0.0 for _ in rewards]
    gae = 0.0
    for t in reversed(range(len(rewards))):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[t] = gae
        next_value = values[t]
    returns = [advantages[t] + values[t] for t in range(len(rewards))]
    return advantages, returns


def estimate_advantages(
    method: str,
    rewards: list[float],
    values: list[float],
    dones: list[float],
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    """按 method 选择优势估计策略。"""
    if method == "mc":
        return compute_advantages_mc(rewards, values, gamma)
    if method == "td":
        return compute_advantages_td(rewards, values, dones, next_value, gamma)
    return compute_advantages_gae(rewards, values, dones, next_value, gamma, gae_lambda)


def collect_batch(
    env: LineWorld,
    model: TinyActorCritic,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    """采样一批轨迹并计算优势。"""
    model.eval()
    states_all: list[torch.Tensor] = []
    actions_all: list[int] = []
    adv_all: list[float] = []
    ret_all: list[float] = []
    ep_rewards: list[float] = []
    ep_lens: list[float] = []

    for _ in range(args.episodes_per_iter):
        s = env.reset()
        done = False
        rewards: list[float] = []
        values: list[float] = []
        dones: list[float] = []
        states_ep: list[torch.Tensor] = []
        actions_ep: list[int] = []
        ep_reward = 0.0
        ep_len = 0

        while not done:
            st = env.state_tensor(s, device)
            with torch.no_grad():
                logits, value = model(st)
                dist = torch.distributions.Categorical(logits=logits)
                action = int(dist.sample().item())

            step = env.step(action)
            rewards.append(float(step.reward))
            values.append(float(value.item()))
            dones.append(1.0 if step.done else 0.0)
            states_ep.append(st.squeeze(0))
            actions_ep.append(action)

            s = step.state
            done = step.done
            ep_reward += float(step.reward)
            ep_len += 1

        next_value = 0.0
        adv_ep, ret_ep = estimate_advantages(
            method=args.advantage_method,
            rewards=rewards,
            values=values,
            dones=dones,
            next_value=next_value,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )

        states_all.extend(states_ep)
        actions_all.extend(actions_ep)
        adv_all.extend(adv_ep)
        ret_all.extend(ret_ep)
        ep_rewards.append(ep_reward)
        ep_lens.append(float(ep_len))

    return {
        "states": torch.stack(states_all, dim=0),
        "actions": torch.tensor(actions_all, dtype=torch.long, device=device),
        "advantages": torch.tensor(adv_all, dtype=torch.float32, device=device),
        "returns": torch.tensor(ret_all, dtype=torch.float32, device=device),
        "reward_mean": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
        "reward_std": float(np.std(ep_rewards)) if ep_rewards else 0.0,
        "episode_len_mean": float(np.mean(ep_lens)) if ep_lens else 0.0,
        "adv_mean": float(np.mean(adv_all)) if adv_all else 0.0,
        "adv_std": float(np.std(adv_all)) if adv_all else 0.0,
    }


def save_checkpoint(
    checkpoints_dir: Path,
    iteration: int,
    model: TinyActorCritic,
    optimizer: torch.optim.Optimizer,
    log_item: dict[str, float],
) -> None:
    """保存 checkpoint。"""
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "log": log_item,
        },
        checkpoints_dir / f"checkpoint-{iteration}.pt",
    )


def train_loop(
    env: LineWorld,
    model: TinyActorCritic,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    checkpoints_dir: Path,
) -> list[dict[str, float]]:
    """执行训练并返回日志。"""
    logs: list[dict[str, float]] = []
    for it in range(1, args.iterations + 1):
        batch = collect_batch(env, model, args, device)
        model.train()
        logits, values = model(batch["states"])
        values = values.squeeze(-1)

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(batch["actions"])
        entropy = dist.entropy().mean()

        advantages = batch["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, batch["returns"])
        loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        log_item = {
            "iteration": float(it),
            "reward_mean": batch["reward_mean"],
            "reward_std": batch["reward_std"],
            "episode_len_mean": batch["episode_len_mean"],
            "adv_mean": batch["adv_mean"],
            "adv_std": batch["adv_std"],
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        logs.append(log_item)

        if it % args.log_every == 0 or it == 1 or it == args.iterations:
            print(
                f"[Iter {it:04d}] reward_mean={log_item['reward_mean']:.4f} "
                f"adv_std={log_item['adv_std']:.4f} "
                f"value_loss={log_item['value_loss']:.6f}"
            )
        if it % args.save_every == 0 or it == args.iterations:
            save_checkpoint(checkpoints_dir, it, model, optimizer, log_item)

    return logs


def build_method_comparison(
    env: LineWorld,
    model: TinyActorCritic,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, list[float]]:
    """在同一条轨迹上对比 MC/TD/GAE 优势估计。"""
    model.eval()
    s = env.reset()
    done = False
    rewards: list[float] = []
    values: list[float] = []
    dones: list[float] = []
    while not done:
        st = env.state_tensor(s, device)
        with torch.no_grad():
            logits, value = model(st)
            action = int(torch.argmax(logits, dim=-1).item())
        step = env.step(action)
        rewards.append(float(step.reward))
        values.append(float(value.item()))
        dones.append(1.0 if step.done else 0.0)
        s = step.state
        done = step.done

    next_value = 0.0
    adv_mc, _ = compute_advantages_mc(rewards, values, args.gamma)
    adv_td, _ = compute_advantages_td(rewards, values, dones, next_value, args.gamma)
    adv_gae, _ = compute_advantages_gae(rewards, values, dones, next_value, args.gamma, args.gae_lambda)
    return {"mc": adv_mc, "td": adv_td, "gae": adv_gae}


def export_artifacts(
    logs: list[dict[str, float]],
    comparison: dict[str, list[float]],
    output_dir: Path,
    models_dir: Path,
    model: TinyActorCritic,
    args: argparse.Namespace,
) -> Path:
    """导出指标、图像和最终模型。"""
    metrics_dir = output_dir / "advantage_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    (metrics_dir / "training_log.json").write_text(
        json.dumps(logs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (metrics_dir / "training_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iteration",
                "reward_mean",
                "reward_std",
                "episode_len_mean",
                "adv_mean",
                "adv_std",
                "policy_loss",
                "value_loss",
                "entropy",
                "learning_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(logs)

    (metrics_dir / "advantage_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"`matplotlib` is required for visualization: {exc}") from exc

    iters = [x["iteration"] for x in logs]
    reward_mean = [x["reward_mean"] for x in logs]
    reward_std = [x["reward_std"] for x in logs]
    policy_loss = [x["policy_loss"] for x in logs]
    value_loss = [x["value_loss"] for x in logs]
    adv_std = [x["adv_std"] for x in logs]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(iters, reward_mean, label="reward_mean")
    axes[0, 0].fill_between(
        iters,
        np.array(reward_mean) - np.array(reward_std),
        np.array(reward_mean) + np.array(reward_std),
        alpha=0.2,
    )
    axes[0, 0].set_title("Episode Reward")
    axes[0, 0].set_xlabel("iteration")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iters, value_loss, color="#d62728")
    axes[0, 1].set_title("Value Loss")
    axes[0, 1].set_xlabel("iteration")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(iters, policy_loss, color="#1f77b4")
    axes[1, 0].set_title("Policy Loss")
    axes[1, 0].set_xlabel("iteration")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(iters, adv_std, color="#2ca02c")
    axes[1, 1].set_title("Advantage Std")
    axes[1, 1].set_xlabel("iteration")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(metrics_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    # 方法对比图：同一轨迹上 MC/TD/GAE 的 A_t。
    fig2, ax = plt.subplots(1, 1, figsize=(8, 4))
    x = list(range(1, len(comparison["mc"]) + 1))
    ax.plot(x, comparison["mc"], marker="o", label="MC")
    ax.plot(x, comparison["td"], marker="o", label="TD(1-step)")
    ax.plot(x, comparison["gae"], marker="o", label="GAE")
    ax.set_title("Advantage Estimation Comparison")
    ax.set_xlabel("timestep")
    ax.set_ylabel("A_t")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(metrics_dir / "advantage_methods.png", dpi=160)
    plt.close(fig2)

    summary = {
        "method": args.advantage_method,
        "iterations": len(logs),
        "final_reward_mean": reward_mean[-1] if reward_mean else None,
        "best_reward_mean": max(reward_mean) if reward_mean else None,
        "final_adv_std": adv_std[-1] if adv_std else None,
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    model_path = models_dir / "advantage_actor_critic.pt"
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, model_path)
    return metrics_dir


def main() -> None:
    """主流程入口。"""
    args = build_default_args()
    set_seed(args.seed)
    device = detect_device()
    print(f"Runtime: device={device.type}, method={args.advantage_method}")

    code_dir = Path(__file__).resolve().parent
    module_dir = code_dir.parent
    layout = ensure_layout_dirs(module_dir=module_dir, output_arg=args.output_dir)
    (layout["output"] / "advantage_run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    env = LineWorld(args)
    model = TinyActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    logs = train_loop(
        env=env,
        model=model,
        optimizer=optimizer,
        args=args,
        device=device,
        checkpoints_dir=layout["checkpoints"],
    )
    comparison = build_method_comparison(env, model, args, device)
    metrics_dir = export_artifacts(
        logs=logs,
        comparison=comparison,
        output_dir=layout["output"],
        models_dir=layout["models"],
        model=model,
        args=args,
    )
    print(f"Advantage done. Visualization exported to: {metrics_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
