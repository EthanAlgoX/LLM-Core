#!/usr/bin/env python3
"""
GAE（Generalized Advantage Estimation）最小可运行示例。

一、GAE 原理（面向实现）
1) GAE 用于估计优势函数 A_t，降低策略梯度方差并保持较低偏差。
2) 基础项是 TD 残差：delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)。
3) 通过指数加权累计 TD 残差，得到：
   A_t = delta_t + gamma * lambda * delta_{t+1} + ...
4) lambda 越大，估计更接近蒙特卡洛回报；越小更接近一步 TD。

新人阅读顺序（建议）
1) 先看 `build_default_args`：明确可调参数和默认值。
2) 再看 `main`：把握执行主链路（准备 -> 训练/推理 -> 导出）。
3) 最后看可视化导出函数（如 `export_artifacts`）理解输出文件。

二、代码框架（从入口到结果）
1) `build_default_args`：读取环境与训练参数。
2) `LineWorld`：构建轻量环境采样轨迹。
3) `compute_gae`：根据 rewards/values 计算 advantage 与 return。
4) `train_loop`：执行 Actor-Critic + GAE 训练。
5) `export_artifacts`：导出 CSV/JSON/曲线图/summary。
6) `main`：串联完整流程。

用法：
  python code/gae.py
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
    """解析命令行参数，返回 GAE 训练配置。"""
    parser = argparse.ArgumentParser(description="Run GAE demo training and export visualization artifacts.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)

    # 环境参数：一维 LineWorld，左端失败(-1)、右端成功(+1)。
    parser.add_argument("--line-size", type=int, default=9)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--step-penalty", type=float, default=-0.01)
    parser.add_argument("--goal-reward", type=float, default=1.0)
    parser.add_argument("--fail-reward", type=float, default=-1.0)

    # 训练参数。
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--episodes-per-iter", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=50)
    return parser.parse_known_args([])[0]


def detect_device() -> torch.device:
    """选择运行设备：CUDA > MPS > CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """设置随机种子，增强可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_output_dir(base_dir: Path, output_dir: str) -> Path:
    """解析输出目录：相对路径按 gae 目录解析，绝对路径原样使用。"""
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


@dataclass
class StepResult:
    """环境一步交互结果。"""

    state: int
    reward: float
    done: bool


class LineWorld:
    """一维离散环境，用于演示 GAE 与 Actor-Critic。"""

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
        """重置到中间状态。"""
        self.state = self.n // 2
        self.t = 0
        return self.state

    def _is_terminal(self, s: int) -> bool:
        return s == 0 or s == self.n - 1

    def step(self, action: int) -> StepResult:
        """执行动作：0 向左，1 向右。"""
        if action not in (0, 1):
            raise ValueError(f"invalid action: {action}")
        if self._is_terminal(self.state):
            return StepResult(self.state, 0.0, True)

        self.t += 1
        self.state += -1 if action == 0 else 1
        self.state = max(0, min(self.n - 1, self.state))

        if self.state == self.n - 1:
            return StepResult(self.state, self.goal_reward, True)
        if self.state == 0:
            return StepResult(self.state, self.fail_reward, True)
        if self.t >= self.max_steps:
            return StepResult(self.state, self.step_penalty, True)
        return StepResult(self.state, self.step_penalty, False)

    def state_tensor(self, s: int, device: torch.device) -> torch.Tensor:
        """状态编码为归一化标量特征 [-1, 1]。"""
        x = (float(s) / float(self.n - 1)) * 2.0 - 1.0
        return torch.tensor([[x]], dtype=torch.float32, device=device)


class TinyActorCritic(nn.Module):
    """轻量 Actor-Critic 网络：共享 trunk + policy/value head。"""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, 2)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[float],
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    """根据轨迹计算 GAE advantages 与 returns。"""
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


def collect_batch(
    env: LineWorld,
    model: TinyActorCritic,
    episodes_per_iter: int,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
) -> dict[str, Any]:
    """采样一批 episode，并计算对应的 advantages/returns。"""
    model.eval()
    all_states: list[torch.Tensor] = []
    all_actions: list[int] = []
    all_advantages: list[float] = []
    all_returns: list[float] = []
    episode_returns: list[float] = []
    episode_lengths: list[float] = []

    for _ in range(episodes_per_iter):
        s = env.reset()
        done = False
        rewards: list[float] = []
        values: list[float] = []
        dones: list[float] = []
        states_ep: list[torch.Tensor] = []
        actions_ep: list[int] = []
        ep_reward = 0.0
        steps = 0

        while not done:
            state_tensor = env.state_tensor(s, device)
            with torch.no_grad():
                logits, value = model(state_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = int(dist.sample().item())

            result = env.step(action)
            rewards.append(float(result.reward))
            values.append(float(value.item()))
            dones.append(1.0 if result.done else 0.0)
            states_ep.append(state_tensor.squeeze(0))
            actions_ep.append(action)

            ep_reward += float(result.reward)
            steps += 1
            s = result.state
            done = result.done

        # 终止状态不做 bootstrap；若未终止可用 V(s_T) bootstrap。
        if done:
            next_value = 0.0
        else:
            with torch.no_grad():
                _, nv = model(env.state_tensor(s, device))
                next_value = float(nv.item())

        adv_ep, ret_ep = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            next_value=next_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        all_states.extend(states_ep)
        all_actions.extend(actions_ep)
        all_advantages.extend(adv_ep)
        all_returns.extend(ret_ep)
        episode_returns.append(ep_reward)
        episode_lengths.append(float(steps))

    batch = {
        "states": torch.stack(all_states, dim=0),  # [N, 1]
        "actions": torch.tensor(all_actions, dtype=torch.long, device=device),  # [N]
        "advantages": torch.tensor(all_advantages, dtype=torch.float32, device=device),  # [N]
        "returns": torch.tensor(all_returns, dtype=torch.float32, device=device),  # [N]
        "reward_mean": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "reward_std": float(np.std(episode_returns)) if episode_returns else 0.0,
        "episode_len_mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
    }
    return batch


def save_checkpoint(
    checkpoints_dir: Path,
    iteration: int,
    model: TinyActorCritic,
    optimizer: torch.optim.Optimizer,
    log_item: dict[str, float],
) -> None:
    """保存训练 checkpoint。"""
    ckpt_path = checkpoints_dir / f"checkpoint-{iteration}.pt"
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "log": log_item,
        },
        ckpt_path,
    )


def train_loop(
    env: LineWorld,
    model: TinyActorCritic,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    checkpoints_dir: Path,
) -> list[dict[str, float]]:
    """执行 GAE 训练循环并返回日志。"""
    logs: list[dict[str, float]] = []
    for it in range(1, args.iterations + 1):
        batch = collect_batch(
            env=env,
            model=model,
            episodes_per_iter=args.episodes_per_iter,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            device=device,
        )

        model.train()
        logits, values = model(batch["states"])
        values = values.squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(batch["actions"])
        entropy = dist.entropy().mean()

        # 对优势归一化，提升训练稳定性。
        adv = batch["advantages"]
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        policy_loss = -(log_probs * adv).mean()
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
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        logs.append(log_item)

        if it % args.log_every == 0 or it == 1 or it == args.iterations:
            print(
                f"[Iter {it:04d}] reward_mean={log_item['reward_mean']:.4f} "
                f"value_loss={log_item['value_loss']:.6f} "
                f"policy_loss={log_item['policy_loss']:.6f}"
            )
        if it % args.save_every == 0 or it == args.iterations:
            save_checkpoint(checkpoints_dir, it, model, optimizer, log_item)

    return logs


def export_artifacts(
    logs: list[dict[str, float]],
    output_dir: Path,
    models_dir: Path,
    model: TinyActorCritic,
    args: argparse.Namespace,
) -> Path:
    """导出日志、曲线图、summary 与最终模型。"""
    metrics_dir = output_dir / "gae_metrics"
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
                "policy_loss",
                "value_loss",
                "entropy",
                "learning_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(logs)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"`matplotlib` is required for visualization: {exc}") from exc

    iters = [x["iteration"] for x in logs]
    reward_mean = [x["reward_mean"] for x in logs]
    reward_std = [x["reward_std"] for x in logs]
    value_loss = [x["value_loss"] for x in logs]
    policy_loss = [x["policy_loss"] for x in logs]
    entropy = [x["entropy"] for x in logs]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(iters, reward_mean, label="reward_mean")
    axes[0, 0].fill_between(
        iters,
        np.array(reward_mean) - np.array(reward_std),
        np.array(reward_mean) + np.array(reward_std),
        alpha=0.2,
        label="reward_std",
    )
    axes[0, 0].set_title("Episode Reward")
    axes[0, 0].set_xlabel("iteration")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(iters, value_loss, color="#d62728", label="value_loss")
    axes[0, 1].set_title("Value Loss")
    axes[0, 1].set_xlabel("iteration")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(iters, policy_loss, color="#1f77b4", label="policy_loss")
    axes[1, 0].set_title("Policy Loss")
    axes[1, 0].set_xlabel("iteration")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(iters, entropy, color="#2ca02c", label="entropy")
    axes[1, 1].set_title("Policy Entropy")
    axes[1, 1].set_xlabel("iteration")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(metrics_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    summary = {
        "iterations": len(logs),
        "final_reward_mean": reward_mean[-1] if reward_mean else None,
        "best_reward_mean": max(reward_mean) if reward_mean else None,
        "final_value_loss": value_loss[-1] if value_loss else None,
        "final_policy_loss": policy_loss[-1] if policy_loss else None,
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    model_path = models_dir / "gae_actor_critic.pt"
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, model_path)
    return metrics_dir


def main() -> None:
    """主流程入口：执行 GAE 训练并导出可视化结果。"""
    print("=== GAE 主流程（学习版）===", flush=True)

    # 步骤 1：读取参数、设置随机种子、选择设备。
    args = build_default_args()
    set_seed(args.seed)
    device = detect_device()
    print(f"Runtime: device={device.type}")

    # 步骤 2：创建目录并保存运行配置。
    code_dir = Path(__file__).resolve().parent
    module_dir = code_dir.parent
    layout = ensure_layout_dirs(module_dir=module_dir, output_arg=args.output_dir)
    (layout["output"] / "gae_run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 步骤 3：初始化环境、Actor-Critic 网络与优化器。
    env = LineWorld(args)
    model = TinyActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 步骤 4：执行 GAE 训练，记录损失与优势统计。
    logs = train_loop(
        env=env,
        model=model,
        optimizer=optimizer,
        args=args,
        device=device,
        checkpoints_dir=layout["checkpoints"],
    )

    # 步骤 5：导出曲线图和 summary，观察训练是否稳定收敛。
    metrics_dir = export_artifacts(
        logs=logs,
        output_dir=layout["output"],
        models_dir=layout["models"],
        model=model,
        args=args,
    )
    print(f"GAE done. Visualization exported to: {metrics_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
