#!/usr/bin/env python3
"""
TD Learning 最小可运行示例：GridWorld + Tabular Q-Learning。

一、TD Learning 原理（面向实现）
1) TD（Temporal-Difference）用“当前估计”去 bootstrap“下一步估计”。
2) 不需要等整条轨迹结束，就能逐步更新价值估计。
3) 这里使用 Q-learning（离策略 TD 控制）：
   Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]。
4) 通过 epsilon-greedy 探索，最终得到近似最优策略。

二、代码框架（从入口到结果）
1) `parse_args`：读取环境与训练参数。
2) `GridWorld`：定义状态、动作、转移与奖励。
3) `train_q_learning`：执行 TD 学习。
4) `extract_policy`：从 Q 表提取贪心策略。
5) `export_artifacts`：导出日志、Q 表、策略与可视化。
6) `main`：串联完整流程。

用法：
  python code/td_learning.py
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


ACTIONS = ["U", "D", "L", "R"]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}
ACTION_DELTA = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
ARROW = {"U": "↑", "D": "↓", "L": "←", "R": "→"}


def parse_args() -> argparse.Namespace:
    """解析命令行参数，返回 TD Learning 配置。"""
    parser = argparse.ArgumentParser(description="Run TD Learning (Q-learning) and export visualization artifacts.")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--seed", type=int, default=42)

    # 环境参数。
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--start", default="0,0")
    parser.add_argument("--goal", default="4,4")
    parser.add_argument("--traps", default="1,3;3,1")
    parser.add_argument("--walls", default="2,2")
    parser.add_argument("--step-reward", type=float, default=-0.04)
    parser.add_argument("--goal-reward", type=float, default=1.0)
    parser.add_argument("--trap-reward", type=float, default=-1.0)
    parser.add_argument("--slip-prob", type=float, default=0.1)

    # TD 训练参数。
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps-per-episode", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.2, help="学习率。")
    parser.add_argument("--gamma", type=float, default=0.95, help="折扣因子。")
    parser.add_argument("--epsilon-start", type=float, default=0.3)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--log-every-episodes", type=int, default=10)
    parser.add_argument("--save-every-episodes", type=int, default=100)
    return parser.parse_args()


def resolve_output_dir(base_dir: Path, output_dir: str) -> Path:
    """解析输出目录：相对路径按 td_learning 目录解析，绝对路径原样使用。"""
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


def parse_coord(s: str) -> tuple[int, int]:
    """解析坐标字符串 `r,c`。"""
    r_str, c_str = s.split(",")
    return int(r_str.strip()), int(c_str.strip())


def parse_coord_list(s: str) -> set[tuple[int, int]]:
    """解析坐标列表字符串 `r1,c1;r2,c2`。"""
    s = s.strip()
    if not s:
        return set()
    return {parse_coord(item) for item in s.split(";") if item.strip()}


class GridWorld:
    """带障碍/陷阱/目标的网格环境。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.rows = args.rows
        self.cols = args.cols
        self.start = parse_coord(args.start)
        self.goal = parse_coord(args.goal)
        self.traps = parse_coord_list(args.traps)
        self.walls = parse_coord_list(args.walls)

        self.step_reward = args.step_reward
        self.goal_reward = args.goal_reward
        self.trap_reward = args.trap_reward
        self.slip_prob = args.slip_prob

        self.state = self.start
        self.states = sorted(
            [(r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in self.walls]
        )
        self.terminal = {self.goal} | set(self.traps)

        if self.start in self.walls or self.goal in self.walls:
            raise ValueError("start/goal cannot be in walls")

    def reset(self) -> tuple[int, int]:
        """重置环境到起点。"""
        self.state = self.start
        return self.state

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _move(self, s: tuple[int, int], action: str) -> tuple[int, int]:
        if s in self.terminal:
            return s
        dr, dc = ACTION_DELTA[action]
        nr, nc = s[0] + dr, s[1] + dc
        if not self._in_bounds(nr, nc) or (nr, nc) in self.walls:
            return s
        return nr, nc

    def _sample_real_action(self, intended: str) -> str:
        """根据 slip 概率采样真实执行动作。"""
        side_actions = {
            "U": ("L", "R"),
            "D": ("R", "L"),
            "L": ("D", "U"),
            "R": ("U", "D"),
        }
        p = random.random()
        if p < 1.0 - self.slip_prob:
            return intended
        if p < 1.0 - self.slip_prob / 2.0:
            return side_actions[intended][0]
        return side_actions[intended][1]

    def step(self, action: str) -> tuple[tuple[int, int], float, bool]:
        """执行一步状态转移，返回 next_state, reward, done。"""
        real_action = self._sample_real_action(action)
        next_state = self._move(self.state, real_action)
        self.state = next_state

        if next_state == self.goal:
            return next_state, self.goal_reward, True
        if next_state in self.traps:
            return next_state, self.trap_reward, True
        return next_state, self.step_reward, False


def epsilon_greedy_action(q: dict[tuple[int, int], np.ndarray], s: tuple[int, int], epsilon: float) -> str:
    """epsilon-greedy 选动作。"""
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return ACTIONS[int(np.argmax(q[s]))]


def save_checkpoint(checkpoints_dir: Path, episode: int, q: dict[tuple[int, int], np.ndarray]) -> None:
    """保存 Q 表 checkpoint。"""
    payload = {
        "episode": episode,
        "q_table": {f"{r},{c}": q[(r, c)].tolist() for (r, c) in q.keys()},
    }
    (checkpoints_dir / f"checkpoint-{episode}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train_q_learning(env: GridWorld, args: argparse.Namespace, checkpoints_dir: Path) -> tuple[dict, list[dict[str, float]]]:
    """执行 Q-learning（TD 控制）并返回 Q 表与训练日志。"""
    q = {s: np.zeros(len(ACTIONS), dtype=np.float64) for s in env.states}
    logs: list[dict[str, float]] = []
    epsilon = args.epsilon_start

    for ep in range(1, args.episodes + 1):
        s = env.reset()
        total_reward = 0.0
        td_errors = []

        for _ in range(args.max_steps_per_episode):
            a = epsilon_greedy_action(q, s, epsilon)
            s_next, reward, done = env.step(a)

            a_idx = ACTION_TO_ID[a]
            td_target = reward if done else reward + args.gamma * float(np.max(q[s_next]))
            td_error = td_target - q[s][a_idx]
            q[s][a_idx] += args.alpha * td_error

            total_reward += reward
            td_errors.append(abs(float(td_error)))
            s = s_next
            if done:
                break

        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
        avg_td_error = float(np.mean(td_errors)) if td_errors else 0.0

        logs.append(
            {
                "episode": float(ep),
                "reward": float(total_reward),
                "avg_td_error": float(avg_td_error),
                "epsilon": float(epsilon),
            }
        )

        if ep % args.log_every_episodes == 0 or ep == 1 or ep == args.episodes:
            print(
                f"[Episode {ep:04d}] reward={total_reward:.4f} "
                f"avg_td_error={avg_td_error:.6f} epsilon={epsilon:.4f}"
            )
        if ep % args.save_every_episodes == 0 or ep == args.episodes:
            save_checkpoint(checkpoints_dir, ep, q)

    return q, logs


def extract_policy(env: GridWorld, q: dict[tuple[int, int], np.ndarray]) -> dict[tuple[int, int], str]:
    """从 Q 表提取贪心策略。"""
    policy: dict[tuple[int, int], str] = {}
    for s in env.states:
        if s in env.terminal:
            policy[s] = "T"
        else:
            policy[s] = ACTIONS[int(np.argmax(q[s]))]
    return policy


def export_artifacts(
    env: GridWorld,
    q: dict[tuple[int, int], np.ndarray],
    policy: dict[tuple[int, int], str],
    logs: list[dict[str, float]],
    output_dir: Path,
    models_dir: Path,
) -> Path:
    """导出训练日志、Q 表、策略和可视化结果。"""
    metrics_dir = output_dir / "td_learning_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 日志导出。
    (metrics_dir / "episode_log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    with (metrics_dir / "episode_log.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "avg_td_error", "epsilon"])
        writer.writeheader()
        writer.writerows(logs)

    # 模型导出（Q 表与策略）。
    q_payload = {f"{r},{c}": q[(r, c)].tolist() for (r, c) in q.keys()}
    p_payload = {f"{r},{c}": policy[(r, c)] for (r, c) in sorted(policy.keys())}
    (metrics_dir / "q_table.json").write_text(json.dumps(q_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (metrics_dir / "policy.json").write_text(json.dumps(p_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (models_dir / "td_q_table.json").write_text(json.dumps(q_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (models_dir / "td_policy.json").write_text(json.dumps(p_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 可视化：奖励曲线、TD 误差曲线、epsilon 曲线、策略热力图。
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"`matplotlib` is required for visualization: {exc}") from exc

    episodes = [x["episode"] for x in logs]
    rewards = [x["reward"] for x in logs]
    errors = [x["avg_td_error"] for x in logs]
    epsilons = [x["epsilon"] for x in logs]

    value_grid = np.full((env.rows, env.cols), np.nan, dtype=np.float64)
    for r in range(env.rows):
        for c in range(env.cols):
            s = (r, c)
            if s in env.walls:
                continue
            value_grid[r, c] = float(np.max(q[s]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(episodes, rewards, alpha=0.8)
    axes[0, 0].set_title("Episode Reward")
    axes[0, 0].set_xlabel("episode")
    axes[0, 0].set_ylabel("reward")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(episodes, errors, color="#d62728", alpha=0.8)
    axes[0, 1].set_title("Average TD Error")
    axes[0, 1].set_xlabel("episode")
    axes[0, 1].set_ylabel("avg |td error|")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(episodes, epsilons, color="#2ca02c", alpha=0.8)
    axes[1, 0].set_title("Epsilon Decay")
    axes[1, 0].set_xlabel("episode")
    axes[1, 0].set_ylabel("epsilon")
    axes[1, 0].grid(True, alpha=0.3)

    im = axes[1, 1].imshow(value_grid, cmap="viridis")
    axes[1, 1].set_title("Greedy Policy + State Value")
    axes[1, 1].set_xlabel("col")
    axes[1, 1].set_ylabel("row")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    for r in range(env.rows):
        for c in range(env.cols):
            s = (r, c)
            if s in env.walls:
                txt = "W"
            elif s == env.goal:
                txt = "G"
            elif s in env.traps:
                txt = "X"
            else:
                txt = ARROW.get(policy.get(s, "U"), "?")
            axes[1, 1].text(c, r, txt, ha="center", va="center", color="white", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(metrics_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    summary = {
        "episodes": len(logs),
        "final_reward": rewards[-1] if rewards else None,
        "best_reward": max(rewards) if rewards else None,
        "final_avg_td_error": errors[-1] if errors else None,
        "final_epsilon": epsilons[-1] if epsilons else None,
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics_dir


def main() -> None:
    """主流程入口：训练 TD Learning 并导出可视化结果。"""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    code_dir = Path(__file__).resolve().parent
    module_dir = code_dir.parent
    layout = ensure_layout_dirs(module_dir=module_dir, output_arg=args.output_dir)

    (layout["output"] / "td_learning_run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    env = GridWorld(args)
    q, logs = train_q_learning(env=env, args=args, checkpoints_dir=layout["checkpoints"])
    policy = extract_policy(env, q)
    metrics_dir = export_artifacts(
        env=env,
        q=q,
        policy=policy,
        logs=logs,
        output_dir=layout["output"],
        models_dir=layout["models"],
    )
    print(f"TD Learning done. Visualization exported to: {metrics_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
