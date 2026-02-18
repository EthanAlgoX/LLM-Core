#!/usr/bin/env python3
"""
MDP（马尔可夫决策过程）最小可运行示例：GridWorld + Value Iteration。

一、MDP 原理（面向实现）
1) 状态 s：智能体所在位置（网格坐标）。
2) 动作 a：上/下/左/右移动。
3) 转移 P(s'|s,a)：执行动作后到达下一状态的概率。
4) 奖励 R(s,a,s')：每一步的即时反馈（到达目标奖励高，陷阱奖励低）。
5) 目标：找到最优策略 pi*(s)，使长期折扣回报最大。

新人阅读顺序（建议）
1) 先看 `build_default_args`：明确可调参数和默认值。
2) 再看 `main`：把握执行主链路（准备 -> 训练/推理 -> 导出）。
3) 最后看可视化导出函数（如 `export_artifacts`）理解输出文件。

二、代码框架（从入口到结果）
1) `build_default_args`：读取网格与迭代参数。
2) `build_grid_mdp`：构建状态空间、转移和奖励规则。
3) `value_iteration`：求解状态价值函数 V(s)。
4) `extract_policy`：由 V(s) 提取贪心策略。
5) `export_artifacts`：导出 CSV/JSON/可视化图。
6) `main`：串联完整流程。

用法：
  python code/mdp.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


ACTION_LIST = ["U", "D", "L", "R"]
ACTION_DELTA = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
ARROW = {"U": "↑", "D": "↓", "L": "←", "R": "→"}


def build_default_args() -> argparse.Namespace:
    """解析命令行参数，返回 MDP 求解配置。"""
    parser = argparse.ArgumentParser(description="Run MDP value iteration and export visualization artifacts.")
    parser.add_argument("--output-dir", default="output")

    # 环境参数：默认 5x5 网格，含 1 个目标和 2 个陷阱。
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--goal", default="4,4", help="目标位置，格式: row,col")
    parser.add_argument("--traps", default="1,3;3,1", help="陷阱位置列表，格式: r1,c1;r2,c2")
    parser.add_argument("--walls", default="2,2", help="墙体位置列表，格式: r1,c1;r2,c2")
    parser.add_argument("--start", default="0,0", help="起点位置，格式: row,col")

    # 奖励与转移参数。
    parser.add_argument("--step-reward", type=float, default=-0.04)
    parser.add_argument("--goal-reward", type=float, default=1.0)
    parser.add_argument("--trap-reward", type=float, default=-1.0)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--slip-prob", type=float, default=0.1, help="动作偏移概率（左右偏移）。")

    # 迭代参数。
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--save-every-iters", type=int, default=50)
    return parser.parse_known_args([])[0]


def resolve_output_dir(base_dir: Path, output_dir: str) -> Path:
    """解析输出目录：相对路径按 mdp 目录解析，绝对路径原样使用。"""
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


def parse_coord(s: str) -> tuple[int, int]:
    """解析单个坐标字符串 `r,c`。"""
    r_str, c_str = s.split(",")
    return int(r_str.strip()), int(c_str.strip())


def parse_coord_list(s: str) -> set[tuple[int, int]]:
    """解析坐标列表字符串 `r1,c1;r2,c2`。"""
    s = s.strip()
    if not s:
        return set()
    return {parse_coord(item) for item in s.split(";") if item.strip()}


def in_bounds(r: int, c: int, rows: int, cols: int) -> bool:
    """检查坐标是否在网格范围内。"""
    return 0 <= r < rows and 0 <= c < cols


def build_grid_mdp(args: argparse.Namespace) -> dict[str, Any]:
    """构建网格 MDP 定义。"""
    rows, cols = args.rows, args.cols
    goal = parse_coord(args.goal)
    start = parse_coord(args.start)
    traps = parse_coord_list(args.traps)
    walls = parse_coord_list(args.walls)

    all_cells = {(r, c) for r in range(rows) for c in range(cols)}
    states = sorted(all_cells - walls)
    terminal = {goal} | traps

    if goal in walls or start in walls:
        raise ValueError("goal/start cannot be inside walls.")
    if not in_bounds(*goal, rows, cols) or not in_bounds(*start, rows, cols):
        raise ValueError("goal/start out of bounds.")

    # 动作左右偏移映射，用于模拟 slip。
    side_actions = {
        "U": ("L", "R"),
        "D": ("R", "L"),
        "L": ("D", "U"),
        "R": ("U", "D"),
    }

    def move(s: tuple[int, int], action: str) -> tuple[int, int]:
        if s in terminal:
            return s
        dr, dc = ACTION_DELTA[action]
        nr, nc = s[0] + dr, s[1] + dc
        if not in_bounds(nr, nc, rows, cols) or (nr, nc) in walls:
            return s
        return nr, nc

    def reward(s_next: tuple[int, int]) -> float:
        if s_next == goal:
            return args.goal_reward
        if s_next in traps:
            return args.trap_reward
        return args.step_reward

    transitions: dict[tuple[tuple[int, int], str], list[tuple[float, tuple[int, int], float]]] = {}
    for s in states:
        for a in ACTION_LIST:
            if s in terminal:
                transitions[(s, a)] = [(1.0, s, 0.0)]
                continue

            main_p = 1.0 - args.slip_prob
            side_p = args.slip_prob / 2.0
            candidates = [
                (main_p, move(s, a)),
                (side_p, move(s, side_actions[a][0])),
                (side_p, move(s, side_actions[a][1])),
            ]
            transitions[(s, a)] = [(p, s_next, reward(s_next)) for p, s_next in candidates]

    return {
        "rows": rows,
        "cols": cols,
        "start": start,
        "goal": goal,
        "traps": traps,
        "walls": walls,
        "states": states,
        "terminal": terminal,
        "transitions": transitions,
    }


def value_iteration(mdp: dict[str, Any], gamma: float, max_iters: int, tol: float) -> tuple[dict, list[dict[str, float]]]:
    """执行值迭代，返回最优价值函数及收敛日志。"""
    states = mdp["states"]
    terminal = mdp["terminal"]
    transitions = mdp["transitions"]

    v = {s: 0.0 for s in states}
    logs: list[dict[str, float]] = []

    for it in range(1, max_iters + 1):
        delta = 0.0
        v_new = dict(v)
        for s in states:
            if s in terminal:
                v_new[s] = 0.0
                continue

            q_values = []
            for a in ACTION_LIST:
                q_sa = 0.0
                for p, s_next, r in transitions[(s, a)]:
                    q_sa += p * (r + gamma * v[s_next])
                q_values.append(q_sa)
            v_new[s] = max(q_values)
            delta = max(delta, abs(v_new[s] - v[s]))

        v = v_new
        logs.append({"iter": float(it), "delta": float(delta)})
        if delta < tol:
            break
    return v, logs


def extract_policy(mdp: dict[str, Any], v: dict, gamma: float) -> dict:
    """依据价值函数提取贪心策略。"""
    states = mdp["states"]
    terminal = mdp["terminal"]
    transitions = mdp["transitions"]

    policy: dict[tuple[int, int], str] = {}
    for s in states:
        if s in terminal:
            policy[s] = "T"
            continue
        best_a = None
        best_q = -1e18
        for a in ACTION_LIST:
            q_sa = 0.0
            for p, s_next, r in transitions[(s, a)]:
                q_sa += p * (r + gamma * v[s_next])
            if q_sa > best_q:
                best_q = q_sa
                best_a = a
        policy[s] = best_a or "U"
    return policy


def save_checkpoint(checkpoints_dir: Path, iter_idx: int, v: dict, delta: float) -> None:
    """保存中间迭代状态。"""
    ckpt_path = checkpoints_dir / f"checkpoint-{iter_idx}.json"
    payload = {
        "iter": iter_idx,
        "delta": delta,
        "values": {f"{r},{c}": val for (r, c), val in v.items()},
    }
    ckpt_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def export_artifacts(
    mdp: dict[str, Any],
    v: dict,
    policy: dict,
    logs: list[dict[str, float]],
    output_dir: Path,
    models_dir: Path,
) -> Path:
    """导出日志、价值函数、策略与可视化结果。"""
    metrics_dir = output_dir / "mdp_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 导出迭代日志。
    (metrics_dir / "iteration_log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    with (metrics_dir / "iteration_log.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iter", "delta"])
        writer.writeheader()
        writer.writerows(logs)

    # 导出最终价值函数和策略（同时存入 models 目录，方便复用）。
    value_payload = {f"{r},{c}": float(val) for (r, c), val in v.items()}
    policy_payload = {f"{r},{c}": policy[(r, c)] for (r, c) in sorted(policy.keys())}

    (metrics_dir / "value_function.json").write_text(
        json.dumps(value_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (metrics_dir / "policy.json").write_text(
        json.dumps(policy_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (models_dir / "mdp_value_function.json").write_text(
        json.dumps(value_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (models_dir / "mdp_policy.json").write_text(
        json.dumps(policy_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows, cols = mdp["rows"], mdp["cols"]
    goal = mdp["goal"]
    traps = mdp["traps"]
    walls = mdp["walls"]

    # 生成价值热力图矩阵。
    heat = [[math.nan for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if (r, c) in walls:
                continue
            heat[r][c] = v.get((r, c), 0.0)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        raise RuntimeError(f"`matplotlib` and `numpy` are required for visualization: {exc}") from exc

    arr = np.array(heat, dtype=float)
    deltas = [item["delta"] for item in logs]
    iters = [item["iter"] for item in logs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].imshow(arr, cmap="viridis")
    axes[0].set_title("State Value Heatmap")
    axes[0].set_xlabel("col")
    axes[0].set_ylabel("row")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    for r in range(rows):
        for c in range(cols):
            s = (r, c)
            if s in walls:
                axes[0].text(c, r, "W", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
            elif s == goal:
                axes[0].text(c, r, "G", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
            elif s in traps:
                axes[0].text(c, r, "X", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
            else:
                axes[0].text(c, r, ARROW.get(policy.get(s, "U"), "?"), ha="center", va="center", color="white")

    axes[1].plot(iters, deltas, marker="o")
    axes[1].set_title("Value Iteration Convergence")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("max delta")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale("log")

    fig.tight_layout()
    fig.savefig(metrics_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    summary = {
        "num_states": len(mdp["states"]),
        "num_terminal_states": len(mdp["terminal"]),
        "iterations": len(logs),
        "final_delta": deltas[-1] if deltas else None,
        "goal": f"{goal[0]},{goal[1]}",
        "start": f"{mdp['start'][0]},{mdp['start'][1]}",
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics_dir


def main() -> None:
    """主流程入口：构建 MDP、执行值迭代并导出可视化结果。"""
    print("=== MDP 主流程（学习版）===", flush=True)

    # 步骤 1：读取参数并创建目录结构。
    args = build_default_args()
    code_dir = Path(__file__).resolve().parent
    module_dir = code_dir.parent
    layout = ensure_layout_dirs(module_dir=module_dir, output_arg=args.output_dir)

    # 步骤 2：保存运行配置，方便复现实验。
    (layout["output"] / "mdp_run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 步骤 3：构建环境并执行值迭代，得到价值函数与迭代日志。
    mdp = build_grid_mdp(args)
    v, logs = value_iteration(
        mdp=mdp,
        gamma=args.discount,
        max_iters=args.max_iters,
        tol=args.tol,
    )

    # 步骤 4：按固定间隔保存 checkpoint，便于观察价值函数收敛过程。
    for idx, item in enumerate(logs, start=1):
        if idx % args.save_every_iters == 0 or idx == len(logs):
            save_checkpoint(layout["checkpoints"], idx, v, item["delta"])

    # 步骤 5：从价值函数提取策略并导出可视化产物。
    policy = extract_policy(mdp, v, args.discount)
    metrics_dir = export_artifacts(
        mdp=mdp,
        v=v,
        policy=policy,
        logs=logs,
        output_dir=layout["output"],
        models_dir=layout["models"],
    )
    print(f"MDP done. Visualization exported to: {metrics_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
