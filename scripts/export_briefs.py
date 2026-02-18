#!/usr/bin/env python3
"""
批量导出所有模块的面试口述稿到统一目录，便于考前速读。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run  # noqa: E402
from scripts.interview_brief import build_brief, load_summary  # noqa: E402


def build_default_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export interview briefs for all modules.")
    parser.add_argument(
        "--output-dir",
        default="output/interview_briefs",
        help="输出目录（默认：output/interview_briefs）",
    )
    return parser.parse_known_args()[0]


def main() -> None:
    args = build_default_args()
    out_dir = (ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# 面试口述稿索引", ""]
    for module in sorted(run.MODULES.keys()):
        summary_path, data = load_summary(module=module, summary_arg="")
        text = build_brief(module=module, summary_path=summary_path, data=data)

        file_path = out_dir / f"{module}.md"
        file_path.write_text(text, encoding="utf-8")
        lines.append(f"- `{module}`: {file_path}")

    index = out_dir / "README.md"
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Exported briefs to: {out_dir}")
    print(f"Index: {index}")


if __name__ == "__main__":
    main()
