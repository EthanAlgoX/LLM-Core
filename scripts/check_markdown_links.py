#!/usr/bin/env python3
"""
Check local markdown links (relative paths and heading anchors).

Usage:
  python scripts/check_markdown_links.py README.md modules docs
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

LINK_RE = re.compile(r"!?\[[^\]]+\]\(([^)]+)\)")
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*$")
FENCE_RE = re.compile(r"^\s*```")
HTML_ANCHOR_RE = re.compile(r"<a\s+(?:id|name)=['\"]([^'\"]+)['\"]", re.IGNORECASE)

SKIP_SCHEMES = (
    "http://",
    "https://",
    "mailto:",
    "tel:",
    "data:",
    "javascript:",
)

MARKDOWN_EXTS = {".md", ".markdown", ".mdx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local markdown links.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["README.md", "modules", "docs"],
        help="Files or directories to scan.",
    )
    return parser.parse_args()


def normalize_fragment(fragment: str) -> str:
    return unquote(fragment).strip().lower()


def slugify_heading(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"`+", "", text)
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "-", text).strip("-")
    return text


def collect_markdown_files(paths: list[str], root: Path) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()

    for raw in paths:
        p = (root / raw).resolve()
        if not p.exists():
            continue
        if p.is_file():
            if p.suffix.lower() in MARKDOWN_EXTS and p not in seen:
                files.append(p)
                seen.add(p)
            continue
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix.lower() in MARKDOWN_EXTS and f not in seen:
                files.append(f)
                seen.add(f)
    return files


def build_anchor_index(path: Path, cache: dict[Path, set[str]]) -> set[str]:
    if path in cache:
        return cache[path]

    anchors: set[str] = set()
    counts: dict[str, int] = {}
    in_fence = False

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        for match in HTML_ANCHOR_RE.findall(line):
            anchors.add(normalize_fragment(match))

        heading = HEADING_RE.match(line)
        if not heading:
            continue
        slug = slugify_heading(heading.group(2))
        if not slug:
            continue
        count = counts.get(slug, 0)
        anchor = slug if count == 0 else f"{slug}-{count}"
        counts[slug] = count + 1
        anchors.add(anchor)

    cache[path] = anchors
    return anchors


def extract_link_target(raw: str) -> str:
    target = raw.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    if " " in target:
        target = target.split(" ", 1)[0]
    return target


def should_skip(target: str) -> bool:
    lower = target.lower()
    return lower.startswith(SKIP_SCHEMES)


def resolve_target(source_file: Path, target: str) -> tuple[Path, str]:
    path_part, fragment = (target.split("#", 1) + [""])[:2] if "#" in target else (target, "")
    if path_part == "":
        target_file = source_file
    else:
        target_file = (source_file.parent / unquote(path_part)).resolve()
    return target_file, normalize_fragment(fragment)


def check_file(path: Path, anchor_cache: dict[Path, set[str]]) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    in_fence = False

    for line_no, line in enumerate(text.splitlines(), start=1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        # Ignore inline code spans to reduce false positives.
        clean_line = re.sub(r"`[^`]*`", "", line)

        for raw_link in LINK_RE.findall(clean_line):
            target = extract_link_target(raw_link)
            if not target or should_skip(target):
                continue

            target_file, fragment = resolve_target(path, target)
            if not target_file.exists():
                errors.append(f"{path}:{line_no}: missing target '{target}'")
                continue

            if fragment:
                if target_file.suffix.lower() in MARKDOWN_EXTS:
                    anchors = build_anchor_index(target_file, anchor_cache)
                    if fragment not in anchors:
                        errors.append(
                            f"{path}:{line_no}: missing anchor '#{fragment}' in '{target_file}'"
                        )
    return errors


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()
    files = collect_markdown_files(args.paths, root)

    if not files:
        print("No markdown files found for the given paths.")
        return 0

    anchor_cache: dict[Path, set[str]] = {}
    all_errors: list[str] = []

    for f in files:
        all_errors.extend(check_file(f, anchor_cache))

    if all_errors:
        print("Markdown link check failed:")
        for err in all_errors:
            print(f"- {err}")
        return 1

    print(f"Markdown link check passed: {len(files)} files checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
