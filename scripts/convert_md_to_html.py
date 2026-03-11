#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

import markdown
from bs4 import BeautifulSoup


MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdx"}
IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    ".conda",
    "__pycache__",
    "node_modules",
    "html",
}
REMOTE_SCHEMES = {"http", "https", "mailto", "tel", "data"}
ASSET_SOURCE_DIR = Path(__file__).resolve().parent / "site_assets"
ASSET_OUTPUT_DIRNAME = "_site"

CALLOUT_STYLES = {
    "TIP": ("tip", "Tip"),
    "NOTE": ("note", "Note"),
    "INFO": ("note", "Info"),
    "WARNING": ("warning", "Warning"),
    "CAUTION": ("caution", "Caution"),
}

GROUP_ORDER = [
    "home",
    "guides",
    "01_foundation_rl",
    "02_architecture",
    "03_alignment",
    "04_advanced_topics",
    "05_engineering",
    "06_agent",
    "07_classic_models",
    "reports",
]

GROUP_LABELS = {
    "home": "Overview",
    "guides": "Guides",
    "01_foundation_rl": "RL Foundation",
    "02_architecture": "Architecture",
    "03_alignment": "Alignment",
    "04_advanced_topics": "Advanced Topics",
    "05_engineering": "Engineering",
    "06_agent": "Agent",
    "07_classic_models": "Classic Models",
    "reports": "Reports",
}

GROUP_DESCRIPTIONS = {
    "home": "Project overview and entry points.",
    "guides": "Navigation, terminology, style, and learning paths.",
    "01_foundation_rl": "Reinforcement learning basics, estimation, and policy signals.",
    "02_architecture": "Transformer internals, multimodality, and generative architectures.",
    "03_alignment": "SFT, preference optimization, and RLHF post-training workflows.",
    "04_advanced_topics": "Offline RL and more specialized follow-up topics.",
    "05_engineering": "Systems, kernels, parallelism, and inference optimization.",
    "06_agent": "Agent orchestration, memory, and framework patterns.",
    "07_classic_models": "Representative model case studies and retrospectives.",
    "reports": "Generated reports and technical brief artifacts.",
}

INLINE_MATH_RE = re.compile(r"(?<!\\)\$([^\n$]+?)(?<!\\)\$")
BLOCK_MATH_RE = re.compile(r"(?<!\\)\$\$([\s\S]+?)(?<!\\)\$\$")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <meta name="description" content="{description}" />
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cdefs%3E%3ClinearGradient id='g' x1='0' x2='1' y1='0' y2='1'%3E%3Cstop offset='0' stop-color='%23b76436'/%3E%3Cstop offset='1' stop-color='%230a6d67'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='64' height='64' rx='18' fill='url(%23g)'/%3E%3Ctext x='50%25' y='54%25' text-anchor='middle' font-family='Arial, sans-serif' font-size='28' font-weight='700' fill='white'%3ELL%3C/text%3E%3C/svg%3E" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link rel="preconnect" href="https://cdn.jsdelivr.net" />
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Manrope:wght@500;700;800&family=Noto+Sans+SC:wght@400;500;700;800&family=Noto+Serif+SC:wght@700;900&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css" />
  <link rel="stylesheet" href="{assets_href}site.css" />
  <script>
    (() => {{
      let theme = "light";
      try {{
        const stored = localStorage.getItem("llm-core-theme");
        theme = stored === "dark" ? "dark" : "light";
      }} catch (_error) {{}}
      document.documentElement.dataset.theme = theme;
    }})();
  </script>
</head>
<body data-site-path="{page_path}">
  <div class="scroll-progress"><div class="scroll-progress-bar" id="scroll-progress-bar"></div></div>
  <header class="site-header">
    <div class="site-header-inner">
      <div class="brand-row">
        <button class="nav-toggle" id="nav-toggle" type="button" aria-expanded="false" aria-controls="site-sidebar">Menu</button>
        <a class="brand-link" href="{home_href}">
          <span class="brand-mark">LLM</span>
          <span class="brand-copy">
            <strong>LLM-Core</strong>
            <small>Documentation Site</small>
          </span>
        </a>
      </div>
      <div class="header-actions">
        <button class="search-button" id="search-button" type="button">Search</button>
        <button class="theme-button" id="theme-toggle" type="button" aria-pressed="false">Theme</button>
        <a class="source-button" href="{source_href}">Markdown</a>
      </div>
    </div>
  </header>
  <div class="site-layout">
    <aside class="site-sidebar" id="site-sidebar">
      <div class="sidebar-panel">
        <div class="sidebar-head">
          <p class="sidebar-kicker">{section_label}</p>
          <h2 class="sidebar-title">{title}</h2>
          <p class="sidebar-summary">{description}</p>
        </div>
        <div class="sidebar-search">
          <label class="search-label" for="site-search-input">Search docs</label>
          <input id="site-search-input" class="search-input" type="search" placeholder="Search title, topic, summary" autocomplete="off" />
          <div class="search-results" id="search-results"></div>
        </div>
        <nav class="site-nav">
{nav_html}
        </nav>
      </div>
    </aside>
    <main class="page-column">
      <nav class="breadcrumbs">{breadcrumbs_html}</nav>
      <article class="article-shell">
        <header class="hero">
          <div class="hero-meta">
            <span class="hero-pill">{section_label}</span>
            <span class="hero-pill ghost">{source_path}</span>
          </div>
          <h1 class="hero-title">{title}</h1>
          <p class="hero-description">{description}</p>
        </header>
{showcase_html}
        <article class="article-content" id="article-content">
{content}
        </article>
{pagination_html}
      </article>
    </main>
    <aside class="toc-panel">
      <div class="toc-card">
        <p class="toc-title">On This Page</p>
        <nav class="toc-nav" id="toc-nav"></nav>
      </div>
    </aside>
  </div>
  <script>window.__LLM_CORE_PAGE__ = {{"path": "{page_path}"}};</script>
  <script src="{assets_href}site-data.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/contrib/auto-render.min.js"></script>
  <script type="module" src="{assets_href}site.js"></script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch convert Markdown files to a static HTML site.")
    parser.add_argument("--root", default=".", help="Project root to scan. Defaults to current directory.")
    parser.add_argument("--output", default="html", help="Output directory for generated HTML.")
    return parser.parse_args()


def is_ignored(path: Path, output_dir: Path) -> bool:
    for part in path.parts:
        if part in IGNORED_DIRS:
            return True
    try:
        path.relative_to(output_dir)
        return True
    except ValueError:
        return False


def iter_markdown_files(root: Path, output_dir: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in MARKDOWN_EXTENSIONS:
            continue
        if is_ignored(path.relative_to(root), output_dir.relative_to(root)):
            continue
        yield path


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate_text(text: str, limit: int = 120) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def load_asset_source(name: str) -> str:
    asset_path = ASSET_SOURCE_DIR / name
    return asset_path.read_text(encoding="utf-8")


def protect_code_segments(text: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}

    def store(match: re.Match[str]) -> str:
        token = f"@@CODE_{len(protected)}@@"
        protected[token] = match.group(0)
        return token

    fenced = re.compile(r"(?ms)^([`~]{3,})[^\n]*\n.*?^\1[ \t]*$")
    inline = re.compile(r"(?s)(`+)(.+?)\1")
    text = fenced.sub(store, text)
    text = inline.sub(store, text)
    return text, protected


def replace_display_math(text: str) -> tuple[str, dict[str, str]]:
    replacements: dict[str, str] = {}
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    i = 0

    def make_token(formula: str, indent: str, newline: str = "\n") -> str:
        token = f"@@MATH_BLOCK_{len(replacements)}@@"
        replacements[token] = f'<div class="math-display">\\[\n{html.escape(formula.strip())}\n\\]</div>'
        return f"{indent}{token}{newline}"

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped.startswith("$$"):
            output.append(line)
            i += 1
            continue

        indent = line[: len(line) - len(line.lstrip(" \t"))]
        newline = "\n" if line.endswith("\n") else ""

        if stripped != "$$" and stripped.endswith("$$"):
            output.append(make_token(stripped[2:-2].strip(), indent, newline))
            i += 1
            continue

        if stripped == "$$":
            collected: list[str] = []
            j = i + 1
            while j < len(lines):
                current = lines[j]
                if current.strip() == "$$":
                    output.append(make_token("".join(collected), indent, "\n"))
                    i = j + 1
                    break
                collected.append(current)
                j += 1
            else:
                output.append(line)
                i += 1
            continue

        output.append(line)
        i += 1

    return "".join(output), replacements


def replace_inline_math(text: str) -> tuple[str, dict[str, str]]:
    replacements: dict[str, str] = {}
    result: list[str] = []
    i = 0
    length = len(text)

    while i < length:
        ch = text[i]
        if ch != "$" or (i > 0 and text[i - 1] == "\\") or (i + 1 < length and text[i + 1] == "$"):
            result.append(ch)
            i += 1
            continue

        j = i + 1
        found = False
        while j < length and text[j] != "\n":
            if text[j] == "$" and text[j - 1] != "\\" and (j + 1 == length or text[j + 1] != "$"):
                formula = text[i + 1 : j].strip()
                if formula:
                    token = f"@@MATH_INLINE_{len(replacements)}@@"
                    replacements[token] = f'<span class="math-inline">\\({html.escape(formula)}\\)</span>'
                    result.append(token)
                    i = j + 1
                    found = True
                break
            j += 1

        if not found:
            result.append(ch)
            i += 1

    return "".join(result), replacements


def restore_code_segments(text: str, protected: dict[str, str]) -> str:
    for token, original in protected.items():
        text = text.replace(token, original)
    return text


def restore_math_placeholders(fragment: str, replacements: dict[str, str]) -> str:
    for token, rendered in replacements.items():
        fragment = fragment.replace(f"<p>{token}</p>", rendered)
        fragment = fragment.replace(token, rendered)
    return fragment


def preprocess_markdown(text: str) -> tuple[str, dict[str, str]]:
    protected_text, protected = protect_code_segments(text)
    protected_text, block_math = replace_display_math(protected_text)
    protected_text, inline_math = replace_inline_math(protected_text)
    restored = restore_code_segments(protected_text, protected)
    return restored, {**block_math, **inline_math}


def rewrite_local_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme.lower() in REMOTE_SCHEMES or parsed.netloc or not parsed.path:
        return url
    path = Path(parsed.path)
    if path.suffix.lower() in MARKDOWN_EXTENSIONS:
        parsed = parsed._replace(path=str(path.with_suffix(".html")))
        return urlunsplit(parsed)
    return url


def extract_local_targets(markdown_text: str, source_path: Path) -> set[Path]:
    targets: set[Path] = set()
    for match in re.finditer(r"!\[[^\]]*]\(([^)]+)\)|\[[^\]]*]\(([^)]+)\)", markdown_text):
        raw = match.group(1) or match.group(2)
        url = raw.strip().strip("<>").split(maxsplit=1)[0]
        parsed = urlsplit(url)
        if parsed.scheme.lower() in REMOTE_SCHEMES or parsed.netloc or not parsed.path:
            continue
        target = (source_path.parent / parsed.path).resolve()
        if target.is_file() and target.suffix.lower() not in MARKDOWN_EXTENSIONS:
            targets.add(target)
    return targets


def copy_local_targets(targets: Iterable[Path], root: Path, output_dir: Path) -> None:
    for target in targets:
        try:
            relative = target.relative_to(root)
        except ValueError:
            continue
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, destination)


def enhance_blockquotes(soup: BeautifulSoup) -> None:
    for blockquote in soup.find_all("blockquote"):
        first_p = blockquote.find("p")
        if first_p is None:
            continue

        raw_html = first_p.decode_contents()
        match = re.match(r"\s*\[!(\w+)\]\s*(?:<br\s*/?>)?\s*", raw_html, flags=re.IGNORECASE)
        if not match:
            continue

        callout_key = match.group(1).upper()
        style, label = CALLOUT_STYLES.get(callout_key, ("note", callout_key.title()))
        cleaned_html = re.sub(
            r"\s*\[!\w+\]\s*(?:<br\s*/?>)?\s*",
            "",
            raw_html,
            count=1,
            flags=re.IGNORECASE,
        )
        first_p.clear()
        cleaned_fragment = BeautifulSoup(cleaned_html, "html.parser")
        for node in list(cleaned_fragment.contents):
            first_p.append(node)

        blockquote["class"] = [*blockquote.get("class", []), "callout", f"callout-{style}"]
        blockquote["data-callout-label"] = label


def enhance_media(soup: BeautifulSoup) -> None:
    for paragraph in soup.find_all("p"):
        children = [child for child in paragraph.children if getattr(child, "name", None) or str(child).strip()]
        if len(children) != 1:
            continue
        image = children[0]
        if getattr(image, "name", None) != "img":
            continue

        figure = soup.new_tag("figure")
        paragraph.replace_with(figure)
        figure.append(image)

        alt = image.get("alt", "").strip()
        if alt:
            caption = soup.new_tag("figcaption")
            caption.string = alt
            figure.append(caption)


def enhance_tables(soup: BeautifulSoup) -> None:
    for table in soup.find_all("table"):
        parent = table.parent
        if parent and parent.name == "div" and "table-scroll" in parent.get("class", []):
            continue
        wrapper = soup.new_tag("div")
        wrapper["class"] = ["table-scroll"]
        table.wrap(wrapper)


def normalize_math_blocks(soup: BeautifulSoup) -> None:
    for paragraph in soup.find_all("p"):
        contents = list(paragraph.contents)
        math_blocks = [
            child
            for child in contents
            if getattr(child, "name", None) == "div" and "math-display" in child.get("class", [])
        ]
        if not math_blocks:
            continue

        segments: list[list] = [[]]
        extracted_blocks = []
        for child in contents:
            if getattr(child, "name", None) == "div" and "math-display" in child.get("class", []):
                extracted_blocks.append(child.extract())
                segments.append([])
            else:
                segments[-1].append(child.extract())

        for index, segment in enumerate(segments):
            if any(str(node).strip() for node in segment):
                new_paragraph = soup.new_tag("p")
                for node in segment:
                    new_paragraph.append(node)
                paragraph.insert_before(new_paragraph)

            if index < len(extracted_blocks):
                paragraph.insert_before(extracted_blocks[index])

        paragraph.decompose()


def enhance_headings(soup: BeautifulSoup) -> None:
    for heading in soup.find_all(re.compile(r"^h[2-6]$")):
        heading_id = heading.get("id")
        if not heading_id:
            continue
        anchor = soup.new_tag("a", href=f"#{heading_id}")
        anchor["class"] = ["heading-anchor"]
        anchor["aria-label"] = "Jump to heading"
        anchor.string = "#"
        heading.append(anchor)


def enhance_code_blocks(soup: BeautifulSoup) -> None:
    for pre in soup.find_all("pre"):
        code = pre.find("code")
        language = "text"
        if code is not None:
            for class_name in code.get("class", []):
                if class_name.startswith("language-"):
                    language = class_name.split("-", 1)[1]
                    break
        pre["data-lang"] = language


def enhance_links(soup: BeautifulSoup) -> None:
    for link in soup.find_all("a"):
        href = link.get("href", "")
        parsed = urlsplit(href)
        if parsed.scheme.lower() in REMOTE_SCHEMES and parsed.scheme.lower() not in {"mailto", "tel"}:
            link["target"] = "_blank"
            link["rel"] = "noreferrer"


def extract_summary_from_soup(soup: BeautifulSoup) -> str:
    first_callout = soup.find(class_="callout")
    if first_callout:
        callout_text = normalize_text(first_callout.get_text(" ", strip=True))
        if callout_text:
            return callout_text

    for paragraph in soup.find_all("p"):
        text = normalize_text(paragraph.get_text(" ", strip=True))
        if len(text) >= 24:
            return text
    return ""


def derive_group_key(source_path: Path) -> str:
    parts = source_path.parts
    if source_path.as_posix() == "README.md":
        return "home"
    if parts[0] == "docs":
        return "guides"
    if parts[0] == "modules" and len(parts) > 1:
        return parts[1]
    if parts[0] == "output":
        return "reports"
    return parts[0]


def derive_section_label(source_path: Path) -> str:
    return GROUP_LABELS.get(derive_group_key(source_path), "Documentation")


def group_order_index(group_key: str) -> int:
    try:
        return GROUP_ORDER.index(group_key)
    except ValueError:
        return len(GROUP_ORDER)


def is_overview_page(source_path: Path) -> bool:
    if source_path.as_posix() == "README.md":
        return True
    if source_path.name == "README.md":
        return True
    if source_path.parent.name and source_path.stem == source_path.parent.name:
        return True
    return False


def page_sort_key(source_path: Path) -> tuple[object, ...]:
    group_key = derive_group_key(source_path)
    return (
        group_order_index(group_key),
        0 if is_overview_page(source_path) else 1,
        source_path.as_posix(),
    )


def relative_site_href(from_site_path: str, to_site_path: str) -> str:
    return Path(os.path.relpath(to_site_path, Path(from_site_path).parent)).as_posix()


def is_literal_math_example(expression: str) -> bool:
    stripped = expression.strip()
    return stripped in {"...", ". . .", r"\ldots"}


def find_unresolved_math_in_html(html_path: Path) -> list[str]:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    for tag in soup(["script", "style", "code", "pre"]):
        tag.decompose()

    text = soup.get_text("\n")
    matches: list[str] = []
    for regex in (BLOCK_MATH_RE, INLINE_MATH_RE):
        for match in regex.finditer(text):
            expression = match.group(1).strip()
            if not expression or is_literal_math_example(expression):
                continue
            matches.append(match.group(0).strip())
    return matches


def validate_generated_html(output_dir: Path) -> list[tuple[Path, list[str]]]:
    unresolved: list[tuple[Path, list[str]]] = []
    for html_path in sorted(output_dir.rglob("*.html")):
        matches = find_unresolved_math_in_html(html_path)
        if matches:
            unresolved.append((html_path, matches))
    return unresolved


def convert_markdown(source_path: Path) -> tuple[str, str, str, set[Path]]:
    raw = source_path.read_text(encoding="utf-8", errors="ignore")
    processed, math_replacements = preprocess_markdown(raw)
    fragment = markdown.markdown(
        processed,
        extensions=[
            "extra",
            "tables",
            "fenced_code",
            "toc",
            "sane_lists",
            "nl2br",
        ],
        output_format="html5",
    )
    fragment = restore_math_placeholders(fragment, math_replacements)
    soup = BeautifulSoup(fragment, "html.parser")

    for tag in soup.find_all(["a", "img"]):
        attr = "href" if tag.name == "a" else "src"
        value = tag.get(attr)
        if not value:
            continue
        tag[attr] = rewrite_local_url(value)

    for code in soup.select("pre > code.language-mermaid"):
        pre = code.parent
        mermaid_div = soup.new_tag("div")
        mermaid_div["class"] = ["mermaid"]
        mermaid_div.string = code.get_text()
        pre.replace_with(mermaid_div)

    title = source_path.stem
    first_heading = soup.find(re.compile(r"^h[1-6]$"))
    if first_heading:
        title = first_heading.get_text(strip=True) or title
        if first_heading.name == "h1":
            first_heading.decompose()

    enhance_blockquotes(soup)
    enhance_media(soup)
    enhance_tables(soup)
    normalize_math_blocks(soup)
    enhance_headings(soup)
    enhance_code_blocks(soup)
    enhance_links(soup)

    summary = extract_summary_from_soup(soup)
    return str(soup), title, summary, extract_local_targets(raw, source_path)


def build_pages(markdown_files: list[Path], root: Path) -> list[dict[str, object]]:
    pages: list[dict[str, object]] = []
    for source_path in sorted(markdown_files, key=lambda path: page_sort_key(path.relative_to(root))):
        relative_source = source_path.relative_to(root)
        content, title, summary, assets = convert_markdown(source_path)
        pages.append(
            {
                "source_path": relative_source,
                "source_path_str": relative_source.as_posix(),
                "site_path": relative_source.with_suffix(".html").as_posix(),
                "title": title,
                "summary": summary or GROUP_DESCRIPTIONS.get(derive_group_key(relative_source), "Documentation page."),
                "content": content,
                "assets": assets,
                "group_key": derive_group_key(relative_source),
                "section_label": derive_section_label(relative_source),
                "is_overview": is_overview_page(relative_source),
                "order_key": page_sort_key(relative_source),
            }
        )
    return pages


def build_group_overviews(pages: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    overviews: dict[str, dict[str, object]] = {}
    for page in pages:
        group_key = str(page["group_key"])
        if group_key not in overviews and bool(page["is_overview"]):
            overviews[group_key] = page
    return overviews


def group_pages(pages: list[dict[str, object]]) -> list[tuple[str, list[dict[str, object]]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for page in pages:
        grouped.setdefault(str(page["group_key"]), []).append(page)

    ordered_keys = sorted(grouped, key=group_order_index)
    return [(group_key, grouped[group_key]) for group_key in ordered_keys]


def render_nav_html(current_page: dict[str, object], pages: list[dict[str, object]]) -> str:
    current_path = str(current_page["site_path"])
    blocks: list[str] = []
    for group_key, group_items in group_pages(pages):
        open_attr = " open" if group_key == current_page["group_key"] or group_key in {"home", "guides"} else ""
        blocks.append(
            f'<details class="nav-group"{open_attr}><summary>{html.escape(GROUP_LABELS.get(group_key, group_key))}'
            f'<span>{len(group_items)}</span></summary>'
        )
        for page in group_items:
            target_path = str(page["site_path"])
            href = html.escape(relative_site_href(current_path, target_path))
            title = html.escape(str(page["title"]))
            active_class = " active" if target_path == current_path else ""
            summary = html.escape(truncate_text(str(page["summary"]), 88))
            blocks.append(
                f'<a class="nav-link{active_class}" href="{href}"><span>{title}</span><small>{summary}</small></a>'
            )
        blocks.append("</details>")
    return "\n".join(f"          {line}" for line in blocks)


def render_breadcrumbs_html(
    current_page: dict[str, object],
    group_overviews: dict[str, dict[str, object]],
) -> str:
    current_path = str(current_page["site_path"])
    crumbs = [f'<a href="{html.escape(relative_site_href(current_path, "README.html"))}">Home</a>']
    group_key = str(current_page["group_key"])
    if group_key != "home":
        overview = group_overviews.get(group_key)
        label = html.escape(GROUP_LABELS.get(group_key, group_key))
        if overview and overview["site_path"] != current_page["site_path"]:
            href = html.escape(relative_site_href(current_path, str(overview["site_path"])))
            crumbs.append(f'<a href="{href}">{label}</a>')
        else:
            crumbs.append(f"<span>{label}</span>")
    crumbs.append(f"<span>{html.escape(str(current_page['title']))}</span>")
    return "\n".join(crumbs)


def render_pagination_html(current_page: dict[str, object], pages: list[dict[str, object]]) -> str:
    index = pages.index(current_page)
    prev_page = pages[index - 1] if index > 0 else None
    next_page = pages[index + 1] if index + 1 < len(pages) else None
    if prev_page is None and next_page is None:
        return ""

    cards: list[str] = ['      <footer class="page-footer">', '        <div class="page-footer-grid">']
    for label, page in (("Previous", prev_page), ("Next", next_page)):
        if page is None:
            cards.append('          <div class="page-card empty"></div>')
            continue
        href = html.escape(relative_site_href(str(current_page["site_path"]), str(page["site_path"])))
        title = html.escape(str(page["title"]))
        summary = html.escape(str(page["summary"]))
        section = html.escape(str(page["section_label"]))
        cards.append(
            "          "
            f'<a class="page-card" href="{href}"><span class="page-card-label">{label}</span>'
            f"<strong>{title}</strong><small>{section}</small><p>{summary}</p></a>"
        )
    cards.extend(["        </div>", "      </footer>"])
    return "\n".join(cards)


def render_home_showcase(current_page: dict[str, object], pages: list[dict[str, object]]) -> str:
    if str(current_page["site_path"]) != "README.html":
        return ""

    total_groups = len([group_key for group_key, _ in group_pages(pages) if group_key != "home"])
    total_pages = len(pages)
    total_module_pages = len([page for page in pages if str(page["group_key"]).startswith("0")])
    featured_groups = []
    for group_key, group_items in group_pages(pages):
        if group_key in {"home", "reports"}:
            continue
        overview = next((page for page in group_items if bool(page["is_overview"])), group_items[0])
        featured_groups.append(
            {
                "title": GROUP_LABELS.get(group_key, group_key),
                "href": relative_site_href("README.html", str(overview["site_path"])),
                "count": len(group_items),
                "summary": GROUP_DESCRIPTIONS.get(group_key, "Documentation section"),
            }
        )

    highlighted_pages = []
    priority_titles = [
        "Transformer Core (大语言模型架构)",
        "PPO (Proximal Policy Optimization) 强化学习对齐",
        "Generation & Decoding (推理与生成优化)",
        "RL Basics 分类",
        "多模态 VLM (Vision-Language Models)",
        "高效微调 (PEFT: Parameter-Efficient Fine-Tuning)",
    ]
    title_to_page = {str(page["title"]): page for page in pages}
    for title in priority_titles:
        page = title_to_page.get(title)
        if page is not None:
            highlighted_pages.append(page)
    if len(highlighted_pages) < 6:
        for page in pages:
            if page in highlighted_pages or str(page["group_key"]) in {"home", "guides", "reports"}:
                continue
            highlighted_pages.append(page)
            if len(highlighted_pages) == 6:
                break

    learning_tracks = [
        {
            "title": "Interview Sprint",
            "summary": "从 Transformer、Alignment 到 Inference，快速构建高频面试主线。",
            "items": [
                "modules/02_architecture/llm/llm.html",
                "modules/03_alignment/ppo/ppo.html",
                "modules/05_engineering/inference/inference.html",
                "modules/07_classic_models/chatgpt/chatgpt.html",
            ],
        },
        {
            "title": "Engineering Track",
            "summary": "偏系统与落地，把训练、推理、并行和高效微调串成一条路。",
            "items": [
                "modules/02_architecture/02_architecture.html",
                "modules/05_engineering/cuda/cuda.html",
                "modules/05_engineering/inference/inference.html",
                "modules/03_alignment/peft/peft.html",
            ],
        },
        {
            "title": "Research Track",
            "summary": "从 RL 基础延伸到对齐与 Offline RL，适合继续做方法论推演。",
            "items": [
                "modules/01_foundation_rl/01_foundation_rl.html",
                "modules/03_alignment/rlhf/rlhf.html",
                "modules/04_advanced_topics/offline_rl/offline_rl.html",
                "modules/07_classic_models/deepseek_r1/deepseek_r1.html",
            ],
        },
    ]

    cards: list[str] = [
        '        <section class="home-hero-panel">',
        '          <div class="home-hero-copy">',
        '            <p class="showcase-kicker">Knowledge System</p>',
        '            <h2>Turn the document collection into a browsable learning site</h2>',
        '            <p>Use the left navigation for systematic reading, the built-in search for concept jumps, and the grouped modules below for continuous exploration.</p>',
        '          </div>',
        '          <div class="home-stat-grid">',
        f'            <div class="home-stat-card"><strong>{total_pages}</strong><span>Total Pages</span></div>',
        f'            <div class="home-stat-card"><strong>{total_groups}</strong><span>Main Tracks</span></div>',
        f'            <div class="home-stat-card"><strong>{total_module_pages}</strong><span>Module Docs</span></div>',
        '          </div>',
        '        </section>',
        '        <section class="home-showcase">',
        '          <div class="showcase-intro">',
        '            <p class="showcase-kicker">Site Index</p>',
        '            <h2>Browse by module</h2>',
        "            <p>Jump straight into the main tracks of the LLM-Core knowledge base. Each section card links to its overview page and keeps the full module set one click away.</p>",
        "          </div>",
        '          <div class="showcase-grid">',
    ]

    for group in featured_groups:
        cards.append(
            "            "
            f'<a class="showcase-card" href="{html.escape(group["href"])}"><span class="showcase-card-count">{group["count"]} pages</span>'
            f"<strong>{html.escape(group['title'])}</strong><p>{html.escape(group['summary'])}</p></a>"
        )

    cards.extend(["          </div>", "        </section>"])

    cards.extend(
        [
            '        <section class="home-feature-section">',
            '          <div class="showcase-intro">',
            '            <p class="showcase-kicker">Recommended</p>',
            '            <h2>Recommended reading</h2>',
            "            <p>Start from these pages if you want the shortest route to the most reused concepts in the repository.</p>",
            "          </div>",
            '          <div class="featured-grid">',
        ]
    )
    for page in highlighted_pages:
        cards.append(
            "            "
            f'<a class="featured-card" href="{html.escape(relative_site_href("README.html", str(page["site_path"])))}">'
            f'<span class="featured-card-group">{html.escape(str(page["section_label"]))}</span>'
            f"<strong>{html.escape(str(page['title']))}</strong>"
            f"<p>{html.escape(truncate_text(str(page['summary']), 128))}</p></a>"
        )
    cards.extend(["          </div>", "        </section>"])

    cards.extend(
        [
            '        <section class="home-feature-section">',
            '          <div class="showcase-intro">',
            '            <p class="showcase-kicker">Learning Paths</p>',
            '            <h2>Pick a reading route</h2>',
            "            <p>Choose a route based on your current goal. Each path chains overview pages and high-signal modules into a coherent pass.</p>",
            "          </div>",
            '          <div class="track-grid">',
        ]
    )

    page_by_path = {str(page["site_path"]): page for page in pages}
    for track in learning_tracks:
        cards.append(
            "            "
            f'<section class="track-card"><p class="track-card-kicker">{html.escape(track["title"])}</p>'
            f"<p class=\"track-card-summary\">{html.escape(track['summary'])}</p><ol class=\"track-list\">"
        )
        for item in track["items"]:
            page = page_by_path.get(item)
            if page is None:
                continue
            cards.append(
                "              "
                f'<li><a href="{html.escape(relative_site_href("README.html", item))}">{html.escape(str(page["title"]))}</a></li>'
            )
        cards.append("            </ol></section>")

    cards.extend(["          </div>", "        </section>"])
    return "\n".join(cards)


def write_shared_assets(output_dir: Path, pages: list[dict[str, object]]) -> None:
    asset_output_dir = output_dir / ASSET_OUTPUT_DIRNAME
    asset_output_dir.mkdir(parents=True, exist_ok=True)

    (asset_output_dir / "site.css").write_text(load_asset_source("site.css"), encoding="utf-8")
    (asset_output_dir / "site.js").write_text(load_asset_source("site.js"), encoding="utf-8")

    search_pages = [
        {
            "title": str(page["title"]),
            "summary": truncate_text(str(page["summary"]), 120),
            "group": str(page["section_label"]),
            "group_key": str(page["group_key"]),
            "path": str(page["site_path"]),
            "source": str(page["source_path_str"]),
            "is_overview": bool(page["is_overview"]),
            "order": list(page["order_key"]),
        }
        for page in pages
    ]
    site_data = {"pages": search_pages}
    payload = "window.__LLM_CORE_SITE__ = " + json.dumps(site_data, ensure_ascii=False) + ";"
    (asset_output_dir / "site-data.js").write_text(payload, encoding="utf-8")


def write_html(
    page: dict[str, object],
    pages: list[dict[str, object]],
    root: Path,
    output_dir: Path,
    group_overviews: dict[str, dict[str, object]],
) -> tuple[Path, set[Path]]:
    destination = output_dir / str(page["site_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)

    assets_href = Path(os.path.relpath(output_dir / ASSET_OUTPUT_DIRNAME, destination.parent)).as_posix() + "/"
    source_href = Path(os.path.relpath(root / str(page["source_path_str"]), destination.parent)).as_posix()
    home_href = relative_site_href(str(page["site_path"]), "README.html")
    document = HTML_TEMPLATE.format(
        title=html.escape(str(page["title"])),
        description=html.escape(str(page["summary"])),
        assets_href=html.escape(assets_href),
        page_path=html.escape(str(page["site_path"])),
        home_href=html.escape(home_href),
        source_href=html.escape(source_href),
        section_label=html.escape(str(page["section_label"])),
        source_path=html.escape(str(page["source_path_str"])),
        nav_html=render_nav_html(page, pages),
        breadcrumbs_html=render_breadcrumbs_html(page, group_overviews),
        showcase_html=render_home_showcase(page, pages),
        content=str(page["content"]),
        pagination_html=render_pagination_html(page, pages),
    )
    destination.write_text(document, encoding="utf-8")
    return destination, set(page["assets"])


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = (root / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_files = list(iter_markdown_files(root, output_dir))
    if not markdown_files:
        print("No markdown files found.")
        return 0

    pages = build_pages(markdown_files, root)
    group_overviews = build_group_overviews(pages)
    write_shared_assets(output_dir, pages)

    total_assets = 0
    for page in pages:
        _, assets = write_html(page, pages, root, output_dir, group_overviews)
        copy_local_targets(assets, root, output_dir)
        total_assets += len(assets)

    unresolved = validate_generated_html(output_dir)
    if unresolved:
        print("Generated HTML contains unresolved math expressions:")
        for html_path, matches in unresolved:
            print(f"- {html_path.relative_to(root)}")
            for snippet in matches[:3]:
                print(f"  {snippet}")
        print(
            f"Done with errors. Generated {len(markdown_files)} HTML files, copied {total_assets} local assets, "
            f"and found unresolved math in {len(unresolved)} files."
        )
        return 1

    print(
        f"Done. Generated {len(markdown_files)} HTML files, copied {total_assets} local assets, "
        "validated math rendering markers, and wrote site assets."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
