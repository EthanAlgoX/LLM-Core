# Documentation Style Guide

This repository is documentation-first. Use this guide to keep docs consistent and easy to maintain.

## Scope

- Applies to `README.md` and all files under `modules/`.
- Applies to new docs under `docs/`.

## Writing Rules

- Keep section titles descriptive and stable.
- Keep one core idea per section.
- Prefer short paragraphs and explicit definitions.
- Define acronyms on first use.
- Use the canonical term from `docs/TERMINOLOGY.md`.

## Link Rules

- Prefer relative links for in-repo references.
- Use heading anchors that match section titles.
- Avoid dead links before committing changes.

## Review Checklist

- Links are valid.
- Terms follow the glossary.
- Navigation tables still map to existing files.
- New topic pages are discoverable from `README.md`.

## Suggested Commands

```bash
python scripts/check_markdown_links.py README.md modules docs
```

