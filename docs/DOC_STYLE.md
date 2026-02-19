# Documentation Style Guide

本仓库是文档优先仓库。所有模块文档按统一结构编写，确保读者能从概念快速走到工程实现。

## Scope

- 适用于 `README.md` 与 `modules/` 下所有 Markdown 文档。
- 适用于 `docs/` 下规范与模板文档。

## Required Structure

模块文档推荐遵循以下顺序：

1. 一句话通俗理解
2. 定义与目标（它解决什么问题）
3. 适用场景与边界（什么时候用/不用）
4. 关键步骤（流程图或编号步骤）
5. 关键公式（含符号说明）
6. 关键步骤代码（纯文档示例，突出核心逻辑）
7. 工程实现要点（性能、显存、并行、稳定性）
8. 常见错误与排查
9. 与相近方法对比
10. 参考资料（论文/官方文档）

模板见：`docs/MODULE_DOC_TEMPLATE.md`。

## Writing Rules

- 每节只讲一个核心问题，避免重复叙述。
- 先给结论，再给解释，再给例子。
- 首次出现缩写要展开全称（如 LLM、VLM、RLHF）。
- 术语以 `docs/TERMINOLOGY.md` 为准。
- 代码示例只保留关键步骤，不引用历史脚本路径。
- 公式优先使用 GitHub 兼容写法：行内 `$...$`、块级 `$$...$$`。

## Navigation Rules

- 章节入口优先放在模块总览页（如 `01_foundation_rl.md`）。
- 仓库级导航统一维护在 `docs/NAVIGATION.md`。
- 学习路线统一维护在 `docs/LEARNING_PATH.md`。

## Link Rules

- 仓库内链接使用相对路径。
- 提交前必须通过链接检查。
- 标题锚点应与实际标题一致。

## Review Checklist

- 结构是否符合 `Required Structure`。
- 公式是否可被 GitHub 正常渲染且给出符号含义。
- 代码块是否体现关键步骤，且不引用历史 code 文件。
- 模块总览页是否包含子模块导航。
- README 是否可导航到对应模块。
- 链接检查是否通过。

## Suggested Commands

```bash
python scripts/check_markdown_links.py README.md modules docs
```
