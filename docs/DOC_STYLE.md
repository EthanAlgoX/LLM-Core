# Documentation Style Guide

本仓库是文档优先仓库。所有模块文档按统一结构编写，确保读者能够快速从概念走到实现。

## Scope

- 适用于 `README.md` 与 `modules/` 下所有 Markdown 文档。
- 适用于 `docs/` 下所有规范与模板文档。

## Required Structure

模块文档应尽量遵循以下顺序（可按主题微调）：

1. 一句话通俗理解
2. 定义与目标
3. 关键步骤
4. 关键公式（含符号说明）
5. 关键步骤代码（纯文档示例）
6. 工程实现要点
7. 常见错误与排查
8. 与相近方法对比
9. 参考资料

模板见：`docs/MODULE_DOC_TEMPLATE.md`。

## Writing Rules

- 每节只讲一个核心问题，避免重复叙述。
- 先给结论，再给解释，再给例子。
- 首次出现缩写要展开全称（如 LLM、VLM、RLHF）。
- 术语以 `docs/TERMINOLOGY.md` 为准。
- 代码示例只保留关键步骤，不引用历史脚本路径。

## Link Rules

- 仓库内链接使用相对路径。
- 提交前必须通过链接检查。
- 标题锚点应与实际标题一致。

## Review Checklist

- 结构是否符合 `Required Structure`。
- 公式是否给出符号含义。
- 代码块是否体现关键步骤而非历史入口命令。
- README 索引是否能导航到对应模块。
- 链接检查是否通过。

## Suggested Commands

```bash
python scripts/check_markdown_links.py README.md modules docs
```
