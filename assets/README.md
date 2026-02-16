# Assets 目录

用于存放非核心源码资产，避免根目录堆积。

- `data/`: 示例数据与临时学习数据
- `history/`: 历史实验产物归档（如旧模型导出目录）

说明：
- 核心可运行代码仍在 `pre_train/`、`post_train/`、`run.py`、`scripts/`。
- 训练运行时的实时结果仍输出到各模块自己的 `output/` 与根目录 `output/`（回归报告）。
