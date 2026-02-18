# Systems 分类

本目录聚焦训练工程优化：

- `deepspeed`
- `cuda`
- `mixed_precision`

建议学习顺序：
`cuda -> mixed_precision -> deepspeed`

## 子模块目录说明（统一）
- `code/`：工程优化主流程脚本（含 benchmark 或训练对比逻辑）。
- `data/`：测试输入、配置文件或示例数据。
- `models/`：优化后训练得到的模型产物（如有）。
- `checkpoints/`：中间训练状态，支持恢复与实验复现。
- `output/`：性能对比、吞吐/显存指标、可视化图与总结。
