# Generation 分类

本目录聚焦生成模型：

- `diffusion`
- `dit`

建议学习顺序：
`diffusion -> dit`

## 子模块目录说明（统一）
- `code/`：主流程脚本，展示训练与采样关键逻辑。
- `data/`：训练样本、配置或数据说明。
- `models/`：最终导出的生成模型参数。
- `checkpoints/`：训练中间快照，用于断点续训与对比实验。
- `output/`：采样结果、曲线图、指标表和总结文件。
