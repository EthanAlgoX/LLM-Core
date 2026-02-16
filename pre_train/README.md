# Pre-Train 学习总览

## 定位与分类
`pre_train` 用于学习模型预训练与多模态建模，覆盖 LLM、VLM 和生成模型。

当前模块按学习主题分为四类：

1. 语言模型预训练：`llm/{nanoGPT,megatron}`
2. 图像生成建模：`generation/{diffusion,dit}`
3. 视觉语言模型（VLM）：`vlm/{blip2,llava,flamingo}`
4. 工程扩展：可与 `post_train` 中 `deepspeed/cuda/mixed_precision` 联动学习

## 分类区别（学习路径）
1. `nanoGPT -> Megatron`：先理解单机简洁训练，再理解大规模并行训练。
2. `Diffusion -> DiT`：先理解扩散思想，再理解 Transformer 化扩散架构。
3. `BLIP2/LLaVA/Flamingo`：对比不同视觉-语言融合路线。

## 通用目录规范（每个模块）
- `code/`: 单文件主流程脚本（训练/推理入口）。
- `data/`: 数据集、样本或数据说明。
- `models/`: 最终导出的模型文件（用于推理）。
- `checkpoints/`: 训练中间快照（用于断点续训、回溯实验）。
- `output/`: 指标、曲线图、日志、总结等结果文件。

## 通用运行方式
```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/<category>/<module>
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/<module>.py
```

> 注：`nanoGPT` 为原始项目结构，入口是 `train.py / sample.py`。

## 统一入口（推荐）
在项目根目录可使用统一入口，便于面试时快速切换模块：

```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune

python run.py --list
python run.py --module diffusion --toy
python run.py --module llava --toy
```
