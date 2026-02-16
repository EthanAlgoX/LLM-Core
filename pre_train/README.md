# Pretrain

该目录用于放置预训练相关项目。

## 已整合项目

- `nanoGPT`: `/Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/nanoGPT`
- `diffusion`: `/Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/diffusion`
- `dit`: `/Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/dit`
- `blip2`: `/Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/blip2`
- `llava`: `/Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/llava`
- `flamingo`: `/Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/flamingo`

## 快速开始（nanoGPT）

```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/nanoGPT
pip install torch numpy transformers datasets tiktoken wandb tqdm
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
python sample.py --out_dir=out-shakespeare-char
```

## 快速开始（diffusion）

```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/diffusion
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/diffusion.py
```

## 快速开始（DiT）

```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/dit
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/dit.py
```

## 快速开始（BLIP2）

```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/blip2
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/blip2.py --dry-run
```

## 快速开始（LLaVA）

```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/llava
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/llava.py --dry-run
```

## 快速开始（Flamingo）

```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pre_train/flamingo
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/flamingo.py --dry-run
```

> 说明：以上命令与原始 `nanoGPT` 用法一致，只是路径迁移到了 `Finetune/pre_train/nanoGPT`。
