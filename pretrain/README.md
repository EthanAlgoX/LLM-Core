# Pretrain

该目录用于放置预训练相关项目。

## 已整合项目

- `nanoGPT`: `/Users/yunxuanhan/Documents/workspace/ai/Finetune/pretrain/nanoGPT`

## 快速开始（nanoGPT）

```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/pretrain/nanoGPT
pip install torch numpy transformers datasets tiktoken wandb tqdm
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
python sample.py --out_dir=out-shakespeare-char
```

> 说明：以上命令与原始 `nanoGPT` 用法一致，只是路径迁移到了 `Finetune/pretrain/nanoGPT`。
