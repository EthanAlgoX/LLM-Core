import os
import subprocess
import sys
import json
import torch

# 设置基础工作目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACTORY_DIR = os.path.join(BASE_DIR, "LLaMA-Factory")

def safe_chdir(path):
    if os.path.exists(path):
        os.chdir(path)
        print(f"Changed directory to {os.getcwd()}")
    else:
        print(f"Warning: Directory {path} not found.")

# Llama-3 微调脚本 (基于 LLaMA Factory)
# ==============================================================================

# 1. 环境准备与依赖安装
# ------------------------------------------------------------------------------
# 已手动完成安装，此处代码已注释

# 2. 检查 GPU 环境 (支持 CUDA 或 MPS)
# ------------------------------------------------------------------------------
try:
  # 确保 GPU 可用：CUDA (英伟达) 或 MPS (苹果芯片)
  if torch.cuda.is_available():
      print("GPU is available (CUDA).")
  elif torch.backends.mps.is_available():
      print("GPU is available (Apple Silicon MPS).")
  else:
      raise AssertionError("No GPU detected.")
except AssertionError:
  print("Warning: No GPU (CUDA or MPS) detected. Fine-tuning might be extremely slow on CPU.")

# 3. 更新数据集身份信息 (Identity Dataset)
# ------------------------------------------------------------------------------
# 切换到 LLaMA-Factory 目录以修改基础数据
safe_chdir(FACTORY_DIR)

# 设置模型名称和作者信息
NAME = "Llama-3"
AUTHOR = "LLaMA Factory"

identity_path = os.path.join(FACTORY_DIR, "data/identity.json")
if os.path.exists(identity_path):
    with open(identity_path, "r", encoding="utf-8") as f:
      dataset = json.load(f)

    for sample in dataset:
      sample["output"] = sample["output"].replace("{{"+ "name" + "}}", NAME).replace("{{"+ "author" + "}}", AUTHOR)

    with open(identity_path, "w", encoding="utf-8") as f:
      json.dump(dataset, f, indent=2, ensure_ascii=False)
    print("Identity dataset updated.")
else:
    print(f"Warning: {identity_path} not found.")

# 4. 运行微调/推理
# ------------------------------------------------------------------------------
# 根据需要选择运行 Web UI 或 命令行微调

# 选项 A: 启动 LLaMA Board (Web UI)
# os.environ['GRADIO_SHARE'] = '1'
# os.system('llamafactory-cli webui')

# 选项 B: 命令行微调 (SFT)
args = dict(
  stage="sft",
  do_train=True,
  model_name_or_path="Qwen/Qwen3-0.6B",           # 更换为 Qwen3 小参数模型
  dataset="identity,alpaca_en_demo",
  template="qwen",                                          # 模板使用 qwen
  finetuning_type="lora",
  lora_target="all",
  output_dir="qwen3_lora",                                  # 输出目录改为 qwen3_lora
  per_device_train_batch_size=1,                             # 针对 Mac 内存进行优化
  gradient_accumulation_steps=8,                             # 增加梯度累积
  lr_scheduler_type="cosine",
  logging_steps=5,
  warmup_ratio=0.1,
  save_steps=1000,
  learning_rate=5e-5,
  num_train_epochs=3.0,
  max_samples=500,
  max_grad_norm=1.0,
  loraplus_lr_ratio=16.0,
  bf16=True,                                                 # 使用 BF16 以适配 MPS
  report_to="none",
)

config_path = os.path.join(FACTORY_DIR, "train_qwen3.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(args, f, indent=2)

print("Starting training (Qwen3)...")
os.environ["FORCE_TORCHRUN"] = "1"
subprocess.run(['llamafactory-cli', 'train', config_path], check=True)

# 5. 推理测试 (使用微调后的模型)
# ------------------------------------------------------------------------------
# if os.path.exists(os.path.join(FACTORY_DIR, "qwen3_lora")):
#     from llamafactory.chat import ChatModel
#     from llamafactory.extras.misc import torch_gc
#     args = dict(
#       model_name_or_path="Qwen/Qwen3-0.6B-Instruct",
#       adapter_name_or_path="qwen3_lora",
#       template="qwen",
#       finetuning_type="lora",
#     )
#     chat_model = ChatModel(args)
#     # ... 推理对话逻辑 ...

# 6. 模型合并与导出
# ------------------------------------------------------------------------------
# args = dict(
#   model_name_or_path="Qwen/Qwen3-0.6B-Instruct",
#   adapter_name_or_path="qwen3_lora",
#   template="qwen",
#   finetuning_type="lora",
#   export_dir="qwen3_lora_merged",
#   export_size=2,
#   export_device="cpu",
# )
# config_path_merge = os.path.join(FACTORY_DIR, "merge_qwen3.json")
# with open(config_path_merge, "w", encoding="utf-8") as f:
#     json.dump(args, f, indent=2)
# os.system(f'llamafactory-cli export {config_path_merge}')
