import os
import subprocess
import sys
import json
import torch

# ==============================================================================
# Qwen3 微调脚本 (基于 LLaMA Factory)
# ==============================================================================

# 基础配置
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACTORY_DIR = os.path.join(BASE_DIR, "LLaMA-Factory")
MODEL_ID = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "qwen3_lora"
TEMPLATE = "qwen"
NAME = "Qwen3"
AUTHOR = "LLaMA Factory"

def safe_chdir(path):
    if os.path.exists(path):
        os.chdir(path)
        print(f"Changed directory to {os.getcwd()}")
    else:
        print(f"Warning: Directory {path} not found.")

# 1. 检查 GPU 环境 (支持 CUDA 或 MPS)
# ------------------------------------------------------------------------------
try:
    if torch.cuda.is_available():
        print("GPU is available (CUDA).")
    elif torch.backends.mps.is_available():
        print("GPU is available (Apple Silicon MPS).")
    else:
        raise AssertionError("No GPU detected.")
except AssertionError:
    print("Warning: No GPU (CUDA or MPS) detected. Fine-tuning might be extremely slow on CPU.")

# 2. 更新数据集身份信息 (Identity Dataset)
# ------------------------------------------------------------------------------
safe_chdir(FACTORY_DIR)

identity_path = os.path.join(FACTORY_DIR, "data/identity.json")
if os.path.exists(identity_path):
    with open(identity_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    for sample in dataset:
        sample["output"] = sample["output"].replace("{{name}}", NAME).replace("{{author}}", AUTHOR)

    with open(identity_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Identity dataset updated with NAME='{NAME}' and AUTHOR='{AUTHOR}'.")
else:
    print(f"Warning: {identity_path} not found.")

# 3. 运行微调
# ------------------------------------------------------------------------------
# 训练参数配置
args = dict(
    stage="sft",
    do_train=True,
    model_name_or_path=MODEL_ID,
    dataset="identity,alpaca_en_demo",
    template=TEMPLATE,
    finetuning_type="lora",
    lora_target="all",
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,             # 针对 Mac 内存优化
    gradient_accumulation_steps=8,             # 增加梯度累积以平衡 Batch Size
    lr_scheduler_type="cosine",
    logging_steps=5,
    warmup_ratio=0.1,
    save_steps=1000,
    learning_rate=5e-5,
    num_train_epochs=3.0,
    max_samples=500,
    max_grad_norm=1.0,
    loraplus_lr_ratio=16.0,
    bf16=True,                                 # MPS 推荐使用 BF16
    report_to="none",
)

config_path = os.path.join(FACTORY_DIR, f"train_{NAME.lower()}.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(args, f, indent=2)

print(f"Starting training ({NAME})...")
os.environ["FORCE_TORCHRUN"] = "1"
try:
    subprocess.run(['llamafactory-cli', 'train', config_path], check=True)
    print("Training completed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error during training: {e}")
    sys.exit(1)

# 4. 推理测试 (使用微调后的模型)
# ------------------------------------------------------------------------------
# if os.path.exists(os.path.join(FACTORY_DIR, OUTPUT_DIR)):
#     from llamafactory.chat import ChatModel
#     chat_args = dict(
#         model_name_or_path=MODEL_ID,
#         adapter_name_or_path=OUTPUT_DIR,
#         template=TEMPLATE,
#         finetuning_type="lora",
#     )
#     chat_model = ChatModel(chat_args)
#     # ... 实现对话逻辑 ...

# 5. 模型合并与导出
# ------------------------------------------------------------------------------
# merge_args = dict(
#     model_name_or_path=MODEL_ID,
#     adapter_name_or_path=OUTPUT_DIR,
#     template=TEMPLATE,
#     finetuning_type="lora",
#     export_dir=f"{OUTPUT_DIR}_merged",
#     export_size=2,
#     export_device="cpu",
# )
# merge_config_path = os.path.join(FACTORY_DIR, f"merge_{NAME.lower()}.json")
# with open(merge_config_path, "w", encoding="utf-8") as f:
#     json.dump(merge_args, f, indent=2)
# os.system(f'llamafactory-cli export {merge_config_path}')
