# ==============================================================================
# Llama-3 微调脚本 (基于 LLaMA Factory)
# ==============================================================================

# 1. 环境准备与依赖安装
# ------------------------------------------------------------------------------
# 切换到工作目录并清理旧的 LLaMA-Factory 文件夹
%cd /content/
%rm -rf LLaMA-Factory
# 克隆最新的 LLaMA-Factory 仓库
!git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
# 切换到仓库目录
%cd LLaMA-Factory
# 查看目录内容
%ls
# 以可编辑模式安装 LLaMA-Factory 及其依赖 (包含 torch 和 bitsandbytes)
!pip install -e .[torch,bitsandbytes]

# 2. 检查 GPU 环境
# ------------------------------------------------------------------------------
import torch
try:
  # 确保 CUDA (GPU) 可用，这对于大模型训练至关重要
  assert torch.cuda.is_available() is True
except AssertionError:
  print("Please set up a GPU before using LLaMA Factory: https://medium.com/mlearning-ai/training-yolov4-on-google-colab-316f8fff99c6")

# 3. 更新数据集身份信息 (Identity Dataset)
# ------------------------------------------------------------------------------
import json

# 切换到 LLaMA-Factory 目录以修改基础数据
%cd /content/LLaMA-Factory/

# 设置模型名称和作者信息，用于微调时的身份认知
NAME = "Llama-3"
AUTHOR = "LLaMA Factory"

# 读取身份数据集
with open("data/identity.json", "r", encoding="utf-8") as f:
  dataset = json.load(f)

# 替换数据集中的占位符
for sample in dataset:
  sample["output"] = sample["output"].replace("{{"+ "name" + "}}", NAME).replace("{{"+ "author" + "}}", AUTHOR)

# 将修改后的数据集保存回文件
with open("data/identity.json", "w", encoding="utf-8") as f:
  json.dump(dataset, f, indent=2, ensure_ascii=False)

# 4. 通过 Web UI 进行微调 (可选)
# ------------------------------------------------------------------------------
# 启动 LLaMA Board (Gradio 界面)，可以通过浏览器远程访问并进行可视化微调
%cd /content/LLaMA-Factory/
!GRADIO_SHARE=1 llamafactory-cli webui

# 5. 通过命令行进行命令行微调 (SFT - 有监督微调)
# ------------------------------------------------------------------------------
import json

# 配置微调参数
args = dict(
  stage="sft",                                               # 阶段：有监督微调
  do_train=True,                                             # 执行训练
  model_name_or_path="unsloth/llama-3-8b-Instruct-bnb-4bit", # 模型路径：使用 4-bit 量化的 Llama-3-8B-Instruct
  dataset="identity,alpaca_en_demo",                         # 数据集：使用身份数据集和 Alpaca 英文示例
  template="llama3",                                         # 模板：使用 llama3 提示词模板
  finetuning_type="lora",                                    # 微调类型：使用 LoRA 适配器以节省内存
  lora_target="all",                                         # LoRA 目标：将 LoRA 适配器应用于所有线性层
  output_dir="llama3_lora",                                  # 输出目录：保存 LoRA 适配器的路径
  per_device_train_batch_size=2,                             # 训练批次大小
  gradient_accumulation_steps=4,                             # 梯度累积步数
  lr_scheduler_type="cosine",                                # 学习率调度器：余弦退火
  logging_steps=5,                                           # 日志记录步数：每 5 步记录一次
  warmup_ratio=0.1,                                          # 预热比例
  save_steps=1000,                                           # 保存步数：每 1000 步保存一次检查点
  learning_rate=5e-5,                                        # 学习率
  num_train_epochs=3.0,                                      # 训练轮数
  max_samples=500,                                           # 每个数据集最大样本数
  max_grad_norm=1.0,                                         # 梯度裁剪：最大梯度范数为 1.0
  loraplus_lr_ratio=16.0,                                    # LoRA+ 参数：lambda=16.0
  fp16=True,                                                 # 使用混合精度训练 (float16)
  report_to="none",                                          # 禁用外部平台 (如 wandb) 的日志记录
)

# 将参数保存为 JSON 文件
json.dump(args, open("train_llama3.json", "w", encoding="utf-8"), indent=2)

# 切换到执行目录
%cd /content/LLaMA-Factory/

# 使用 llamafactory-cli 启动训练
!llamafactory-cli train train_llama3.json

# 6. 推理测试 (使用微调后的模型)
# ------------------------------------------------------------------------------
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc

%cd /content/LLaMA-Factory/

# 配置推理参数
args = dict(
  model_name_or_path="unsloth/llama-3-8b-Instruct-bnb-4bit", # 基础模型路径
  adapter_name_or_path="llama3_lora",                        # 加载保存的 LoRA 适配器
  template="llama3",                                         # 提示词模板 (需与训练时一致)
  finetuning_type="lora",                                    # 微调类型 (需与训练时一致)
)
# 初始化聊天模型
chat_model = ChatModel(args)

# 简单的 CLI 对话循环
messages = []
print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")
while True:
  query = input("\nUser: ")
  if query.strip() == "exit":
    break
  if query.strip() == "clear":
    messages = []
    torch_gc() # 清理显存
    print("History has been removed.")
    continue

  messages.append({"role": "user", "content": query})
  print("Assistant: ", end="", flush=True)

  response = ""
  # 使用流式输出进行对话
  for new_text in chat_model.stream_chat(messages):
    print(new_text, end="", flush=True)
    response += new_text
  print()
  messages.append({"role": "assistant", "content": response})

# 再次清理显存
torch_gc()

# 7. 合并 LoRA 适配器并归档模型
# ------------------------------------------------------------------------------
# 登录 Hugging Face (如果需要上传模型)
!huggingface-cli login

import json

# 配置合并导出参数
args = dict(
  model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct", # 使用官方非量化的原版模型进行合并
  adapter_name_or_path="llama3_lora",                       # 加载保存的 LoRA 适配器
  template="llama3",                                        # 提示词模板
  finetuning_type="lora",                                   # 微调类型
  export_dir="llama3_lora_merged",                          # 合并后模型的保存路径
  export_size=2,                                            # 合并后模型的分片大小 (GB)
  export_device="cpu",                                      # 导出设备：可以使用 cpu 或 auto
  # export_hub_model_id="your_id/your_model",               # 如果需要，设置 HF 模型 ID 进行上传
)

# 将参数保存为 JSON 配置文件
json.dump(args, open("merge_llama3.json", "w", encoding="utf-8"), indent=2)

# 切换到执行目录
%cd /content/LLaMA-Factory/

# 使用 llamafactory-cli 导出/合并模型
!llamafactory-cli export merge_llama3.json
