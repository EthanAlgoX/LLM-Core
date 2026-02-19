# SFT (Supervised Fine-Tuning) 监督微调

> [!TIP]
> **一句话通俗理解**：拿人类写的高质量问答对，手把手教模型"怎么说话"

## 定位与分类

- **阶段**：后训练（Post-training）之对齐起点。
- **类型**：监督学习（Supervised Learning）。
- **作用**：将预训练基座模型（Base Model）转化为能够遵循人类指令（Instruction Following）的对话模型（Chat Model）。它是 RLHF 流程的物理基础。

## 模型训练的关键步骤

SFT 处理流程遵循以下核心步骤：

1. **数据分词 (Tokenization)**：将指令（Instruction）与回答（Output）拼接，并转换为模型可读的 Token IDs。
2. **掩码处理 (Label Masking)**：在计算损失时，通常将指令部分的标签置为 `-100`（忽略），确保模型仅学习如何生成回答，而不去学如何复述指令。
3. **前向传播 (Forward Pass)**：模型根据 Prompt 预测下一个字符（Token）的概率分布。
4. **损失计算 (Loss Calculation)**：使用**交叉熵（Cross-Entropy）**对比预测值与标准答案。
5. **反向传播与优化 (Backprop & Update)**：根据梯度更新模型权重（或 LoRA 权重）。

## 核心原理与损失函数

### 1. 关键公式：交叉熵损失 (Cross-Entropy Loss)

SFT 的本质是**最大似然估计（MLE）**，其核心数学目标是最小化回答序列的负对数似然：

$$L(\theta) = - \sum_{i=1}^{T} \log P_\theta(y_i | x, y_{1}, \dots, y_{i-1})$$

**公式拆解与理解：**

- **$x$ (Input)**：输入的指令内容（Prompt）。
- **$y_i$ (Target)**：标准答案中第 $i$ 个位置的词（Token）。
- **$P_\theta(\dots)$**：模型根据当前参数 $\theta$，在已知指令和前序文字的前提下，预测出正确下一个词的“概率”。
- **$\log$ 与负号**：将概率转化为损失值。概率越大（预测越准）， $\log$ 越接近 0，损失值越小。

### 2. 深度解读：如何直观理解这个过程？

- **逐词对齐 (Token-level Alignment)**：模型在每一个步长上都在尝试预测“下一个词”。它在学习标准答案中词与词之间的统计规律。
- **Teacher Forcing (强制纠偏)**：这是 SFT 的关键特征。在训练前向传播时，无论模型预测出的上一个词是否正确，模型在计算当前词时输入的永远是**真实答案**中的前文。就像老师牵着手写字，错了一笔立即拉回。
- **概率最大化**：公式的终极目的是让模型在看到特定指令时，能够以“最大概率”吐出数据集里的标准字句。

### 3. 与 PPO/GRPO 的本质区别

| 特性 | SFT (监督微调) | RL (PPO/GRPO) |
| :--- | :--- | :--- |
| **学习源** | **静态标签**（Output 字对字模仿）。 | **动态反馈**（Reward 打分驱动）。 |
| **灵活性** | 低。模型被限制在模仿数据集。 | 高。模型可以探索数据集之外更好的解。 |
| **稳定性** | 极高。最简单的梯度下降。 | 低。容易发散，需要复杂的超参控制。 |

## 关键配置解读

| 参数 | 建议值 | 原理解读 |
| :--- | :--- | :--- |
| `learning_rate` | `1e-4` 或 `5e-5` | 相比 RL，SFT 使用较高的学习率以快速学习任务模式。 |
| `cutoff_len` | `1024` | 决定了模型单次能处理的问题+答案的总长度。 |
| `lora_target` | `all` | 为所有线性层添加低秩适配器，可以在提升效果的同时极大节省显存。 |

## 🛠️ 工程实战：使用 LLaMA Factory 进行 SFT

[LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) 是目前最流行的开源微调框架，支持 100+ 模型、LoRA/QLoRA/全量微调、WebUI 可视化训练。

### Step 1: 环境准备

```bash
# 安装 LLaMA Factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### Step 2: 数据集准备

LLaMA Factory 使用 `data/dataset_info.json` 注册数据集。自定义数据集只需两步：

**2a. 准备 JSONL 数据文件**（Alpaca 格式）：

```json
[
  {
    "instruction": "请解释什么是 Transformer 中的注意力机制。",
    "input": "",
    "output": "Transformer 中的注意力机制（Attention Mechanism）是一种让模型在处理序列时，能够动态关注不同位置信息的方法..."
  },
  {
    "instruction": "将以下文本翻译成英文。",
    "input": "大语言模型正在改变人工智能的格局。",
    "output": "Large language models are reshaping the landscape of artificial intelligence."
  }
]
```

**2b. 在 `dataset_info.json` 中注册**：

```json
{
  "my_custom_sft": {
    "file_name": "my_custom_sft.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

### Step 3: 训练配置（YAML）

创建训练配置文件 `examples/train_lora/my_sft.yaml`：

```yaml
### 模型配置
model_name_or_path: Qwen/Qwen2.5-7B           # 基座模型（HuggingFace ID 或本地路径）
trust_remote_code: true

### 微调方式
stage: sft                                      # 训练阶段：sft
do_train: true
finetuning_type: lora                           # 微调类型：lora / full / freeze

### LoRA 超参
lora_target: all                                # 对所有线性层注入 LoRA
lora_rank: 64                                   # 秩越大，表达力越强，但显存越多
lora_alpha: 128                                 # 缩放系数，通常为 rank 的 2 倍
lora_dropout: 0.05

### 数据配置
dataset: my_custom_sft                          # 对应 dataset_info.json 中的 key
template: qwen                                  # 对话模板（qwen / llama3 / chatglm 等）
cutoff_len: 2048                                # 最大序列长度
preprocessing_num_workers: 16

### 训练超参
per_device_train_batch_size: 2
gradient_accumulation_steps: 8                  # 有效批次 = 2 × 8 = 16
num_train_epochs: 3.0
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true                                      # BF16 混合精度（A100/H100）
gradient_checkpointing: true                    # 用时间换显存

### 日志与保存
logging_steps: 10
save_steps: 500
output_dir: saves/qwen2.5-7b/lora/my_sft
report_to: tensorboard
```

### Step 4: 启动训练

```bash
# 方式一：CLI 命令行启动（推荐）
llamafactory-cli train examples/train_lora/my_sft.yaml

# 方式二：WebUI 可视化启动
llamafactory-cli webui
```

> **显存估算**：Qwen2.5-7B + LoRA (rank=64) + BF16 + Gradient Checkpointing ≈ **16~20GB VRAM**（单卡 A100/4090 可跑）。

### Step 5: 合并 LoRA 权重

训练完成后，LoRA 权重需要合并回基座模型才能独立部署：

```yaml
# merge_lora.yaml
model_name_or_path: Qwen/Qwen2.5-7B
adapter_name_or_path: saves/qwen2.5-7b/lora/my_sft
template: qwen
finetuning_type: lora
export_dir: models/qwen2.5-7b-sft-merged        # 合并后的完整模型输出路径
export_size: 4                                    # 每个分片大小 (GB)
export_legacy_format: false
```

```bash
llamafactory-cli export merge_lora.yaml
```

### Step 6: 推理验证

```bash
# 快速对话测试（使用 LoRA 适配器，无需合并）
llamafactory-cli chat examples/train_lora/my_sft.yaml
```

或使用 Python 加载合并后的模型：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/qwen2.5-7b-sft-merged"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")

messages = [{"role": "user", "content": "请解释什么是 LoRA 微调？"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.9)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

---

## 🔧 进阶：多 GPU / DeepSpeed 分布式训练

```yaml
# 在 YAML 中添加 DeepSpeed 配置
deepspeed: examples/deepspeed/ds_z2_config.json   # ZeRO-2（推荐 SFT）
```

```bash
# 多卡启动
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train examples/train_lora/my_sft.yaml
```

---

## 📊 训练监控

```bash
# 查看 TensorBoard 训练曲线
tensorboard --logdir saves/qwen2.5-7b/lora/my_sft
```

**关键指标**：

- `train/loss`：应平滑下降至 1.0 以下。
- `eval/loss`：若与 train/loss 差距持续增大，说明**过拟合**，需减少 epoch 或增加数据。

---

## 原始脚本运行

本模块也提供了不依赖框架的纯 PyTorch SFT 实现，供理解底层机制：

```bash
python code/sft.py
```
