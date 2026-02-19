# 高效微调 (PEFT: Parameter-Efficient Fine-Tuning)

> [!TIP]
> **一句话通俗理解**：只改模型一小部分权重，就能让它学会新技能，省时省显存

## 定位与分类

- **阶段**：后训练对齐 / 微调优化。
- **类型**：轻量化微调技术。
- **作用**：PEFT 旨在仅训练极少量参数，在大规模预训练模型上实现下游任务适配。LoRA 是技术解析中几乎必考的高频考点。

## 核心算法：LoRA (Low-Rank Adaptation)

### 核心思想

假设模型权重的更新量 $\Delta W$ 是 **低秩 (Low-Rank)** 的。
我们可以将 $\Delta W$ 分解为两个极小的矩阵相乘：

$$\Delta W = A \times B$$

- 其中 $W \in \mathbb{R}^{d \times k}$，$A \in \mathbb{R}^{d \times r}$，$B \in \mathbb{R}^{r \times k}$，秩 $r \ll d, k$。

### 训练与推理

1. **训练阶段**：冻结原始权重 $W$，仅训练 $A$ 和 $B$。
2. **推理阶段**：将 $A \times B$ 重新合并回 $W$ （即 $W_{new} = W + AB$ ），因此**推理延迟为零**。

### 为什么 LoRA 显存占用低？

因为它不存储庞大的梯度矩阵 $\Delta W$ ，仅存储细小的 $A$ 和 $B$ 。

## 进阶：QLoRA (Quantized LoRA)

### QLoRA 技术亮点

1. **4-bit NormalFloat (NF4)**：专门为正态分布权重设计的量化格式，比 4-bit Float 精度更高。
2. **Double Quantization**：对量化常数本身再进行一次量化，节省额外的几百 MB 显存。
3. **Paged Optimizers**：将优化器状态在显存和内存之间自动切换，防止 OOM。

## 其他轻量化技术

### 1. Prefix Tuning

- **核心逻辑**：在输入 Token 前拼接一组可训练的 **Virtual Tokens (Prefix)**。
- **与 LoRA 区别**：
  - **Prefix Tuning**：改变的是输入 Hidden State，增加了一定的推理计算量。
  - **LoRA**：改变的是权重 $W$，可直接合并，推理零额外开销。

### 2. P-Tuning / Prompt Tuning

- 仅在 Embedding 层增加可训练向量，适用于任务指令极其明确的场景。

## 知识蒸馏 (Knowledge Distillation)

### 蒸馏技术亮点

- **Teacher-Student 架构**：大模型 (Teacher) 引导小模型 (Student) 学习。
- **Logits 蒸馏**：Student 拟合 Teacher 输出的概率分布。
- **能力提取**：常用于将 175B 模型的复杂逻辑蒸馏到 7B 模型中，提升端侧执行速度。

## 技术核心解析

1. LoRA 的 $r$ （秩）选多少合适？
   - 通常 8 或 16 已经足够。过大的 $r$ 会增加显存但并不一定会提升精度。
2. LoRA 与全参微调的收敛速度？
   - LoRA 收敛通常更快，因为它优化的是低秩残差，更容易在局部搜索到最优解。
3. PEFT 在多模态模型中的应用？
   - 常用于固定 ViT 编码器，仅对 Projector 或 LLM 部分进行 LoRA 微调，实现跨模态对齐。

---

## 🛠️ 工程实战

### 方式一：PEFT 库（HuggingFace 原生）

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载基座模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="auto")

# 2. 定义 LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,                          # 秩（推荐 8~64）
    lora_alpha=128,                # 缩放系数，通常 = 2 × r
    lora_dropout=0.05,
    target_modules="all-linear",   # 对所有线性层注入 LoRA
)

# 3. 包装模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出示例：trainable params: 83,886,080 || all params: 7,699,726,336 || trainable%: 1.089%
```

### 方式二：QLoRA（4-bit 量化 + LoRA）

```python
from transformers import BitsAndBytesConfig
import torch

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4，精度优于 FP4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # 二次量化，再省几百 MB
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    quantization_config=bnb_config,
    device_map="auto",
)

# 然后同样使用 get_peft_model 包装
model = get_peft_model(model, lora_config)
# QLoRA: 7B 模型仅需 ~6GB VRAM
```

### 方式三：LLaMA Factory 一键微调

```yaml
# peft_lora.yaml
model_name_or_path: Qwen/Qwen2.5-7B
finetuning_type: lora              # 或 qlora（自动启用 4-bit）
quantization_bit: 4                # 启用 QLoRA
lora_rank: 64
lora_target: all
stage: sft
dataset: my_custom_sft
template: qwen
output_dir: saves/qwen2.5-7b/qlora
```

```bash
llamafactory-cli train peft_lora.yaml
```

### LoRA 权重合并与导出

```python
from peft import PeftModel

# 加载基座 + LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
model = PeftModel.from_pretrained(base_model, "saves/qwen2.5-7b/qlora")

# 合并权重（推理零延迟）
merged_model = model.merge_and_unload()
merged_model.save_pretrained("models/qwen2.5-7b-merged")
```
