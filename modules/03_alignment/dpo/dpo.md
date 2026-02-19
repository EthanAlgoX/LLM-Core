# DPO（直接偏好优化）

> [!TIP]
> **一句话通俗理解**：给 AI 两个答案让它选好的，通过"偏好打分"驯化它说人话

## 定位与分类

- **阶段**：后训练（Post-training）之偏好对齐。
- **类型**：直接偏好学习（Direct Preference Learning）。
- **作用**：取代复杂的“奖励模型 + PPO”流程，直接通过对比“好回答”与“坏回答”，将人类的偏好注入模型中。

## 什么是 DPO？

DPO（Direct Preference Optimization）是由斯坦福大学提出的一种简化版对齐算法。它的核心思想是：**不再训练一个裁判（奖励模型），而是直接让模型在“好坏对”中学习。**
它在数学上证明了，通过对数比例（Log-Ratio）的优化，可以达到与传统 RLHF 相同的对齐效果，但工程实现难度降低了 90%。

## DPO 训练的关键步骤

1. **构建偏好对 (Preference Pairs)**：准备数据，格式为 `(Prompt, Chosen_Answer, Rejected_Answer)`。
2. **加载双模型**：
   - **Policy Model (待训模型)**：我们要优化的 Actor。
   - **Reference Model (参考模型)**：通常是 SFT 后的冻结模型，作为动态基准。
3. **计算 Log-Prob**：待训模型和参考模型分别对 Chosen 和 Rejected 答案计算预测概率的对数（Log-Probability）。
4. **计算对数比例差距 (Log-Ratio Gap)**：计算待训模型相对于参考模型，在 Chosen 上的进步是否比在 Rejected 上的进步更大。
5. **偏好更新 (Optimization)**：通过 Sigmoid 激活函数和梯度下降，拉大好坏答案之间的差距。

## 核心原理与关键公式

### 1. 关键公式：DPO 损失函数

DPO 的伟大之处在于它证明了可以直接利用偏好数据优化策略，而不需要训练显式的奖励模型。其目标函数为：

$$L_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{ref}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{ref}(y_l | x)} \right) \right]$$

**公式拆解与理解：**

- **$\pi_\theta$ 与 $\pi_{ref}$**：当前优化的模型与冻结的参考模型（通常是 SFT 后的模型）。
- **$y_w$ (Chosen) 与 $y_l$ (Rejected)**：偏好对中的“好答案”与“坏答案”。
- **$\log \frac{\pi_\theta}{\pi_{ref}}$ (Log-Ratio)**：衡量当前模型相对于参考模型，对某个回答概率的“提升程度”。
- **偏好边际 (Preference Margin)**：括号内的两项相减，代表了模型对“好答案”的提升程度是否远大于对“坏答案”的提升程度。
- **$\beta$ (Beta 系数)**：调节因子。控制对偏好的敏感度，同时也起到了类似 PPO 中 KL 散度的约束作用，防止模型跑得太偏。

### 2. 深度解读：为什么它能取代奖励模型？

- **隐含奖励 (Implicit Reward)**：DPO 发现，一个最优策略 $\pi$ 与奖励函数之间存在一一映射关系。
- **直接对比**：DPO 不去问“这个回答得多少分”，而是问“当前的这个模型，是不是比原始模型更喜欢好回答、更讨厌坏回答”。
- **工程简化**：省去了训练奖励模型、在线采样、以及复杂的 Critic 调优过程。

## 与相近方法区别

1. 相比 `SFT`：DPO 学“相对偏好”，而不是“绝对标准答案”。
2. 相比 `PPO/RLHF`：DPO 不需要在线 rollouts，工程更简洁。
3. 相比 `GRPO`：DPO 常基于成对偏好数据，GRPO 常基于组内多采样奖励比较。

## 🛠️ 工程实战：DPO 训练

### 方式一：LLaMA Factory（推荐）

**数据格式**（偏好对格式，在 `dataset_info.json` 中注册）：

```json
[
  {
    "instruction": "解释量子纠缠",
    "input": "",
    "chosen": "量子纠缠是一种量子力学现象，两个粒子的状态相互关联...",
    "rejected": "量子纠缠就是两个东西连在一起。"
  }
]
```

**训练配置 YAML**：

```yaml
### DPO 训练配置
model_name_or_path: Qwen/Qwen2.5-7B
stage: dpo                              # 关键：设为 dpo
do_train: true
finetuning_type: lora

### DPO 特有参数
pref_beta: 0.1                          # β 系数，控制偏好敏感度（默认 0.1）
pref_loss: sigmoid                      # 损失类型：sigmoid / hinge / ipo

### LoRA
lora_rank: 64
lora_target: all

### 数据
dataset: my_dpo_data                    # 偏好对数据集
template: qwen
cutoff_len: 2048

### 训练
per_device_train_batch_size: 1          # DPO 需要同时加载 chosen + rejected，显存翻倍
gradient_accumulation_steps: 16
num_train_epochs: 2.0
learning_rate: 5.0e-6                   # DPO 学习率通常比 SFT 低一个数量级
bf16: true
output_dir: saves/qwen2.5-7b/lora/dpo
```

```bash
llamafactory-cli train dpo_config.yaml
```

> **显存注意**：DPO 需同时维护 Policy + Reference 两个模型，显存需求约为 SFT 的 **2 倍**。

### 方式二：TRL 库（HuggingFace）

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="auto")
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="auto")  # 冻结参考模型
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# 2. 加载偏好数据
dataset = load_dataset("json", data_files="data/dpo_pairs.json")

# 3. 训练配置
training_args = DPOConfig(
    output_dir="saves/dpo",
    beta=0.1,                           # KL 约束强度
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,
    num_train_epochs=2,
    bf16=True,
)

# 4. 启动 DPO 训练
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
)
trainer.train()
```

---

## 原始脚本运行

```bash
cd <YOUR_PROJECT_ROOT>/post_train/alignment/dpo
conda activate finetune
# 纯文档仓库：历史脚本命令已归档
```

## 输出结果

默认输出到 `output/dpo_metrics`，包含：

- `training_metrics.csv`
- `training_curves.png`
- `summary.json`
- `log_history.json`
