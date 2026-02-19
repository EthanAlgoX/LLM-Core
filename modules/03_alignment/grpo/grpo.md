# GRPO (Group Relative Policy Optimization) 组内相对策略优化

## 定位与分类

- **阶段**：后训练（Post-training）之对齐/推理增强阶段。
- **类型**：强化学习（数学逻辑推理增强）。
- **作用**：由 DeepSeek 提出，通过取消 Critic 模型并采用组内相对分数（Group Relative），显著降低显存开销，并提升模型在逻辑推理任务中的爆发力。

## 核心架构：化繁为简

相比 PPO 的“四角平衡”，GRPO 采用了更轻量化的“三角结构”：

| 角色 | 是否存在 | 职责描述 | 状态 |
| :--- | :--- | :--- | :--- |
| **Actor** | 是 | 核心优化对象。负责根据指令生成回复。 | **动态更新** |
| **Reference** | 是 | 冻结的原型。计算 KL 散度，防止策略崩溃。 | **完全冻结** |
| **Reward** | 是 | 裁判。可以是神经网络模型，也可以是硬性规则（如编译器）。 | **完全冻结** |
| **Critic** | **否** | **取消。** 不再预测期望得分，由组内平均分替代其功能。 | **N/A** |

> **优势**：取消 Critic 模型可节省约 50% 的模型权重显存，支持更大规模的并行采样。

## 核心逻辑：组内对比 (Group Relative)

这是 GRPO 名字的由来。它不再看“历史平均分（Critic）”，而是看“同侪表现”：

1. **组内采样**：对于同一个问题，Actor 一次性生成一组回答（采样数由 `num_generations` 控制，如一组 8 个）。
2. **计算优势 (Advantage)**：
   - 算出这组回答的平均分（Mean）和标准差（Std）。
   - **Advantage 公式**： $A_i = \frac{Reward_i - \mathrm{Mean}(Rewards)}{\mathrm{Std}(Rewards)}$
3. **原理**：只要你的回答比同组的其他“兄弟”好，你就获得正向激励。这种横向对比天然抹平了题目难度的干扰。

## 核心原理与数学公式

### 1. 组内优势函数 (Group Relative Advantage)

这是 GRPO 的核心数学创新。对于针对同一个 Prompt 生成的一组回答 $\{o_1, o_2, \dots, o_G\}$，每个回答的优势 $A_i$ 计算如下：

$$A_i = \frac{r_i - \mathrm{mean}(r_1, r_2, \dots, r_G)}{\mathrm{std}(r_1, r_2, \dots, r_G)}$$

- **$r_i$**：第 $i$ 个回答获得的显式奖励分数。
- **$\mathrm{mean}$ 与 $\mathrm{std}$**：这组回答奖励分的平均值和标准差。
- **直觉理解**：这是一种**归一化**操作。它将绝对分数转化为了“在该组中的表现排名”。

### 2. 目标优化函数 (Objective Function)

GRPO 沿用了 PPO 的剪切（Clipped）思想，但在计算期望时是在组内进行的：

$$J_{GRPO}(\theta) = \mathbb{E} \left[ q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}} \right] \left( \frac{1}{G} \sum_{i=1}^G L_i^{CLIP}(\theta) - \beta D_{KL}(\pi_\theta || \pi_{ref}) \right)$$

- **$\frac{1}{G} \sum$**：对整组回答的损失进行平均。
- **KL 散度约束**：同样保留了 KL 惩罚，防止模型为了赢得组内竞争而写出乱码。

### 场景分析：组内对比如何奏效？

- **题目极难时**：
    假设由于题目太难，全组 8 个回答的绝对得分都很低（平均分仅 10 分）。
  - **A 回答**：得了 12 分。虽然绝对分低，但在组内是“优等生”， $Advantage > 0$，模型会学习奖励这种行为。
- **题目极简单时**：
    假设由于题目太易，全组平均分高达 95 分。
  - **B 回答**：得了 90 分。虽然绝对分很高，但在组内是“差生”， $Advantage < 0$，模型反而会反思这种行为。

> **结论**：GRPO 让模型不再纠结于分数的“绝对值”，而是专注于**“如何做得比同类更好”**。

## GRPO vs. PPO 深度对比

| 特性 | PPO (经典) | GRPO (新型) |
| :--- | :--- | :--- |
| **基准来源** | **纵向对比**：靠 Critic 神经网络预测。 | **横向对比**：靠统计学组内平均值。 |
| **显存压力** | 高（需要维护巨大的 Critic 网络）。 | 低（取消 Critic，省显存）。 |
| **稳定性** | 依赖 Critic 的拟合质量。 | 依赖组内采样数量 (num_generations)。 |
| **最佳场景** | 对话对齐、通用偏好学习。 | **逻辑推理、数学难题、深度思索 (CoT)**。 |

## 关键训练配置

| 参数 | 脚本键值 | 原理解读 |
| :--- | :--- | :--- |
| `num_generations` | `2` (Demo) / `8~16` (生产) | 每组采样个数。越大，组内统计出的平均值越准，训练越稳。 |
| `scale_rewards` | `"group"` | 开启组内标准化模式。这是 GRPO 的核心开关。 |
| `learning_rate` | `5e-7` | 极低的学习率，防止策略梯度在采样不足时产生抖动。 |

## 🛠️ 工程实战：GRPO 训练

### 方式一：LLaMA Factory

**数据格式**（与 PPO 类似，Prompt-only + 可验证奖励）：

```json
[
  {"instruction": "计算 (3 + 5) × 2 = ?", "input": "", "output": "16"},
  {"instruction": "求解方程 2x + 3 = 11", "input": "", "output": "x = 4"}
]
```

**训练配置 YAML**：

```yaml
### GRPO 训练配置
model_name_or_path: Qwen/Qwen2.5-7B
stage: grpo                             # 关键：设为 grpo（而非 ppo）
do_train: true
finetuning_type: lora

### GRPO 特有参数
num_generations: 8                      # 每题采样 G 个答案（核心超参）
pref_beta: 0.04                         # KL 约束强度

### 奖励配置（可验证奖励，无需 RM）
reward_funcs: accuracy,format           # 内置奖励函数：准确率 + 格式检查

### LoRA
lora_rank: 64
lora_target: all

### 数据
dataset: my_math_data
template: qwen
cutoff_len: 4096                        # 推理任务需要更长上下文

### 训练
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 5.0e-7                   # 极低学习率，GRPO 对梯度更敏感
num_train_epochs: 1
bf16: true
output_dir: saves/qwen2.5-7b/lora/grpo
```

```bash
llamafactory-cli train grpo_config.yaml
```

> **显存估算**：GRPO 无需 Critic，但 `num_generations=8` 意味着每步生成 8 条回复。7B + LoRA + 8 采样 ≈ **40~60GB VRAM**（建议多卡或 ZeRO-3）。

### 方式二：TRL 库 + 自定义奖励

```python
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# 2. 定义可验证奖励函数
def accuracy_reward(completions, references, **kwargs):
    """提取答案并与标准答案对比"""
    rewards = []
    for completion, ref in zip(completions, references):
        # 提取 <answer>...</answer> 中的内容
        match = re.search(r"<answer>(.*?)</answer>", completion)
        predicted = match.group(1).strip() if match else ""
        rewards.append(1.0 if predicted == ref else 0.0)
    return rewards

def format_reward(completions, **kwargs):
    """检查输出格式是否包含 think + answer 标签"""
    rewards = []
    for completion in completions:
        has_think = "<think>" in completion and "</think>" in completion
        has_answer = "<answer>" in completion and "</answer>" in completion
        rewards.append(1.0 if has_think and has_answer else 0.0)
    return rewards

# 3. GRPO 配置
training_args = GRPOConfig(
    output_dir="saves/grpo",
    num_generations=8,                   # 每题生成 G 个候选
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    bf16=True,
)

# 4. 启动 GRPO 训练
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    reward_funcs=[accuracy_reward, format_reward],  # 多奖励函数组合
)
trainer.train()
```

---

## 原始脚本运行

```bash
python code/grpo_demo.py
```

**可视化**：默认输出至 `output/grpo_metrics`。关注 `reward`（总分）与 `reward_std`（组内差异）的变化趋势。
