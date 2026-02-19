# GRPO (Group Relative Policy Optimization) 组内相对策略优化

> [!TIP]
> **一句话通俗理解**：GRPO 在同一问题内做组内相对比较，利用标准化优势信号提升推理对齐的样本效率与稳定性。

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

## 关键公式

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

```python
# 关键步骤代码（示意）
state = init_state()
for step in range(num_steps):
    state = step_update(state)
metrics = evaluate(state)
```

**可视化**：默认输出至 `output/grpo_metrics`。关注 `reward`（总分）与 `reward_std`（组内差异）的变化趋势。

---
## 定义与目标

- **定义**：GRPO (Group Relative Policy Optimization) 组内相对策略优化 属于“后训练对齐模块，聚焦 SFT、偏好优化与 RLHF 系列方法。”范畴。
- **目标**：在能力、可控性与安全性之间建立可迭代的对齐训练闭环。
## 适用场景与边界

- **适用场景**：用于构建指令跟随、偏好对齐与奖励驱动优化流程。
- **不适用场景**：不适用于缺少高质量偏好数据或评测体系的直接落地。
- **使用边界**：对齐收益受数据质量、奖励建模与 KL 约束策略影响明显。

## 关键步骤

1. 构建对齐数据与偏好信号（指令数据/偏好对/奖励模型）。
2. 在约束条件下优化策略，使输出更符合人类偏好。
3. 联合有用性、安全性与稳定性指标进行迭代评估。
## 关键步骤代码（纯文档示例）

```python
# 关键流程示意（与具体工程实现解耦）
state = init_state()
for step in range(num_steps):
    state = step_update(state)
metrics = evaluate(state)
```

## 工程实现要点

- 优先保证数据质量与评测一致性，再放大训练规模。
- 在线/离线对齐需分别监控稳定性、奖励漂移与过优化风险。
- 保持参考模型与训练模型版本可追踪，便于回溯问题。

## 常见错误与排查

- **症状**：奖励升高但人工体验下降。  
  **原因**：奖励黑客或偏好模型偏差导致目标错位。  
  **解决**：引入人工抽检与多指标约束，限制单一奖励驱动。
- **症状**：训练不稳定或发散。  
  **原因**：学习率/KL 系数/批量配置不匹配。  
  **解决**：缩小超参搜索范围并分阶段增大训练强度。

## 与相近方法对比

| 方法 | 优点 | 局限 | 适用场景 |
| --- | --- | --- | --- |
| 本文主题方法 | 紧贴本节问题定义 | 依赖数据与实现质量 | 适合结构化评测与迭代优化 |
| 对比方法A | 上手成本更低 | 能力上限可能受限 | 快速原型与基线对照 |
| 对比方法B | 上限潜力更高 | 调参与资源成本更高 | 高要求生产或复杂任务场景 |

## 参考资料

- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

