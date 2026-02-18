# Actor-Critic

## 定位与分类

- **阶段**：后训练（Post-training）之策略优化基础。
- **类型**：混合架构（Policy-based + Value-based）。
- **作用**：它是 PPO / RLHF 的底层范式。通过“行为者-判官”协作，在提升模型性能的同时，极大降低了学习过程中的不确定性（方差）。

## 什么是 Actor-Critic？

Actor-Critic 是一种将“策略梯度”与“价值评估”相结合的经典模型架构：

- **Actor (行为者)**：策略网络。负责根据当前的指令，预测并生成具体的回答（Action）。
- **Critic (判官/记账员)**：价值网络（Value Head）。它不生产内容，而是评估当前状态的“优劣”，并预估未来的总奖励。

在 LLM 训练中，Critic 就像是一个专业的会计，时刻盯着 Actor 的产出，判断其是否超预期地获得了高分。

## 训练的关键步骤

1. **采样 (Sampling)**：Actor 接受指令，生成一组对话。
2. **打分 (Reward Calculation)**：模型获得一个奖励分（来自 RM 模型）。
3. **估值 (Value Estimation)**：Critic 对当前的对话状态给出一个“预估分”。
4. **计算优势 (Advantage Computation)**：计算真实得分比 Critic 预估的得分高出多少（$\text{Reward} - \text{Value}$）。
5. **双向更新 (Update)**：
   - **更新 Actor**：如果优势为正，增加该生成行为出现的概率。
   - **更新 Critic**：减小其预估分与真实分数之间的误差，使其预测更准。

## 核心数学公式

### 1. 优势估计 (Advantage)

$$\hat{A}_t = \text{Reward}_t - V_\phi(s_t)$$

- 如果 $\hat{A}_t > 0$，说明 Actor 的表现优于预期，应当获得正反馈。

### 2. Actor 目标 (策略梯度)

$$L_{actor} = - \log \pi_\theta(a|s) \cdot \hat{A}_t$$

- 通过优势函数加权，使高 Advantage 的动作概率变大。

### 3. Critic 目标 (价值均方误差)

$$L_{critic} = \frac{1}{2} (V_\phi(s_t) - G_t)^2$$

- $G_t$ 为真实累计奖励，Critic 通过回归学习减小误差。

## 与相近方法区别

1. 相比 `Policy Gradient`：多了 Critic，通常更稳定、更高样本效率。
2. 相比 `PPO`：Actor-Critic 是结构范式，PPO 是具体优化目标/约束策略。
3. 相比 `GAE`：GAE 是优势估计技术，可作为 Actor-Critic 的组成部分。

## 运行

```bash
cd /Users/yunxuanhan/Documents/workspace/ai/Finetune/post_train/alignment/actor_critic
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate finetune
python code/actor_critic.py --reward-model <奖励模型路径或名称>
```

## 输出结果

默认输出到 `output/actor_critic_metrics`，包含：

- `training_metrics.csv`
- `training_curves.png`
- `summary.json`
- `log_history.json`

## 目录文件说明（重点）

- `code/`：主流程代码，通常是可直接运行的单文件脚本。
- `data/`：示例数据、训练样本或数据索引配置。
- `models/`：训练完成后导出的最终模型权重（用于推理/部署）。
- `checkpoints/`：训练过程中的阶段性快照（含 step、优化器状态等），用于断点续训与回溯。
- `output/`：可视化图、指标表、训练日志与总结文件（如 `csv/png/json`）。
