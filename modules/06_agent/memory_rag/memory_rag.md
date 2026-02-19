# 记忆与检索 (Memory & RAG)

> [!TIP]
> **一句话通俗理解**：AI 自带可检索"知识库"，让它能回答私有或超出训练范围的问题

Agent 的记忆系统是实现长期交互能力的关键。分为短期记忆和长期记忆两大类。

---

## 短期记忆 (Short-term Memory)

- **Context Window**：利用 LLM 的上下文窗口存储对话历史和中间思考过程
- **Session State**：当前会话中的变量、状态

---

## 长期记忆 (Long-term Memory) - RAG 架构

### 核心流程

1. **Query 理解**
   - 意图识别：判断用户意图（搜索、问答、任务执行）
   - 实体提取：从 Query 中提取关键实体
   - 查询扩展 (Query Expansion)：生成多个同义表述提升召回

2. **向量检索 (Vector Search)**
   - 计算 Embedding 相似度（如余弦相似度）
   - 从向量库召回候选集
   - 典型方案：Chroma、Pinecone、Milvus、Weaviate

3. **Rerank 模型**
   - 对召回结果进行精排
   - 解决语义稀释问题
   - 典型方案：Cross-Encoder、BGE-Reranker

---

## 进阶技术

### 1. 混合检索 (Hybrid Search)

结合向量检索与关键词检索（BM25），平衡语义与精确匹配。

### 2. 分块策略 (Chunking)

- 固定大小分块
- 语义分块（按段落、章节）
- 重叠滑动窗口保持上下文连贯

### 3. Agentic RAG

- 动态决定是否检索
- 多轮检索与推理结合
- 自我反思与验证

---
## 定义与目标

- **定义**：本节主题用于解释该模块的核心概念与实现思路。
- **目标**：帮助读者快速建立问题抽象、方法路径与工程落地方式。
## 关键步骤

1. 明确输入/输出与任务边界。
2. 按模块主流程执行核心算法或系统步骤。
3. 记录指标并做对比分析，形成可复用结论。
## 关键公式（逻辑表达）

\[
\text{Result} = \text{Core Method}(\text{Input}, \text{Config}, \text{Constraints})
\]

符号说明：
- \(\text{Input}\)：任务输入。
- \(\text{Config}\)：训练或推理配置。
- \(\text{Constraints}\)：方法约束（如资源、稳定性或安全边界）。
## 关键步骤代码（纯文档示例）

```python
# 关键流程示意（与具体工程实现解耦）
state = init_state()
for step in range(num_steps):
    state = step_update(state)
metrics = evaluate(state)
```
