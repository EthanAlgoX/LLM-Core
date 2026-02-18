# Agentic Reasoning (智能体推理)

Agent 是大语言模型从“对话框”向“生产力工具”演进的核心形态。

> **核心逻辑**：Agent = LLM + Planning + Memory + Tool Use

---

## 核心组件解析

### 1. 规划 (Planning)

规划层决定了 Agent 如何拆分复杂任务。常用的模式包括：

- **CoT (Chain of Thought)**：思维链。让模型逐步思考，提高逻辑稳定性。
- **ReAct (Reason + Act)**：协同推理与行动。模型先生成一步“思考”（Thought），然后执行一个“动作”（Action），并根据“观察”（Observation）调整下一步。
- **ToT (Tree of Thoughts)**：针对复杂决策，探索多条可能路径并进行剪枝优化。

### 2. 记忆 (Memory)

- **短期记忆 (Short-term)**：即上下文（Context）。利用 LLM 的 Context Window 存储对话历史和中间思考过程。
- **长期记忆 (Long-term)**：通过 **RAG (Retrieval Augmented Generation)** 模式，将海量知识库存储在向量数据库（如 Chroma, Pinecone）中，按需检索相关信息。

### 3. 工具使用 (Tool Use / Function Calling)

Agent 能力的延伸。LLM 通过生成 JSON 格式的函数调用指令，驱动外部 API（搜索、计算、执行代码）获取外部实时信息或改变环境状态。

---

### 📂 模块实战

本目录下包含：

- `code/react_demo.py`：一个不依赖外部 API 的纯逻辑 ReAct 循环演示。
- `data/`：模拟工具及其元数据定义。
