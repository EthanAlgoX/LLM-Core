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

Agent 能力的延伸。LLM 通过生成 JSON 格式的函数调用指令，驱动外部 API（搜索、计算、执行代码）获取外部实时信息或改变环境状态---

## 🏛️ 架构案例：NanoBot 设计模式

NanoBot 是一个极简主义（~4k 行代码）但功能全备的 Agent 架构参考。

### 1. 动态上下文（Dynamic Context）

- **核心思想**：不一次性载入所有指令。
- **实现**：System Prompt 只包含身份摘要，具体的工具说明（SKILL.md）存放在文件系统中。Agent 在需要时通过 `read_file` 主动读取，最大化节省 Context Window。

### 2. 多层记忆体系

- **事实记忆 (`MEMORY.md`)**：存储结构化或半结构化的事实（如：用户偏好、项目路径）。
- **叙事历史 (`HISTORY.md`)**：将旧的对话记录总结并归档。当需要召回时，Agent 利用 `grep` 搜索历史日志，实现极低成本的长程记忆。

### 3. 系统总线与解耦

- **MessageBus**：作为神经中枢，解耦了交互渠道（Telegram, 飞书, CLI）与核心逻辑层。
- **Provider Registry**：统一了不同 LLM 后端（OpenAI, DeepSeek, Claude）的调用接口。

### 4. 任务外包 (Subagents)

- **Delegation**：主智能体通过 `SpawnTool` 派生子智能体执行背景任务，子智能体完成后发送系统消息（Bus Message）回传，主智能体收到信号后再总结给用户。

---

### 📂 模块实战

本目录下包含：

- `code/react_demo.py`：一个不依赖外部 API 的纯逻辑 ReAct 循环演示。
- `data/`：模拟工具及其元数据定义。
