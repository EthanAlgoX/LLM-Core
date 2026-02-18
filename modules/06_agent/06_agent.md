# Agentic Reasoning (智能体推理)

Agent 是大语言模型从“对话框”向“生产力工具”演进的核心形态。

> **核心逻辑**：Agent = LLM + Planning + Memory + Tool Use

---

## 核心组件解析

### 1. 规划 (Planning)

规划层决定了 Agent 如何拆分复杂任务。常用模式包括：

- **ReAct (Reason + Act)**：协同推理与行动。模型根据 Observation 动态调整 Thought。适用于交互频繁、不确定性高的任务。
- **Plan and Execute**：先通过 Planner 生成完整 Task List，再由 Executor 逐一执行。适用于逻辑清晰、长链条的复杂工程任务。
- **CoT (Chain of Thought)**：思维链引导。
- **ToT (Tree of Thoughts)**：针对复杂决策，探索多条可能路径并进行回溯。
- **Self-Ask**：模型在回答前先自我询问子问题，适合复杂事实检索。

### 2. 编排与状态机 (Orchestration & State Machine)

任务编排不仅是线性的，还需要容错、分支与异步处理。

- **状态机编排 (State Machine)**：将 Agent 的不同执行阶段定义为“状态”，通过明确的转移条件驱动（如 **LangGraph**）。适用于生成可靠性要求极高的工程流。
- **异步编排 (Async Orchestration)**：利用 **Message Bus**（如 **OpenClaw Mesh**）解耦请求与响应。Agent 可以在后台异步执行，完成后通过总线反馈。
- **Conditional Routing**：根据 LLM 的输出结果动态路由到不同的子任务或不同能力的模型。

### 3. 多智能体协作 (Multi-Agent Systems)

- **去中心化协作 (Decentralized)**：智能体之间通过标准协议通信，如 **AutoGen**、**MetaGPT**。每个 Agent 拥有独立 Persona 与职责分布。
- **Swarm Intelligence**：通过大量轻量化 Agent 的涌现行为解决极其复杂的分布式问题。
- **通信协议与共识**：定义 Agent 间交换信息的 Schema，确保协作过程中的数据一致性。

### 4. 记忆与检索 (Memory & RAG)

- **短期记忆 (Short-term)**：利用 Context Window 存储对话历史和中间思考过程。
- **长期记忆 (Long-term) - RAG 架构**：
    1. **Query 理解**：意图识别、实体提取或查询扩展（Query Expansion）。
    2. **向量检索 (Vector Search)**：计算 **Embedding** 相似度，从向量库召回候选集。
    3. **Rerank 模型**：对召回结果进行精排，解决语义稀释问题。

### 5. 工具集成与人工干预 (Tools & HITL)

- **Tool-augmented**：通过 Function Calling 驱动外部能力。
- **Human-in-the-Loop (HITL)**：在关键动作（如写库、转账、发送敏感信息）前强制引入人工确认机制，提供调试与纠偏入口。

---

## 🛠️ 主流框架对比 (Frameworks)

| 框架 | 核心设计哲学 | 适用场景 |
| :--- | :--- | :--- |
| **LangChain** | 基于组件（LCEL）的线性链式调用。 | 快速原型开发，标准 RAG 流程。 |
| **LangGraph** | 基于状态机（Graph）的循环逻辑。 | 复杂多智能体协作、需要状态回退的场景。 |
| **OpenClaw** | 基于文件系统与总线（Bus）的深度嵌入。 | 极低功耗、本地化执行的智能体系统。详见 [OpenClaw 架构解析](./openclaw/openclaw.md) |

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
