# Agentic Reasoning (智能体推理)

> [!TIP]
> **一句话通俗理解**：让 AI 从"回答问题"升级为"主动做事"——思考、记忆、用工具、迭代执行

Agent 是大语言模型从“对话框”向“生产力工具”演进的核心形态。

> **核心逻辑**：Agent = LLM + Planning + Memory + Tool Use

---

## 核心组件解析

### 1. 规划 (Planning)

规划层决定了 Agent 如何拆分复杂任务。详见 [Agent Orchestration](./orchestration/orchestration.md)

### 2. 编排与状态机 (Orchestration & State Machine)

任务编排不仅是线性的，还需要容错、分支与异步处理。详见 [Agent Orchestration](./orchestration/orchestration.md)

### 3. 多智能体协作 (Multi-Agent Systems)

详见 [Multi-Agent Systems](./multi_agent/multi_agent.md)

### 4. 记忆与检索 (Memory & RAG)

详见 [Memory & RAG](./memory_rag/memory_rag.md)

### 5. 工具集成与人工干预 (Tools & HITL)

- **Tool-augmented**：通过 Function Calling 驱动外部能力。
- **Human-in-the-Loop (HITL)**：在关键动作前强制引入人工确认机制。

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

- `data/`：模拟工具及其元数据定义。

```python
# 关键步骤代码（纯文档示例）
state = {"goal": user_query, "scratchpad": []}

while not task_finished(state):
    thought = llm_reason(state)
    tool_call = decide_tool(thought)
    observation = run_tool(tool_call) if tool_call else None
    state["scratchpad"].append({"thought": thought, "observation": observation})

answer = llm_finalize(state)
```
