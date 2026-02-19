# Agentic Reasoning (智能体推理)

> [!TIP]
> **一句话通俗理解**：Agent 把语言模型升级为可执行系统：先规划、再调用工具、再反思修正，直到任务完成。

Agent 是大语言模型从“对话框”向“生产力工具”演进的核心形态。

## 定义与目标

- **定义**：Agent 是具备“规划-执行-观察-反思”闭环能力的 LLM 系统。
- **目标**：让模型从单轮回答升级为多步骤任务执行，并可调用外部工具完成真实操作。

## 适用场景与边界

- **适用场景**：用于任务拆解、工具编排、长期记忆与流程自动化场景。
- **不适用场景**：不适用于无需工具调用和状态管理的简单问答任务。
- **使用边界**：系统效果受工具可靠性、提示策略与反馈闭环影响。

## 关键步骤

1. 理解目标并拆分任务（Planning）。
2. 按步骤调用工具执行，并持续观察反馈（Act + Observe）。
3. 根据反馈修正计划（Re-plan）。
4. 在满足停止条件后汇总输出最终答案（Finalize）。

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

## 关键公式（逻辑表达）

`Agent = LLM + Planning + Memory + ToolUse`

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

## 关键步骤代码（纯文档示例）

```python
state = {"goal": user_query, "scratchpad": []}

while not task_finished(state):
    thought = llm_reason(state)
    tool_call = decide_tool(thought)
    observation = run_tool(tool_call) if tool_call else None
    state["scratchpad"].append({"thought": thought, "observation": observation})

answer = llm_finalize(state)
```

## 子模块导航

- `memory_rag`: 记忆与检索机制。
- `orchestration`: ReAct、Plan-and-Execute 与流程编排。
- `multi_agent`: 多智能体协作范式。
- `openclaw`: 本地优先 Agent 架构案例。

## 工程实现要点

- 先定义状态机与失败恢复路径，再扩展工具数量。
- 为关键动作设置审计日志与人工确认机制。
- 区分短期上下文与长期记忆，避免上下文膨胀。

## 常见错误与排查

- **症状**：Agent 循环执行或无法收敛。  
  **原因**：停止条件不明确或反馈信号噪声过大。  
  **解决**：引入明确终止条件与步数上限。
- **症状**：调用工具频繁失败。  
  **原因**：工具接口契约不一致或参数验证不足。  
  **解决**：统一 Tool schema 并增加输入校验与重试策略。

## 与相近方法对比

| 方法 | 优点 | 局限 | 适用场景 |
| --- | --- | --- | --- |
| 本文主题方法 | 紧贴本节问题定义 | 依赖数据与实现质量 | 适合结构化评测与迭代优化 |
| 对比方法A | 上手成本更低 | 能力上限可能受限 | 快速原型与基线对照 |
| 对比方法B | 上限潜力更高 | 调参与资源成本更高 | 高要求生产或复杂任务场景 |

## 参考资料

- [ReAct](https://arxiv.org/abs/2210.03629)
- [Toolformer](https://arxiv.org/abs/2302.04761)
