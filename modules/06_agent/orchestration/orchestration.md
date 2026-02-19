# 智能体编排范式 (Agent Orchestration)

> [!TIP]
> **一句话通俗理解**：Agent 的"大脑逻辑"：先思考再行动，遇到新情况随时调整计划

智能体编排定义了 Agent 如何处理任务流程、状态管理和决策逻辑。

---

## 核心模式

### 1. ReAct (Reason + Act)

- **核心思想**：协同推理与行动
- **流程**：Thought → Action → Observation → Thought...
- **适用场景**：交互频繁、不确定性高的任务
- **优势**：动态调整计划，适应环境反馈

### 2. Plan and Execute

- **核心思想**：先计划后执行
- **流程**：Planner 生成完整 Task List → Executor 逐一执行
- **适用场景**：逻辑清晰、长链条的复杂工程任务
- **优势**：全局视角，便于优化和回溯

### 3. CoT (Chain of Thought)

- **核心思想**：思维链引导
- **适用场景**：复杂推理、数学问题
- **变体**：Self-Consistency、Tree of Thoughts

### 4. Self-Ask

- **核心思想**：模型在回答前先自我询问子问题
- **适用场景**：复杂事实检索、多跳问答

---

## 工具增强 (Tool-augmented)

- **Function Calling**：通过结构化调用驱动外部能力
- **Tool Definitions**：定义工具 Schema，控制执行边界
- **Tool Selection**：根据任务动态选择最合适的工具

---

## 框架对比

| 框架 | 核心设计哲学 | 适用场景 |
| :--- | :--- | :--- |
| LangChain | LCEL 线性链 | 快速原型 |
| LangGraph | 状态机图 | 复杂流程、需要回退 |
| OpenClaw | 文件系统+总线 | 本地化执行 |
