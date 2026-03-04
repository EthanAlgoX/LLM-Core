# Agent Frameworks（常见框架与示例代码）

> [!TIP]
> **一句话通俗理解**：Agent 框架是把“模型 + 工具 + 状态 + 编排”打包成可维护工程系统的脚手架。

## 定义与目标

- **定义**：Agent Framework 是用于构建、编排、调试和部署智能体系统的软件框架。
- **目标**：降低 Agent 工程复杂度，提供统一的工具调用、状态管理和执行流控制能力。

## 适用场景与边界

- **适用场景**：多步骤任务、需要工具调用、需要状态回溯或多角色协作的系统。
- **不适用场景**：单轮问答或简单提示词任务（直接 API 调用通常更简单）。
- **使用边界**：框架不会自动提升模型能力，本质上是工程组织方式与运行时能力增强。

## 关键步骤

1. 明确任务类型（单 Agent / 多 Agent / 工作流图）。
2. 选择框架（线性链、状态图、多角色协作）。
3. 定义工具协议（输入、输出、超时、重试）。
4. 建立状态与日志（可追踪、可回放、可观测）。

## 关键公式

$$
\text{Agent Runtime} = \text{Model} + \text{Tools} + \text{State} + \text{Workflow}
$$

符号说明：
- `Model`：底层推理模型。
- `Tools`：外部能力（检索、数据库、API、执行器）。
- `State`：任务中间状态与上下文记忆。
- `Workflow`：执行流程（链式、图式、协作式）。

## 常见框架概览

| 框架 | 核心定位 | 优势 | 局限 | 适用场景 |
| --- | --- | --- | --- | --- |
| LangChain | 组件化链式编排 | 上手快、生态广 | 复杂流程可维护性一般 | 快速原型、工具增强问答 |
| LangGraph | 状态机/图工作流 | 支持回路、分支、回滚 | 图设计复杂度较高 | 长流程 Agent、可恢复执行 |
| AutoGen | 多 Agent 对话协作 | 角色协作自然 | 对话成本与调试成本较高 | 评审-执行-复核等协作任务 |
| CrewAI | 角色与任务驱动 | 任务分工清晰 | 复杂状态管理需额外设计 | 团队式任务编排 |
| OpenAI Agents SDK / Responses Tool Calling | API 原生工具调用与运行 | 与模型能力一致、接入简单 | 高级编排需自行补齐 | 生产 API Agent 与工具调用 |

## 关键步骤代码（纯文档示例）

```python
# LangChain: 单 Agent + 工具最小闭环
from langchain.agents import create_tool_calling_agent, AgentExecutor

llm = create_llm_client()
tools = [search_tool, calculator_tool]
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt_tpl)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": "比较两种部署方案的月成本"})
```

```python
# LangGraph: 图状态编排（可分支/回路）
from langgraph.graph import StateGraph

workflow = StateGraph(dict)
workflow.add_node("plan", plan_step)
workflow.add_node("act", tool_step)
workflow.add_node("reflect", reflect_step)
workflow.add_edge("plan", "act")
workflow.add_edge("act", "reflect")
workflow.add_conditional_edges("reflect", route_next)
app = workflow.compile()
state = app.invoke({"goal": "生成发布检查清单"})
```

```python
# AutoGen: 多角色协作
planner = AssistantAgent(name="planner", system_message="负责拆解任务")
engineer = AssistantAgent(name="engineer", system_message="负责执行与实现")
reviewer = AssistantAgent(name="reviewer", system_message="负责质量审查")

group = GroupChat(agents=[planner, engineer, reviewer], max_round=8)
manager = GroupChatManager(groupchat=group)
manager.run("实现一个带缓存与重试的天气查询工具")
```

```python
# CrewAI: 角色 + 任务
researcher = Agent(role="Researcher", goal="收集候选方案")
builder = Agent(role="Builder", goal="输出可运行方案")

task1 = Task(description="调研 3 种 Agent 框架并给出取舍", agent=researcher)
task2 = Task(description="基于结论实现最小可运行 Demo", agent=builder)
crew = Crew(agents=[researcher, builder], tasks=[task1, task2])
output = crew.kickoff()
```

```python
# OpenAI Tool Calling: 原生函数调用循环
messages = [{"role": "user", "content": "查询北京天气并给出穿衣建议"}]
while True:
    resp = client.responses.create(model="gpt-4.1", input=messages, tools=tool_specs)
    tool_calls = extract_tool_calls(resp)
    if not tool_calls:
        print(resp.output_text)
        break
    for call in tool_calls:
        tool_result = run_tool(call)
        messages.append(tool_result_to_message(call, tool_result))
```

## 工程实现要点

- 先做最小闭环（1 个模型 + 1 个工具 + 1 条流程），再扩展为多工具/多角色。
- 为工具层统一错误模型（超时、429、5xx）和重试策略。
- 建立可观测性：每一步输入、输出、耗时、工具调用都要可追踪。

## 常见错误与排查

- **症状**：Agent 频繁误用工具。  
  **原因**：工具描述不清晰或参数 schema 约束太弱。  
  **解决**：补强工具描述、增加参数校验与示例。
- **症状**：多 Agent 对话轮数过多、成本失控。  
  **原因**：缺少终止条件或角色边界重叠。  
  **解决**：设置轮数上限、明确角色职责与交接条件。
- **症状**：框架迁移成本高。  
  **原因**：业务逻辑与框架 API 强耦合。  
  **解决**：将业务逻辑抽到框架无关层，仅保留薄适配层。

## 与相近方法对比

| 方法 | 优点 | 局限 | 适用场景 |
| --- | --- | --- | --- |
| 基于框架构建 Agent | 迭代快、可维护性好 | 学习曲线与依赖管理成本 | 中大型 Agent 系统 |
| 纯手写调用循环 | 控制力高、依赖少 | 可观测性和扩展性弱 | 小型 PoC 或短流程任务 |
| 工作流平台低代码编排 | 交付快、协作方便 | 深度定制能力有限 | 业务流程自动化场景 |

## 参考资料

- [LangChain](https://python.langchain.com/docs/introduction/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [AutoGen](https://microsoft.github.io/autogen/)
- [CrewAI](https://docs.crewai.com/)
- [OpenAI API Tool Calling](https://platform.openai.com/docs/guides/function-calling)
