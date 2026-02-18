#!/usr/bin/env python3
"""
Agent 核心原理：ReAct (Reason + Act) 循环演示。
该脚本模拟了 Agent 如何通过“思考-行动-观察”的闭环完成任务。
"""

import json
import time

def mock_search_tool(query: str) -> str:
    """模拟搜索工具"""
    knowledge = {
        "北京天气": "北京今天多云转晴，25摄氏度。",
        "LLM-Core项目": "这是一个专注于大模型核心原理学习的开源项目。",
    }
    return knowledge.get(query, "未找到相关信息。")

def mock_calculator_tool(expression: str) -> str:
    """模拟计算器工具"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"

TOOLS = {
    "search": mock_search_tool,
    "calculator": mock_calculator_tool
}

def mock_llm_api(prompt: str) -> str:
    """模拟 LLM 的 ReAct 步进响应"""
    # 场景1：查询天气
    if "查询北京的天气" in prompt and "Observation" not in prompt:
        return "Thought: 用户想知道北京的天气，我需要使用搜索工具。\nAction: search(\"北京天气\")"
    
    # 场景2：得到天气反馈后结束
    if "Observation: 北京今天多云转晴，25摄氏度。" in prompt:
        return "Thought: 我已经得到了北京的天气信息。\nFinal Answer: 北京今天多云转晴，气温 25 摄氏度。"
    
    return "Thought: 我需要更多信息。\nFinal Answer: 无法处理该请求。"

def run_agent_loop(query: str, max_steps: int = 3):
    print(f"🚀 用户请求: {query}\n" + "="*40)
    
    context = f"Question: {query}\n"
    
    for step in range(max_steps):
        print(f"\n[Step {step + 1}]")
        
        # 1. LLM 思考并输出 Action
        llm_response = mock_llm_api(context)
        print(f"🤖 LLM 响应:\n{llm_response}")
        
        if "Final Answer:" in llm_response:
            print(f"\n✅ 最终答案: {llm_response.split('Final Answer:')[1].strip()}")
            break
            
        # 2. 解析 Action 并调用工具
        if "Action:" in llm_response:
            action_line = llm_response.split("Action:")[1].strip()
            tool_name = action_line.split("(")[0]
            tool_input = action_line.split("(")[1].strip(")\"")
            
            print(f"🛠️ 执行工具: {tool_name}({tool_input})")
            observation = TOOLS[tool_name](tool_input)
            print(f"👁️ 观察结果: {observation}")
            
            # 3. 将观察结果喂回上下文
            context += f"{llm_response}\nObservation: {observation}\n"
        
        time.sleep(0.5)

if __name__ == "__main__":
    run_agent_loop("请帮我查询北京的天气")
