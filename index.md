---
layout: home

hero:
  name: LLM-Core
  text: 面向 AI / LLM / Agent 的系统知识网站
  tagline: 基于 Markdown 编写，使用 VitePress 构建，通过 Vercel 持续部署。
  actions:
    - theme: brand
      text: 开始阅读
      link: /guides/NAVIGATION
    - theme: alt
      text: 查看学习路径
      link: /guides/LEARNING_PATH

features:
  - title: RL Foundation
    details: 从 MDP、TD Learning、GAE、Advantage 建立强化学习基础。
    link: /modules/01_foundation_rl/01_foundation_rl
  - title: Architecture
    details: 系统梳理 Transformer、Attention、VLM、Diffusion、DiT。
    link: /modules/02_architecture/02_architecture
  - title: Alignment
    details: 覆盖 SFT、PEFT、PPO、DPO、GRPO、数据合成与评估闭环。
    link: /modules/03_alignment/03_alignment
  - title: Engineering
    details: 聚焦 CUDA、并行训练、推理框架、混合精度与系统优化。
    link: /modules/05_engineering/05_engineering
  - title: Agents
    details: 从 RAG、编排、多智能体到 OpenClaw 架构设计。
    link: /modules/06_agent/06_agent
  - title: Classic Models
    details: 通过 ChatGPT、DeepSeek-R1、Qwen3 复盘工业级模型路径。
    link: /modules/07_classic_models/07_classic_models
---

## 使用方式

- 本地开发：`npm run docs:dev`
- 生产构建：`npm run docs:build`
- 部署平台：Vercel

## 推荐入口

- [章节导航](/guides/NAVIGATION)
- [学习路径](/guides/LEARNING_PATH)
- [RL 基础](/modules/01_foundation_rl/01_foundation_rl)
- [架构总览](/modules/02_architecture/02_architecture)
- [对齐总览](/modules/03_alignment/03_alignment)
