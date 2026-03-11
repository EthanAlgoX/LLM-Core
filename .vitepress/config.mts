import mathjax3 from "markdown-it-mathjax3";
import { defineConfig } from "vitepress";

const rlSidebar = [
  { text: "概览", link: "/modules/01_foundation_rl/01_foundation_rl" },
  { text: "MDP", link: "/modules/01_foundation_rl/mdp/mdp" },
  { text: "TD Learning", link: "/modules/01_foundation_rl/td_learning/td_learning" },
  { text: "GAE", link: "/modules/01_foundation_rl/gae/gae" },
  { text: "Advantage", link: "/modules/01_foundation_rl/advantage/advantage" },
];

const architectureSidebar = [
  { text: "概览", link: "/modules/02_architecture/02_architecture" },
  {
    text: "LLM",
    items: [
      { text: "Transformer Core", link: "/modules/02_architecture/llm/llm" },
      { text: "Attention", link: "/modules/02_architecture/llm/attention" },
    ],
  },
  {
    text: "Generation",
    items: [
      { text: "Generation", link: "/modules/02_architecture/generation/generation" },
      { text: "Diffusion", link: "/modules/02_architecture/generation/diffusion/diffusion" },
      { text: "DiT", link: "/modules/02_architecture/generation/dit/dit" },
    ],
  },
  {
    text: "VLM",
    items: [
      { text: "VLM Overview", link: "/modules/02_architecture/vlm/vlm" },
      { text: "BLIP-2", link: "/modules/02_architecture/vlm/blip2/blip2" },
      { text: "LLaVA", link: "/modules/02_architecture/vlm/llava/llava" },
      { text: "Flamingo", link: "/modules/02_architecture/vlm/flamingo/flamingo" },
    ],
  },
];

const alignmentSidebar = [
  { text: "概览", link: "/modules/03_alignment/03_alignment" },
  { text: "SFT", link: "/modules/03_alignment/sft/sft" },
  { text: "PEFT", link: "/modules/03_alignment/peft/peft" },
  { text: "PPO", link: "/modules/03_alignment/ppo/ppo" },
  { text: "DPO", link: "/modules/03_alignment/dpo/dpo" },
  { text: "Policy Gradient", link: "/modules/03_alignment/policy_gradient/policy_gradient" },
  { text: "Actor-Critic", link: "/modules/03_alignment/actor_critic/actor_critic" },
  { text: "GRPO", link: "/modules/03_alignment/grpo/grpo" },
  { text: "RLHF", link: "/modules/03_alignment/rlhf/rlhf" },
  { text: "Data Synthesis", link: "/modules/03_alignment/data_synthesis/data_synthesis" },
  { text: "Data Engineering", link: "/modules/03_alignment/data_engineering" },
];

const advancedSidebar = [
  { text: "概览", link: "/modules/04_advanced_topics/04_advanced_topics" },
  { text: "Offline RL Overview", link: "/modules/04_advanced_topics/offline_rl/offline_rl" },
  { text: "BCQ", link: "/modules/04_advanced_topics/offline_rl/bcq/bcq" },
  { text: "CQL", link: "/modules/04_advanced_topics/offline_rl/cql/cql" },
];

const engineeringSidebar = [
  { text: "概览", link: "/modules/05_engineering/05_engineering" },
  { text: "CUDA", link: "/modules/05_engineering/cuda/cuda" },
  { text: "DeepSpeed", link: "/modules/05_engineering/deepspeed/deepspeed" },
  { text: "Inference", link: "/modules/05_engineering/inference/inference" },
  { text: "Megatron", link: "/modules/05_engineering/megatron/megatron" },
  { text: "Mixed Precision", link: "/modules/05_engineering/mixed_precision/mixed_precision" },
];

const agentSidebar = [
  { text: "概览", link: "/modules/06_agent/06_agent" },
  { text: "Frameworks", link: "/modules/06_agent/frameworks/frameworks" },
  { text: "Memory & RAG", link: "/modules/06_agent/memory_rag/memory_rag" },
  { text: "Multi-Agent", link: "/modules/06_agent/multi_agent/multi_agent" },
  { text: "OpenClaw", link: "/modules/06_agent/openclaw/openclaw" },
  { text: "Orchestration", link: "/modules/06_agent/orchestration/orchestration" },
];

const classicSidebar = [
  { text: "概览", link: "/modules/07_classic_models/07_classic_models" },
  { text: "ChatGPT", link: "/modules/07_classic_models/chatgpt/chatgpt" },
  { text: "DeepSeek-R1", link: "/modules/07_classic_models/deepseek_r1/deepseek_r1" },
  { text: "Qwen3", link: "/modules/07_classic_models/qwen3/qwen3" },
];

const guideSidebar = [
  { text: "导航", link: "/guides/NAVIGATION" },
  { text: "学习路径", link: "/guides/LEARNING_PATH" },
  { text: "术语表", link: "/guides/TERMINOLOGY" },
  { text: "文档规范", link: "/guides/DOC_STYLE" },
  { text: "模板", link: "/guides/MODULE_DOC_TEMPLATE" },
];

const briefSidebar = [
  { text: "总览", link: "/briefs/README" },
  { text: "Advantage", link: "/briefs/advantage" },
  { text: "PPO", link: "/briefs/ppo" },
  { text: "GRPO", link: "/briefs/grpo" },
  { text: "Diffusion", link: "/briefs/diffusion" },
  { text: "DeepSpeed", link: "/briefs/deepspeed" },
];

export default defineConfig({
  title: "LLM-Core",
  description: "AI / LLM / Agent 核心知识网站",
  lang: "zh-CN",
  srcExclude: [
    ".conda/**",
    "apps/**",
    "html/**",
    "node_modules/**",
    "README.md",
    "scripts/**",
  ],
  rewrites: {
    "docs/:page": "guides/:page",
    "output/technical_briefs/:page": "briefs/:page",
    "output/template_compliance_report": "reports/template-compliance-report",
  },
  cleanUrls: true,
  lastUpdated: true,
  sitemap: {
    hostname: "https://your-site.vercel.app",
  },
  head: [["link", { rel: "icon", href: "/logo.svg" }]],
  markdown: {
    math: true,
    config(md) {
      md.use(mathjax3);
      const fence = md.renderer.rules.fence;
      md.renderer.rules.fence = (...args) => {
        const [tokens, idx] = args;
        const token = tokens[idx];
        if (token.info.trim() === "mermaid") {
          return `<div class="mermaid">${md.utils.escapeHtml(token.content)}</div>`;
        }
        return fence ? fence(...args) : "";
      };
    },
  },
  themeConfig: {
    logo: "/logo.svg",
    nav: [
      { text: "首页", link: "/" },
      { text: "导航", link: "/guides/NAVIGATION" },
      { text: "RL 基础", link: "/modules/01_foundation_rl/01_foundation_rl" },
      { text: "Architecture", link: "/modules/02_architecture/02_architecture" },
      { text: "Alignment", link: "/modules/03_alignment/03_alignment" },
      { text: "Agents", link: "/modules/06_agent/06_agent" },
    ],
    search: {
      provider: "local",
    },
    outline: {
      level: [2, 3],
      label: "页面目录",
    },
    docFooter: {
      prev: "上一篇",
      next: "下一篇",
    },
    socialLinks: [{ icon: "github", link: "https://github.com/EthanAlgoX/LLM-Core" }],
    editLink: {
      text: "在 GitHub 上编辑此页",
      pattern: "https://github.com/EthanAlgoX/LLM-Core/edit/main/:path",
    },
    sidebar: {
      "/guides/": guideSidebar,
      "/modules/01_foundation_rl/": rlSidebar,
      "/modules/02_architecture/": architectureSidebar,
      "/modules/03_alignment/": alignmentSidebar,
      "/modules/04_advanced_topics/": advancedSidebar,
      "/modules/05_engineering/": engineeringSidebar,
      "/modules/06_agent/": agentSidebar,
      "/modules/07_classic_models/": classicSidebar,
      "/briefs/": briefSidebar,
      "/reports/": [
        { text: "模板合规报告", link: "/reports/template-compliance-report" },
      ],
    },
    footer: {
      message: "Built with VitePress and deployed on Vercel.",
      copyright: "Copyright © 2026 EthanAlgoX",
    },
  },
});
