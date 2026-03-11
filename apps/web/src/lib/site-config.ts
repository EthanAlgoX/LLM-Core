export const siteConfig = {
  name: "LLM-Core Academy",
  description: "付费解锁式 LLM 文档学习站点。",
  previewDocsPerModule: 2,
  plans: {
    pro: {
      name: "Pro Access",
      description: "解锁全部章节、完整公式与工程细节。",
      priceLabel: "¥99 / 月",
      features: [
        "浏览全部章节与技术口述稿",
        "不再受试读章节限制",
        "支持账号内持续解锁更新内容",
      ],
    },
  },
} as const;
