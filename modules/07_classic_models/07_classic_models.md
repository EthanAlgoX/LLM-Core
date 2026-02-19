# Classic Models 分类

> [!TIP]
> **一句话通俗理解**：经典模型模块不是“背参数”，而是复盘“为什么这个技术路线在当时能赢”。

## 定义与目标

- **定义**：Classic Models 模块拆解具有里程碑意义的工业模型路线。
- **目标**：从案例中提炼可复用的“方法选择 + 工程取舍 + 产品化路径”。

## 适用场景与边界

- **适用场景**：技术复盘、路线评估、面试表达与方案设计。
- **不适用场景**：不适用于跨代模型的直接数值对比结论迁移。
- **使用边界**：必须结合发布时间、数据规模和评测口径解释结论。

## 关键步骤

1. 明确模型的发布背景与目标人群。
2. 拆解关键技术创新及其能力增益。
3. 分离算法收益与系统工程收益。
4. 总结可复用经验和不可复用条件。

## 关键公式

$$
\text{Practical Capability} = \text{Base Model} + \text{Post-Training} + \text{System Engineering}
$$

符号说明：
- `Base Model`：预训练基础能力。
- `Post-Training`：对齐、微调和偏好学习收益。
- `System Engineering`：推理与产品化系统收益。

## 关键步骤代码（纯文档示例）

```python
case = load_model_case("chatgpt_or_deepseek_or_qwen")
insights = extract_key_decisions(case)
comparison = compare_tradeoffs(insights)
```

## 子模块导航

- [ChatGPT / InstructGPT](./chatgpt/chatgpt.md)
- [DeepSeek-R1](./deepseek_r1/deepseek_r1.md)
- [Qwen3](./qwen3/qwen3.md)

## 工程实现要点

- 案例对比必须对齐版本和时间点，避免跨版本混淆。
- 除效果外，同时记录成本、延迟、吞吐与部署复杂度。
- 用“可迁移原则”总结，不做品牌绑定式结论。

## 常见错误与排查

- **症状**：对比结论相互矛盾。  
  **原因**：测试集、提示词模板或评测版本不一致。  
  **解决**：统一评测协议并显式记录模型版本。
- **症状**：案例结论无法指导项目落地。  
  **原因**：只有叙事，没有可执行的技术决策抽象。  
  **解决**：增加“可迁移条件 + 不可迁移条件”总结。

## 与相近方法对比

| 方法 | 优点 | 局限 | 适用场景 |
| --- | --- | --- | --- |
| 案例复盘法（本模块） | 能提炼真实工程决策逻辑 | 资料整理成本高 | 技术路线评审与面试表达 |
| 只看排行榜 | 结论直观 | 难解释“为什么好” | 快速筛选候选模型 |
| 单论文精读 | 深度高 | 缺少跨系统视角 | 深入单点技术研究 |

## 参考资料

- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
