# Advanced Topics 分类

> [!TIP]
> **一句话通俗理解**：进阶模块聚焦“标准配方之外”的难题，核心是用更强约束解决更高风险场景。

## 定义与目标

- **定义**：Advanced Topics 汇总高复杂度、强假设依赖的进阶主题。
- **目标**：帮助读者识别“何时需要超出常规 LLM 训练范式”的方法。

## 适用场景与边界

- **适用场景**：高风险决策、交互成本高、在线试错代价大的任务。
- **不适用场景**：数据和评测基础薄弱的快速原型阶段。
- **使用边界**：进阶方法往往对数据覆盖和假设前提非常敏感。

## 关键步骤

1. 先定义问题为什么不能用标准在线训练解决。
2. 再验证数据与约束是否支撑进阶方法假设。
3. 最后用保守评估策略验证泛化与安全边界。

## 关键公式

$$
\text{Reliable Improvement} = \text{Method} + \text{Assumption Validation} + \text{Risk Control}
$$

符号说明：
- `Method`：具体进阶算法（如 Offline RL 的保守学习）。
- `Assumption Validation`：关键假设校验（数据覆盖、分布偏移）。
- `Risk Control`：风险约束（保守正则、离线评测、部署闸门）。

## 关键步骤代码（纯文档示例）

```python
dataset = load_offline_dataset()
assumption_ok = validate_coverage(dataset)
if assumption_ok:
    model = train_conservative_policy(dataset)
    metrics = run_offline_evaluation(model)
```

## 子模块导航

- [Offline RL 总览](./offline_rl/offline_rl.md)
- [CQL](./offline_rl/cql/cql.md)
- [BCQ](./offline_rl/bcq/bcq.md)

## 工程实现要点

- 先做离线评估再做有限上线，避免高风险直接部署。
- 训练日志中显式记录“分布外动作比例”和“保守项强度”。
- 结论必须附带数据分布前提，不做无条件外推。

## 常见错误与排查

- **症状**：离线指标优秀但线上退化严重。  
  **原因**：训练时忽略了部署分布和行为策略差异。  
  **解决**：增加分布偏移检测并提高保守约束。
- **症状**：方法难以复现。  
  **原因**：没有固定数据切分和评测口径。  
  **解决**：版本化数据、配置和评估脚本。

## 与相近方法对比

| 方法 | 优点 | 局限 | 适用场景 |
| --- | --- | --- | --- |
| 进阶约束方法（本模块） | 风险控制更强，适配高代价任务 | 实施复杂度高 | 医疗/金融/工业策略优化 |
| 常规在线 RL | 反馈闭环快 | 在线试错成本高 | 仿真环境充足场景 |
| 监督学习替代 | 实施简单 | 无法直接优化长期回报 | 静态预测任务 |

## 参考资料

- [Conservative Q-Learning](https://arxiv.org/abs/2006.04779)
- [Batch-Constrained deep Q-learning](https://arxiv.org/abs/1812.02900)
