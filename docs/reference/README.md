# 参数与经验参考

[返回文档首页](../README.md) · [流程指令](../workflows/README.md)

这里存放跨流程的参考材料；完整启动命令在各流程页维护。

| 想了解什么 | 文档 |
| --- | --- |
| 怎样采到一致、可用的演示 | [数据质量](data-quality.md) |
| 策略与显存如何匹配 | [策略选择与原有性能快照](policy-selection.md) |
| steps、epoch、batch 和学习率如何配合 | [训练预算](training-budget.md) |
| checkpoint 统计量、干预采样和 padding loss | [训练架构与语义](../architecture/training.md) |
| 历史 SFT 实验为什么使用某个步数 | [研究与实验记录](../research/README.md) |

经验值保留了原指南的适用条件，只用于确定初始试验范围；实际参数以数据规模、显存和评估结果为准。
