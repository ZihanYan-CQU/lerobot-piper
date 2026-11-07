# 研究与实验记录

[返回文档首页](../README.md) · [操作流程](../workflows/README.md) · [代码架构](../architecture/README.md) · [问题排查](../troubleshooting/README.md)

这里保存论文对照、方案取舍、历史实验配方与后续建议。先看每篇的日期、来源和实现状态；计划或建议不代表仓库已经实现，也不代表实机验证通过。

| 主题 | 文档 | 用途 |
| --- | --- | --- |
| 本地 SFT2 数据审计、干预曝光预算和历史命令 | [训练实验记录](training-experiments.md) | 核对 2026-10-04/16 的数据与参数；新训练从流程文档开始 |
| Evo-RL 前两轮 SFT 的论文、公开模型与本项目差异 | [Evo-RL 对照（2026-10-16）](evo_rl_two_stage_sft_2026-10-16.md) | 区分论文协议、模型卡、实际配置和未确认部分 |
| 云端时延下的 A2C2 迁移设计 | [A2C2 调研（2026-10-15）](a2c2_cloud_latency_migration_2026-10-15.md) | 方案分析与迁移边界，按原文实现状态阅读 |
| RECAP 与 RLinf 的集成路线 | [RECAP / RLinf 调研（2026-10-13）](recap_rlinf_integration_2026-10-13.md) | 区分已实现采集基础与后续价值训练/强化学习路线 |

PI05 延迟和精度对齐属于实际诊断，已统一迁到 [问题记录](../troubleshooting/README.md#按日期保留的诊断记录)。旧研究路径只保留章节跳转。
