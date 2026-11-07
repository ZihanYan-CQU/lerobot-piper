# RLT actor 策略

`rlt_actor` 是接收冻结 RLT 特征、归一化状态与 VLA reference chunk 的小型策略。配置 / 特征约束在 `configuration_rlt_actor.py`，模型与权重版本检查在 `modeling_rlt_actor.py`；`processor_rlt_actor.py` 保持已归一化输入的 identity 处理。

- [训练与使用指令](../../../../docs/workflows/rlt/README.md)
- [actor、critic、replay 与远端执行架构](../../../../docs/architecture/rlt.md)
- [特征身份与 checkpoint 兼容边界](../../../../docs/troubleshooting/rlt.md)

critic 和 Q+BC 更新由 [`rlt_ac`](../../rl/algorithms/rlt_ac/) 实现，冻结特征与窗口采集见 [`rl/rlt`](../../rl/rlt/README.md)。本目录不负责加载真机或单独启动在线学习。
