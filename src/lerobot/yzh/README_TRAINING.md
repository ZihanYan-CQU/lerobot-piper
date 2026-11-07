# 模型训练文档导航

原混合文档已按职责拆分。日常操作请直接查看 [模型训练流程指令](../../../docs/workflows/training/README.md)。

- [实现与代码架构](../../../docs/architecture/training.md)
- [问题定位与诊断](../../../docs/troubleshooting/training.md)
- [历史数据预算与对照配方](../../../docs/research/training-experiments.md)
- [全部文档](../../../docs/README.md)

## 原章节去向

<a id="1-先分清四件事"></a>

- [1. 先分清四件事](../../../docs/architecture/training.md#1-先分清四件事)

<a id="2-use_pretrained_stats决定模型使用哪把数值尺度"></a>

- [2. `use_pretrained_stats`：决定模型使用哪把数值尺度](../../../docs/architecture/training.md#2-use_pretrained_stats决定模型使用哪把数值尺度)

<a id="3-intervention_sampling_ratio先决定组再决定组内哪一帧"></a>

- [3. `intervention_sampling_ratio`：先决定组，再决定组内哪一帧](../../../docs/architecture/training.md#3-intervention_sampling_ratio先决定组再决定组内哪一帧)

<a id="4-不放回到底保证什么"></a>

- [4. “不放回”到底保证什么](../../../docs/architecture/training.md#4-不放回到底保证什么)

<a id="5-epoch组内一轮和训练-step-是不同的计数"></a>

- [5. epoch、组内一轮和训练 step 是不同的计数](../../../docs/architecture/training.md#5-epoch组内一轮和训练-step-是不同的计数)

<a id="6-用当前合并数据理解比例和曝光次数"></a>

- [6. 用当前合并数据理解比例和曝光次数](../../../docs/research/training-experiments.md#6-用当前合并数据理解比例和曝光次数)

<a id="7-常用命令"></a>

- [7. 常用命令](../../../docs/research/training-experiments.md#7-常用命令)

<a id="8-数据要求和启动日志"></a>

- [8. 数据要求和启动日志](../../../docs/troubleshooting/training.md#8-数据要求和启动日志)

<a id="9-对应代码"></a>

- [9. 对应代码](../../../docs/architecture/training.md#9-对应代码)

<a id="10-loss-来源与-padding-mask"></a>

- [10. Loss 来源与 padding mask](../../../docs/architecture/training.md#10-loss-来源与-padding-mask)

<a id="11-本地数据量与-25-epochs-的步数"></a>

- [11. 本地数据量与 2.5 epochs 的步数](../../../docs/research/training-experiments.md#11-本地数据量与-25-epochs-的步数)

<a id="12-当前采样器与-evo-rl-的差异"></a>

- [12. 当前采样器与 Evo-RL 的差异](../../../docs/research/training-experiments.md#12-当前采样器与-evo-rl-的差异)

<a id="13-是否支持一次-rollout-内多次接管"></a>

- [13. 是否支持一次 rollout 内多次接管](../../../docs/research/training-experiments.md#13-是否支持一次-rollout-内多次接管)
