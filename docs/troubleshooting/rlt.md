# RLT / RLinf 后端：问题与兼容边界

[问题索引](README.md) · [运行流程](../workflows/rlt/README.md) · [架构说明](../architecture/rlt.md)

这里整理已有模块 README、Pi0.5 教程和实现校验中的限制；它们不是一次真实机器人训练的故障日志。普通 Pi0.5 性能问题见[推理排错](pi05-performance.md)。

## RLinf 后端与 LoRA

| 现象 / 条件 | 核对与处理 |
| --- | --- |
| MEM 配置被拒绝 | RLinf 后端没有实现 MEM；用不含 MEM 的完整 Pi0.5 checkpoint 对比后端，不能静默丢弃 MEM 参数 |
| 裸 RLinf checkpoint 配置加载失败 | 裸 `config.json` 不是 LeRobot 配置；显式提供匹配 horizon / features 的 `PI05Config`，使用对应 LeRobot processors / stats |
| strict loading 缺键、额外张量或形状错误 | 原生网络与转换后的全部有效张量都要匹配；其他 RLinf task-specific heads 不受支持，不应靠关闭 strict 绕过 |
| 跨后端加载未合并 adapter | 先保留 base 并合并为 full / merged checkpoint，再做转换；raw converter 不转换两种网络布局的未合并 LoRA |
| fused vision LoRA 的 dropout / bias 报错 | 这一路径要求 `lora_dropout=0`、`lora_bias=false`，初次包装、加载或增加 adapter 都检查；语言 MLP adapter 仍支持自己的 activation dropout |
| 两后端动作不完全相同 | RLinf 保留 exact GELU，参考 OpenPI 为 tanh 近似，混合精度 / 时间特征也有差异。应测量动作误差与任务表现，不能承诺逐位一致 |

关于 GELU 的上游背景保留于[原说明](https://github.com/RLinf/RLinf/issues/1725)。转换 parity 测试会对齐源 activation 以隔离权重布局变换，不能据此宣称与未经改动的 LeRobot 模型完全等价。

## Stage 1 与 checkpoint

| 现象 / 条件 | 核对与处理 |
| --- | --- |
| 没有 RLT 权重时是否会自动初始化 | 仅明确声明的普通 LeRobot PI05 base 可自动初始化；没有 LeRobot 配置的裸 native base 必须在加载 API 显式 `initialize_rlt=True` |
| 保存配置启用 RLT，但权重部分 / 全部缺失 | 视为损坏或不匹配的训练 checkpoint，non-strict core loading 也不会放过；恢复完整 checkpoint |
| native RLT 权重结构不匹配 | `rlt_embed_dim`、层数、head、MLP ratio、prefix 容量等需与保存配置一致；不是加载时随意调小的内存开关 |
| prefix 超出容量 | 224px 每相机 256 tokens；`rlt_image_only=false` 时再加 `tokenizer_max_length`。3 相机 + 200 tokens 为 968，4 相机为 1224，必须被 `rlt_prefix_seq_len` 覆盖 |
| RLT 训练显存大 | 上游默认 width 2048、encoder/decoder 各 2 层、MLP ratio 4 合计约 7.47 亿参数；FP32 参数 / 梯度 / Adam moments 约 11.1 GiB，尚未包含 VLA、激活和联合 PEFT 的 frozen copies |
| 开启 gradient checkpointing 仍不够 | 它只减少保存的激活，参数和优化器占用不变；更改结构需重新训练匹配模型，不能当作已有 checkpoint 的无损降配 |
| encoder-only 导出不能续训 | `rlt_inference_only=true` 的导出缺重建 decoder，只用于动作推理 / 特征提取；Stage 1 续训使用完整训练 checkpoint |

上述内存数值是教程中的结构估算，不是本机全尺寸训练实测值。

## Stage 2 与远端 RTC

| 现象 / 条件 | 核对与处理 |
| --- | --- |
| frozen feature identity 不匹配 | 当前 fingerprint 为 v2，覆盖模型、processors 和 bundled tokenizer。旧 identity 不能证明实际 tokenizer 来源，应重新计算匹配配置并重新采集 replay，不手工改名冒充兼容 |
| 缺 tokenizer bundle 或包含目录符号链接 | 旧 processors 需重新导出 tokenizer；目录符号链接需物化。Hub 输入需固定完整 commit revision |
| actor / replay 需要再次归一化 | 不应再次归一化；它们已经使用 frozen VLA 的坐标。`feature_model_id` 和形状必须匹配 |
| learner 没有可训练输入 | `--replay` 需非空 replay 文件或含已完成窗口的单一 session 目录；不是 LeRobotDataset / RECAP sidecar |
| `Legacy partial RLT replay` / 缺少 `planned_action` | replay v2 每窗口保存完整计划与实际回执。v1 仅每窗口均执行满 K 步时可自动升级；旧 replay 中不足 K 步的终止 / 截断窗口缺未执行计划，须重新采集，不能用零尾部冒充计划 |
| `Legacy RLT critic/optimizer checkpoints cannot resume` | 修正后的 Q 输入是完整 K 步动作，旧 critic / optimizer checkpoint 不能直接续训。用兼容 replay 和 `--actor-checkpoint .../actor` 仅保留 actor，critics / optimizers 新建；[迁移命令](../workflows/rlt/README.md#2-从已有-replay-训练或续训) |
| `actor-checkpoint` / `hidden-dims` 不匹配 | 仅与 `--replay` 配合；保存的 actor 与 replay 须具有相同 frozen features、窗口和 RTC 约定。保留 actor 的维度与探索配置，显式 `--hidden-dims` 必须与保存值一致 |
| learner 续训参数被拒绝 | `--resume` 与 `--replay` 互斥；输出目录需新建；`--steps` 为追加更新数，`--discount` 不能覆盖续训中保存的值 |
| remote setup 被拒绝 | 仅支持 deployment 的 guided PI05；两端协议、policy fps、window / prefix 一致，满足 `0 < prefix_steps < execution_horizon`，模型 chunk 与客户端 `actions_per_chunk` 都覆盖 window + prefix |
| Stage 1 / actor 不适合远端加载 | Stage 1 必须 full / merged、RLinf 后端、启用 RLT，保存正确动作顺序与 processors；actor 必须属于同一 feature identity |
| 同步 Stage 2 actor 用于 RTC 时 schema 不匹配 | 仅可显式 `warmstart_stage2_actor=true` 做 policy-only 初始化；新增首层输入列置零，RTC critics / optimizers 从新状态开始。完整 replay / optimizer checkpoint 不跨 schema 续训 |
| reward provider 缺失 / 无效 | 选择 `manual` 或提供 `module:function` 回调返回 `RewardSignal`；加载校验在连接机器人前完成，执行步骤必须报告实际发送命令 |
| 预热测时超过 prefix 预算 | 启动有两次无运动预热和至少一次测量，最大 RTT 要小于 `prefix_steps / fps - budget_margin_s`。默认 30 Hz、3 步和 5 ms 余量为 95 ms；降低延迟、增大仍小于 window 的 prefix 或同时降低两端 policy fps |
| `RLT cold observation expired` | 首个计划必须在客户端 `observation_max_age_s` 内开始执行；从初始观测中最老的传感器时间计时，包含推理和电机锁等待。核对相机 / 观测采集时间、实际推理时延及电机锁占用；增大 RPC `request_timeout_s` 不会绕过该限制 |
| `fresh post-command sensors` / `reward observation predates` | 奖励观测的状态采集时间与最老相机时间，都必须不早于最后一次插值电机发送完成，且满足 `observation_max_age_s`。检查后台观测采集和相机帧更新；不能用旧图像搭配新状态。失败时动作可能已发送，但该步不调用奖励回调、不生成回执，需按执行失败恢复 |
| 推理失败后 prefix 未执行完 | prefix 的每个 policy step 开始前检查已完成响应，失败、异常或非法响应会中止后续策略步；若在当前步执行中收到失败，下一策略步才停止，该检查不在电机插值子步内 |
| 运行中超时、复位或取消 | 客户端不会延长已承诺 prefix 或静默重放不完整窗口；未完成数据丢弃，重新开始 episode。时间截断与真正 terminated 不同，需提供复位前最终观测 |

feature fingerprint 的 v2 namespace 与 replay / learner checkpoint 的 `format_version=2` 分别校验特征身份和存储 / critic 约定；满足其中之一并不保证另一项兼容。

同步采集的动作版本更新只能发生在窗口之间；过期版本或中途替换被拒绝。`RobotWindowStep` 要求调用者独占已连接的 RobotIO，并提供观测准备 / 奖励回调；不会替调用者连接、复位或回零。取消 / deadline 在等待电机锁后、驱动写入前检查，不能撤销驱动已经开始的写操作。

当前没有普通 RTC 内自动在线 learner、分布式奖励 / actor-weight 广播服务；远端 RLT 不接入内置 DAgger 人工接管、RECAP 或 trained RTC。原生上游 RLinf Stage 2 MLP checkpoint 与本地 actor 参数布局不互换，sharded Stage 2 训练未验证。

## 验证边界

已有验证覆盖 miniature 真实网络的 mask / 梯度、processors round trip、完整 / PEFT 加载合并、encoder-only 导出、optimizer resume；Stage 2 覆盖真实 `RLTrainer` 更新、奖励 / mask、relative-action round trip、取消 / 复位和精确恢复。远端测试使用真实本地 gRPC 与 mock robot，覆盖 policy / motor 步数、回执、延迟结果、等待电机锁期间复位、journal 训练 / 续训，以及 miniature guided-sampler parity 和 CUDA Inductor 编译。

[逻辑回归测试](../../tests/rl/test_rlt_logic_fixes.py)还覆盖短成功窗口回报的跨窗口 bootstrap、实际执行前缀与计划尾部组合、v1 replay 升级边界、旧 actor 初始化新 critic，以及推理失败和冷启动观测过期时拒绝后续动作。

[运行时契约测试](../../tests/rl/test_rlt_runtime_contracts.py)覆盖一次 fingerprint 检查后的加载兼容性、独立动作坐标快照和最终回执不再解码新计划；[观测时序测试](../../tests/async_inference/test_rlt_freshness.py)覆盖最后一次电机发送后的状态 / 相机时间，以及开始与复位时的手工反馈接收边界。

这些测试验证软件契约，不代表已经测得 Piper 任务成功率、全尺寸训练显存或目标服务器全模型时延。
