# Pi0.5 RLinf 后端与 RLT 架构

[架构导航](README.md) · [运行指令](../workflows/rlt/README.md) · [问题与兼容边界](../troubleshooting/rlt.md)

本页集中原 RLinf 模块 README 的实现说明，并对照现有 RLT 教程标出模块边界。完整推导、Python 采集示例和参数仍见[Pi0.5 网站教程](../source/pi05.mdx#optional-rlinf-network-backend)。

## 后端与阶段边界

`PI05Config(backend="rlinf")` 在原 `pi05` 策略下选择 RLinf 的原生 PyTorch Gemma / SigLIP 网络；默认仍是 `backend="lerobot"`。两者共用 LeRobot 图像准备、tokenizer、归一化、relative actions、action queue 和远端 RTC 协议。这里没有导入 RLinf 自身的 observation transforms / runtime，不需要另装 RLinf、Ray、Hydra、OpenPI 或 JAX。

| 部分 | 输入 / 输出 | 负责范围 |
| --- | --- | --- |
| RLinf 网络后端 | 原 Pi0.5 观测 → action chunk | 网络布局、权重转换和既有 guided / trained RTC 采样 |
| RLT Stage 1 | 最后层 VLM prefix → 连续 `z_rl` | 训练 encoder / reconstruction decoder；不是动作 tokenizer |
| 同步 Stage 2 | 冻结特征、参考 chunk、执行回报 → 小型 actor-critic | 固定执行窗口、Q+BC 学习和完整 checkpoint |
| 远端 RLT（教程 Stage 3） | 上一窗口执行回执 + 下一边界观测 → 带承诺前缀的新计划 | deployment 客户端 / 服务器之间的重叠推理与 replay journal |

Stage 1 本身不采集奖励或启动在线学习。Stage 2 的同步采集 API 不接管普通 RTC 客户端。远端 RLT 需要两端显式启用 `rlt.enabled`，默认关闭；DAgger、RECAP 和 trained RTC 不属于该远端模式。

## RLinf 网络适配、精度与转换

上游来源、提交与许可证保留在[模块 README](../../src/lerobot/policies/pi05/rlinf/README.md#provenance)。`modeling.py` 负责网络装配和 LeRobot 采样适配，复用已有 guided / trained RTC integrator。

时间特征先在 FP32 计算，再转成 projection dtype；Euler 积分与返回动作保持 FP32。现有数值 profile 的含义继续保留：

| profile | VLA 参数 / 计算约定 |
| --- | --- |
| `lerobot` | 视觉路径保持 FP32 |
| `openpi` | RLinf 的选择性 BF16 / FP32 路径 |
| `checkpoint` | 将全部 VLA 参数转换到选定 dtype |

与原后端一致，`openpi` profile 或开启模型编译会把进程级 FP32 matmul precision 设为 `high`。模块内部单个 kernel 的自动编译默认关闭，由 `PI05Config.compile_model` 控制外层模型。采样复用 LeRobot 编译分派：fast guided RTC 在 compiled sampler 外准备 padding、mask 和 delay，固定观测形状 / 去噪步数时可复用不同 delay / prefix 长度的图；native/reference RTC 保留独立的 compiled prefix 和 denoiser 区域。

Adaptive RMSNorm 额外支持 `(batch, action_horizon, width)` conditioning，以适配 trained RTC；标量 conditioning 沿用上游路径。仅转换后端不会让普通权重获得 trained RTC 能力，仍需相应训练配置与权重。

权重加载接受原生 RLinf `img.*` / `llm.*` safetensors，也可在内存中转换 LeRobot / OpenPI PyTorch safetensors，保留源 dtype。转换处理 fused SigLIP QKV、转置 / 堆叠 Gemma MLP、RMSNorm / AdaRMS、共享输入 embedding；不用的 token-output heads 和 SigLIP pooling head 被省略，其余张量必须可识别。

源 checkpoint 不会被修改。保存后的 LeRobot checkpoint 使用 `backend="rlinf"`，网络权重位于 `model.*`。裸 RLinf checkpoint 的 `config.json` 不是 LeRobot policy config，需要显式传入匹配 action horizon / features 的 `PI05Config`，并使用与基线相同的 LeRobot processors / stats。未支持的配置、权重和数值差异见[兼容边界](../troubleshooting/rlt.md#rlinf-后端与-lora)。

## 混合 LoRA 与完整 action expert

`peft_mode="vlm_lora_expert_full"` 为语言和视觉 attention / MLP 添加 LoRA，同时完整训练并保存 action expert、adaptive norms、action projections 和 time MLP。VLM 基础权重保持冻结。

`peft_language_lora` / `peft_vision_lora` 分别选择 `none`、`attention`、`attention_mlp`（默认）；两者都为 `none` 时只训练 / 保存完整 expert。语言 MLP adapter 在激活上分别施加 gate / up / down 低秩投影，合并到原生 `w_gating` / `w_linear` 张量。

`peft.py` 的 fused vision adapter 每次 forward 使用临时有效权重，不反复 merge / unmerge，避免 BF16 冻结权重漂移；永久合并只应用一次 delta。policy 的 PEFT preparation hook 在初次包装和 `make_policy` 重载时安装该自定义分派。转换器接受 full / merged 权重，不负责跨网络布局转换未合并 adapter。

`merge_lora.py` 将 adapter 与已训练的完整 expert 导出为普通 LeRobot checkpoint，同时保存两个 processors，供 Piper / RTC 推理使用；使用 adapter 期间需保留原始 base checkpoint。fused vision 的 dropout / bias 限制见[排错](../troubleshooting/rlt.md#rlinf-后端与-lora)。

## Stage 1：连续特征与导出

`use_rlt=True` 显式启用 RLT。encoder 将最后层 VLM prefix 隐状态压缩为 `(batch, rlt_embed_dim)` 的 `z_rl`，训练时由 autoregressive decoder 重建 prefix。上游参数名得以保留，本地增加了逐样本 loss、encoder-only 导出，以及兼容 PEFT / FSDP module wrapper 的 encode forward 模式。

| 训练方式 | loss 与梯度 |
| --- | --- |
| `frozen_vla` | 冻结全部 VLA 并保持 eval，包括 expert / projections；只训练 encoder / decoder，不计算 action expert 或 flow-matching noise |
| `joint` | `rlt_loss + rlt_alpha * vla_loss`；RLT 输入为 detached VLM 特征，只有 action objective 更新 VLM LoRA 和 expert |

新 RLT 权重保持 FP32；外层 AMP 控制运算精度。`gradient_checkpointing` 同时覆盖 RLT encoder / decoder，包括 frozen-VLA 和 PEFT 副本的训练路径，评估 / 特征提取不使用 checkpointing。它减少保存的激活，不减少参数与优化器内存。

联合 PEFT 将整个 RLT 加入 `modules_to_save` 并完整训练 / 保存。既有 action padding 与 trained-RTC loss mask 保留。RLT loss 在每个样本内对有效 token / channel 平均，再对样本平均；`reduction="none"` 返回合并后的逐样本 loss。`rlt_use_mask=true` 排除缺失相机和 token padding；joint 且关闭 mask 时额外做 observation-only prefix pass，避免被完整 mask 的 query 特征依赖训练动作序列；默认 mask 路径复用联合 pass。

`extract_rlt_features(preprocessed_batch)` 使用训练时相同的最终 prefix 和保存的归一化 / tokenizer；与独立动作 API 分别调用时各做一次 prefix pass。`predict_action_chunk_with_rlt_features(batch)` 在一次 prefix pass 中返回 `(reference_chunk, z_rl)`，供后续 adapter 使用。

`export_rlt.py` 导出完整 VLA、RLT encoder、processors / tokenizer；若有 PEFT 则先合并，并移除 reconstruction decoder。encoder-only 导出可采样动作和提取特征，但拒绝 Stage 1 训练；完整训练 checkpoint 仍用于恢复训练。

## Stage 2：执行窗口、奖励与 replay

`rlt_actor` 的默认输出为 10 个 policy steps，tanh 动作和 replay 都使用 Stage 1 保存的归一化动作坐标。actor 观察参考 chunk 的前 10 行、`z_rl` 与归一化 proprioception；critics 观察特征、proprioception 与完整 K 步动作。实际执行时长及其有效步 mask 不作为 Q 的输入，使短窗口回报与完整窗口的 TD bootstrap 使用相同的动作空间。`observation.rlt`、`observation.state` 和 `observation.reference_chunk` 已处理，learner 不再次归一化。

`rlt_ac` 通过算法工厂接入 `RLTrainer`。actor 最小化 `-q_weight * Q1 + bc_weight * BC`，双 target critics 取较小 Q；普通步的 BC target 是 VLA reference，人工干预步用记录的执行命令。reference dropout 只作用于 actor 输入，没有 entropy-temperature optimizer。小型 actor / critic 以 FP32 训练，与 VLA profile 和外层 AMP 独立。

`FixedWindowRollout` 在一次窗口中不替换 reference；窗口结束后的同一新观测同时作为上一 transition 的 next state 和下一窗口的 state。同步特征 / reference 调用使用 eager 与 `rtc_config=None`，Stage 1 保存的 trained-RTC 训练配置可以保留。

replay v2 分别保存完整 `planned_action` 和控制器报告的**实际发送命令** `action`。后者包括平滑、限位与人工修改，不是已到达关节位置的测量值；命令通过窗口原始 relative-action anchor 转回归一化坐标。训练 critic 时，已执行前缀使用实际命令，未执行尾部保留原计划作为上下文，不把它当成执行回执。每个 policy step 有独立 reward，电机插值子步不增加 learner 的步数。

执行 `n` 步的 target 为 `sum(gamma**i * reward[i]) + (1-terminated) * gamma**n * min(Q_target)`。提前终止 / 时间截断时，实际动作回执与 reward 的未执行尾部补零；有效步 mask 用于回报、折扣和 BC，Q 仍评估完整动作。截断仍 bootstrap，必须提供复位前的最终观测。取消、断连、非法奖励或采集失败会丢弃未完整窗口，必须重新 `begin_episode`，不会伪造 terminal transition。

特征模型 fingerprint 固定 Stage 1 权重、配置、归一化资源与全部 bundled tokenizer 文件；当前为 v2 namespace。actor / replay / sampled batch 都携带身份约束。修改特征模型或 processors 需要匹配的新 actor config 和新 replay，具体不兼容条件见[排错](../troubleshooting/rlt.md#stage-2-与远端-rtc)。

远端启动由 `FrozenRLTFeatures.from_pretrained` 统一计算一次 Stage 1 fingerprint，再通过 `config_factory` 使用源配置与该身份创建 actor 约定。机器人动作顺序、actor 身份和 schema 在加载完整 VLA 前校验；已有 RTC actor 与显式同步 actor 迁移继续保留各自的加载规则。

完整 learner checkpoint 包含 actor、critics / targets、两个 optimizers、更新步数、replay ring、采样 RNG 和 Python / Torch RNG，当前格式版本为 v2。旧 critic 输入布局不能直接恢复；v1 replay 也只有每个窗口均执行满 K 步的数据可自动升级，迁移方式见[学习流程](../workflows/rlt/README.md#2-从已有-replay-训练或续训)。这里的格式版本与 feature fingerprint 的 v2 namespace 是两个独立约定。`algorithm.get_weights()` 只返回 actor 权重、身份与 learner version；同步 rollout owner 可在窗口间 `load_actor_weights`，检查完整快照、拒绝过期版本并记录到 replay，窗口中禁止替换。

## 远端重叠执行

默认远端窗口仍为 10 个 policy actions。稳态边界捕获新观测并上传上一窗口回执后，先执行上一个 frozen VLA reference 的 `[10:13]` 三行 continuation，同时服务器计算新 reference / 特征 / actor 输出；再用新输出完成剩余 7 行。已加载 actor 时，它控制首次冷窗口的全部 10 行和稳态窗口的后 7 行；未加载 actor 时，这些行使用 frozen VLA reference。continuation 不是额外 actor 预测，交接行为需另行评估任务表现。

远端 state 增加 `observation.rtc_prefix` / `observation.rtc_prefix_mask`；actor 固定已承诺行，critics 也观察决定下一 continuation 的 reference，以及已承诺 prefix / mask。该 mask 表示决策前已知的承诺行，与执行结束后得到的有效步 mask 不同。actor 的承诺行固定，不接收 Q 梯度，也不参与 BC；全为承诺行的 batch 跳过 actor 更新。每窗口独立保存归一化 / relative-action 坐标快照和完整 `planned_action`，下一次推理不会覆盖原 anchor 或改变旧计划尾部的坐标。

服务器与客户端同一时间只允许一个待处理请求，用 session / generation / window 校验回执。复位丢弃未完成数据、隔离旧 generation 并沿用 Piper planned reset / homing；新任务需结束并复位旧 rollout。终止 / 截断的最终回执仍编码真实 next state 和 reference、保存最后一个窗口，供截断时 bootstrap；服务器不再为该回执生成或解码新的电机计划，清空 pending 后返回 `RLTFinished`。启动测时、超时与 checkpoint 兼容规则见[排错](../troubleshooting/rlt.md#stage-2-与远端-rtc)。

每个 policy interval 用于奖励与回执的观测，都必须不早于该步最后一次电机插值 `send_action` 返回的时刻。状态采集时间和最老相机时间分别核对，且观测年龄不能超过 `observation_max_age_s`；连续采集模式等待后台 collector 的合格快照，非连续模式也执行相同的时间检查。超时会拒绝该步奖励 / 回执，此时电机动作可能已经发送。

continuation prefix 的每个 policy step 开始前检查已完成的推理结果；若已返回失败、异常或非法响应，就中止剩余的策略步。该检查不在电机插值子步内；若失败在当前策略步执行中到达，会在下一策略步开始前停止。首个冷计划还受初始观测新鲜度限制：从最老传感器时间到首次驱动写入必须在 `observation_max_age_s` 内，推理等待和电机锁等待都计入；即使 RPC request timeout 未到，也会拒绝过期观测触发的动作。

## 代码入口

| 目录 / 文件 | 职责 |
| --- | --- |
| [pi05/rlinf](../../src/lerobot/policies/pi05/rlinf/README.md) | 网络来源、转换、RLT Stage 1 与混合 LoRA 的后端实现 |
| [rlt_actor](../../src/lerobot/policies/rlt_actor/README.md) | actor 配置、模型、特征形状与 identity processor |
| [rl/algorithms/rlt_ac](../../src/lerobot/rl/algorithms/rlt_ac/) | Q+BC 算法、critics / targets 与优化器 |
| [rl/rlt](../../src/lerobot/rl/rlt/README.md) | 冻结特征、回放、同步采集、RTC planner、完整恢复 |
| [server/rlt_runtime.py](../../src/lerobot/async_inference/server/rlt_runtime.py) / [client/rlt_runtime.py](../../src/lerobot/async_inference/client/rlt_runtime.py) | deployment 中显式启用的远端 RLT worker 与执行器 |
| [shared/rlt_config.py](../../src/lerobot/async_inference/shared/rlt_config.py) / [shared/rlt_remote.py](../../src/lerobot/async_inference/shared/rlt_remote.py) | 两端参数与传输协议 |
| [train_rlt_stage2.py](../../examples/tutorial/pi05/train_rlt_stage2.py) | 已有 replay 的离线学习 / 续训 CLI |

这一实现已有软件契约测试，但不等于完成全尺寸训练、目标服务器时延或 Piper 成功率验收；范围集中记录在[验证边界](../troubleshooting/rlt.md#验证边界)。
