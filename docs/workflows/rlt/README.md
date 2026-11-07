# RLT：特征训练、回放学习与远端采集

[流程导航](../README.md) · [架构与数据约定](../../architecture/rlt.md) · [问题与限制](../../troubleshooting/rlt.md)

本页集中整理现有 Pi0.5 教程中的 RLT 操作入口。完整参数解释和 Python 接口示例保留在[网站教程](../../source/pi05.mdx#rlt-stage-1-train-reload-and-export-rl-features)。RLT 使用冻结的 Pi0.5 特征训练小型 actor-critic；它与 [RECAP 完整任务记录](../recap/README.md)是不同的数据流程。

| 当前要做什么 | 从哪里开始 |
| --- | --- |
| 在任务模型上训练 RLT 特征 | [Stage 1](#1-stage-1训练与导出特征模型) |
| 用已采集的 latent replay 训练 / 续训 actor | [离线学习](#2-从已有-replay-训练或续训) |
| 用现有 deployment 客户端采集带奖励的远端执行窗口 | [远端 RLT](#3-远端-rlt-采集与部署) |
| 自己编写同步采集程序、对接任务奖励 | [同步采集接口](#4-同步采集接口与迁移) |

## 0. 环境与文件所在机器

所有命令在**当前机器的 LeRobot 仓库根目录**执行，使用根 `.venv`，不使用 Piper 仿真子项目的环境。环境清理与 uv 安装见[环境准备](../setup/README.md)。先同步依赖，日常命令使用 `uv run --no-sync`。

| 机器 / 用途 | 准备命令 | 文件位置 |
| --- | --- | --- |
| Pi0.5 Stage 1、导出及 RLT 学习机 | `uv sync --locked --inexact --extra pi05-training` | 该机器上的模型、数据和 learner 输出目录 |
| 混合 LoRA 训练或导出 PEFT checkpoint | 在上项基础上执行 `uv sync --locked --inexact --extra peft` | 保留 adapter 对应的原始 base checkpoint |
| 远端推理服务器 | `uv sync --locked --inexact --extra pi05-server` | Stage 1 模型、actor checkpoint、远端采集生成的 replay |
| 连接 Piper 的电脑 | `uv sync --locked --inexact --extra piper-client` | 相机 / CAN 配置和奖励回调模块 |

这是沿用已有 Pi0.5 角色环境的准备方式，不要求单独安装 RLinf、Ray、Hydra、OpenPI 或 JAX。同机承担多个角色时保留所需 extra 的并集；训练机与推理服务器不同时，把完整模型 / replay 复制到实际使用它们的机器，并修改相应路径。服务器和客户端必须使用配套版本代码。

## 1. Stage 1：训练与导出特征模型

起点是已有任务训练的 **full / merged Pi0.5 checkpoint**，包含匹配的 processors、统计量和 tokenizer。LoRA adapter 先按[混合微调与合并教程](../../source/pi05.mdx#vlm-lora-with-a-fully-trained-action-expert)处理。现有普通训练配方的 dataset、batch size、步数和输出目录继续沿用，不在这里另设一套任务训练参数。

两种训练方式的现有参数如下；这是追加到 `uv run --no-sync lerobot-train ...` 的参数片段，不能单独执行：

| 方式 | 需要设置 | 保留什么 |
| --- | --- | --- |
| 冻结 VLA，仅训练 RLT | `--policy.backend=rlinf --policy.use_rlt=true --policy.rlt_training_mode=frozen_vla` | 已有任务 checkpoint 和普通训练参数；移除 `--peft.*` |
| VLM LoRA + 完整 action expert 联合训练 | `--policy.backend=rlinf --policy.use_rlt=true --policy.rlt_training_mode=joint --policy.rlt_alpha=1.0` | 原混合 LoRA 配方及 PEFT 参数 |

冻结模式的完整 dtype、AMP、RLT 维度和 mask 参数见[Stage 1 参数段](../../source/pi05.mdx#rlt-stage-1-train-reload-and-export-rl-features)。该模式不需要奖励标签。RLT 结构参数必须与后续 checkpoint 匹配；默认模块较大，容量和内存边界见[排错](../../troubleshooting/rlt.md#stage-1-与-checkpoint)。Stage 1 的中断恢复仍用[普通训练续训](../training/resume.md)，不是下一节的小型 learner CLI。

将完整训练 checkpoint 导出为自包含推理目录：

```bash
uv run --no-sync python examples/tutorial/pi05/export_rlt.py \
  --checkpoint outputs/piper_rlt_stage1/checkpoints/last/pretrained_model \
  --output-dir outputs/piper_rlt_inference \
  --device cpu
```

输出包含完整 VLA、RLT encoder、processors 和 tokenizer；存在 adapter 时会先合并，重建 decoder 被移除，并标记 `rlt_inference_only=true`。该目录用于动作推理与特征提取，**保留完整训练 checkpoint 供 Stage 1 续训**。联合 PEFT 的 `merge_lora.py --rlt-encoder-only` 提供对应导出方式。

## 2. 从已有 replay 训练或续训

输入必须是 `FixedWindowReplayBuffer.save()` 生成的文件，或远端 RLT 的**一个 session 目录**。普通 LeRobotDataset、DAgger 数据集和 RECAP sidecar 不能直接替代它。

以下命令不加载 VLA 权重，也不连接机器人：

```bash
uv run --no-sync python examples/tutorial/pi05/train_rlt_stage2.py \
  --replay outputs/piper_rlt_replay.pt \
  --output-dir outputs/piper_rlt_stage2_0001 \
  --device cuda --steps 1000 --batch-size 256

uv run --no-sync python examples/tutorial/pi05/train_rlt_stage2.py \
  --resume outputs/piper_rlt_stage2_0001 \
  --output-dir outputs/piper_rlt_stage2_0002 \
  --device cuda --steps 1000 --batch-size 256
```

`--replay` 与 `--resume` 二选一；输出目录必须是新的。这里 `--steps=1000` 表示**追加 1000 次 learner 更新**，不同于普通 Pi0.5 续训的目标总步数。`--resume` 要指向完整输出目录，不能只给其中的 `actor/`；它恢复 actor、critics/targets、两套优化器、replay、采样与 Python/Torch RNG。`--discount` 只用于新训练，续训读取已保存的 discount。

当前 replay 和 learner checkpoint 的格式版本均为 v2。每个窗口均执行满 K 步的 v1 replay 可自动升级；旧 replay 中不足 K 步的终止 / 截断窗口缺少未执行部分的计划，必须重新采集。旧 critic / optimizer checkpoint 不能直接 `--resume`；若要保留旧 actor，使用兼容 replay 显式重新开始学习：

```bash
uv run --no-sync python examples/tutorial/pi05/train_rlt_stage2.py \
  --replay outputs/new_rlt_replay.pt \
  --actor-checkpoint outputs/old_rlt_checkpoint/actor \
  --output-dir outputs/rlt_corrected_0001 \
  --device cuda --steps 1000 --batch-size 256
```

`--actor-checkpoint` 只能配合 `--replay`，只加载 actor，critics / optimizers 从新状态开始。它保留 actor 的维度与探索配置，要求 replay 的 frozen features、窗口和 RTC 约定一致；若显式传 `--hidden-dims`，也必须匹配保存值。新的 v2 完整 checkpoint 可按前述命令续训。这与下一节跨同步 / RTC schema 的 `--rlt.warmstart_stage2_actor` 是不同入口。

远端采集后把第一条命令的输入换成 `--replay outputs/piper_rlt_rtc/SESSION_UUID`。session 位于推理服务器；在另一台学习机训练前先复制它。同步 Stage 2 和远端 RTC replay 的结构不同，不要混合或把前者完整 checkpoint 当作后者续训输入。

## 3. 远端 RLT 采集与部署

这是网站教程的 Stage 3，复用 `policy_server_deployment` / `robot_client_deployment`。先准备[部署配置](../deployment/configuration.md)和[远端连接](../deployment/connection.md)，按实际机器修改 JSON 中的相机、任务、GPU、服务器地址等字段。

Stage 1 模型必须是 full / merged、`backend=rlinf`、`use_rlt=true`，保存 tokenizer / processors 和与机器人一致的 `action_feature_names`。其 chunk 至少覆盖 `execution_horizon + prefix_steps`；下例要求至少 13 行。

在**推理服务器**运行；首次不指定 actor，采集冻结 VLA 的参考轨迹，不执行随机初始化的 actor：

```bash
uv run --no-sync python -m lerobot.async_inference.server.policy_server_deployment \
  --config_path=src/lerobot/async_inference/config/policy_server_deployment.json \
  --rtc.mode=guided \
  --rlt.enabled=true --rlt.execution_horizon=10 --rlt.prefix_steps=3 \
  --rlt.replay_dir=outputs/piper_rlt_rtc
```

在**机器人电脑**运行；`pretrained_name_or_path` 是服务器读取的模型路径：

```bash
uv run --no-sync python -m lerobot.async_inference.client.robot_client_deployment \
  --config_path=src/lerobot/async_inference/config/robot_client_deployment.json \
  --pretrained_name_or_path=/server/checkpoints/pi05_rlt_stage1 \
  --rtc_mode=guided \
  --rlt.enabled=true --rlt.execution_horizon=10 --rlt.prefix_steps=3 \
  --rlt.reward_provider=manual
```

保留配套的 Piper、相机、task、fps 和插值设置。两端 policy fps、RLT window / prefix 必须一致；`rlt.prefix_steps` 与普通 `rtc.execution_horizon` 是不同参数。RLT 使用 Stage 1 保存的数值 profile，普通 `pi05_inference.profile` 不覆盖冻结特征；服务器的 compile 开关和 mode 仍控制编译。

| 客户端命令 | 含义 |
| --- | --- |
| `/start` | 开始一轮 |
| `/success` | reward 1，任务终止 |
| `/failure` | reward 0，任务终止 |
| `/truncate` | reward 0，时间限制截断，仍保留 bootstrap 语义 |
| `/reset` | 丢弃未完成窗口并复位，不伪造成功 / 失败标签 |

手工结果只在 `/start` 成功且当前轮仍在运行时接收；开始新一轮会清除旧反馈。`/reset` 后需再次 `/start`，再提交该轮结果。

其余执行区间 reward 为 0，反馈在 policy interval 消费。也可配置 `--rlt.reward_provider=module:function`；回调接收 `(raw_post_step_observation, actual_absolute_command)` 并返回 `RewardSignal`。这里没有内置 DAgger 接管。

已完成的执行窗口（包括真实终止 / 截断的短窗口）会原子写入服务器上的 `outputs/piper_rlt_rtc/<session_uuid>/g..._s....pt`。按[离线学习](#2-从已有-replay-训练或续训)训练后，下一次服务端启动追加 `--rlt.actor_checkpoint=/server/checkpoints/rlt_rtc/actor`，并用 `--rlt.actor_version` 标识 learner 版本。不会在部署进程中自动训练或广播新权重。

若要把匹配的同步 Stage 2 actor 用作 RTC 初始策略，需显式添加服务端参数 `--rlt.warmstart_stage2_actor=true`。这只迁移策略权重，RTC critics 和优化器仍从新状态开始；兼容性边界见[排错](../../troubleshooting/rlt.md#stage-2-与远端-rtc)。

启动会先做不发运动命令的预热和时延测量。30 Hz、3 步 prefix 和默认 5 ms 余量对应 95 ms 预算。首个冷窗口还要求在客户端 `observation_max_age_s` 内开始执行，计时从初始观测中最老的传感器时间开始，包含推理和等待电机锁的时间；通过预热并不替代该检查。超时处理及参数边界见[排错](../../troubleshooting/rlt.md#stage-2-与远端-rtc)。

## 4. 同步采集接口与迁移

同步 Stage 2 采集是 `FrozenRLTFeatures`、`FixedWindowRollout`、`RobotWindowStep` 的 Python 接口，完整示例见[网站教程](../../source/pi05.mdx#rlt-stage-2-actor-critic-and-fixed-execution-window-replay)。示例中的 `io`、`prepare`、`task_reward` 必须由应用提供，不是开箱即用的机器人采集命令；连接、复位和独占硬件由调用者管理。

代码迁移保留完整仓库与根环境文件；匹配基线上的增量文件清单见[服务器同步说明](../../source/pi05.mdx#synchronizing-these-changes-to-a-training-or-inference-server)。必须一起携带 RLinf 模块的 `LICENSE` / `README.md`，不能只复制模型文件或用少数改动文件替换整个 `policies/` 目录。
