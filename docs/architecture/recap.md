# RECAP 数据约定与实现

[架构目录](README.md) · [采集指令](../workflows/recap/README.md) · [异常与恢复](../troubleshooting/recap.md) · [调研资料](../research/recap_rlinf_integration_2026-10-13.md)

RECAP 当前仅实现完整任务采集、成功/失败标注和模型追溯；其 value/advantage 标注、策略训练及与 RLinf 训练器的集成尚未实现。[RLinf / RLT](rlt.md) 属于独立模型与训练路线，不改变这里的 RECAP 数据边界。日常启动与按键流程统一放在[采集指令](../workflows/recap/README.md)。

RECAP 单向复用 DAgger 的双臂控制、接管校验、RTC generation 隔离与复位逻辑；DAgger 不依赖 RECAP。RECAP 的 `dagger_*` 接管参数和 `recap_*` 数据参数分别配置这两部分。

这些是本项目的采集约定。RECAP 论文 §IV-A 使用到 episode 结束的经验回报，§V-C 根据成功/失败标签构造末步奖励；论文没有定义 `unknown/aborted`，也没有明确要求保留无任务结局的中断片段。因此当前实现不把这类片段保存进训练数据。[RECAP 论文](https://arxiv.org/html/2511.14759v2)

## 数据约定

记录器从任务启动后开始按 `recap_dataset_fps` 采样，默认 `0` 继承 policy fps。它覆盖任务各阶段，**不等于逐条记录 60 Hz 插值命令或原始 CAN 帧**。采样线程与推理观察线程共享观察锁，避免同时调用相机/机器人读取。

| 帧字段 | 含义 |
| --- | --- |
| `observation.*` | 从臂关节、相机等原 LeRobot 观测 |
| `action` | 自主段：最近实际下发的插值/共享 EMA 命令；接管段：主臂控制目标 |
| `intervention` | 当前采样属于人工接管 |
| `recap.phase` | `0=autonomous, 1=intervention, 2=paused` |
| `recap.action_valid` | 有可用动作标签；暂停、首个命令到来前为 false |
| `recap.segment_id` | 每次阶段变化或恢复采样时递增，用于识别切换边界 |
| `recap.generation` | RTC generation，避免跨接管/复位关联旧请求 |
| `recap.policy_step` / `recap.source_observation_id` | 自主命令的 policy step 与来源请求；无对应命令时为 -1 |
| `recap.elapsed_s` / `recap.wall_time_s` | 实际单调时钟相对时间 / Unix 时间 |
| `recap.observation_duration_s` | 本次观察读取和样本获取耗时 |
| `recap.command_age_s` | 自主命令距离采样完成的时间；接管目标无精确发送时间，填 -1 |

动作单位继承当前 Piper 实现：关节 rad、夹爪 m。无效动作帧以测得的关节位置填充 `action`，后续 RECAP 训练适配器必须通过 `recap.action_valid` 排除动作监督；现有普通训练器不会自动应用该字段。

这里记录的是**观察读取完成时最近的执行目标**，不是严格同步的“执行前状态 → 新动作”对，也不是原始策略 chunk。推理等待时最近目标可能连续重复，使用 command age 判断等待长度。物理接管转换和程序复位不采样；转换前后由 segment 和实际时间暴露间隙。停止录制前的人工回 home 仍在采样范围内，与程序复位不同。

LeRobot 自带的 `timestamp=frame_index/fps` 仍是名义视频时间；计算任务耗时、时间间隔或后续 RL 折扣时应使用 `recap.elapsed_s`。慢相机造成的掉速不会通过补采突发帧掩盖。全程采样会额外占用相机读取、编码和磁盘资源，首次真机试采需要检查实际采样时间和控制频率。

## 结果与模型版本

数据保持标准 LeRobot v3 格式，另有：

```text
<dataset-root>/meta/recap/<rollout-uuid>.json
```

每个已保存的 sidecar 包含 `episode_index`、任务、起止时间、帧数、结束原因、`outcome/success`、模型身份和客户端/服务端配置。`outcome` 只有 `success/failure`，`success` 对应布尔值，不存在第三种任务结果。存在已知采集/控制故障的本轮不能通过 S/F 保存。

模型身份用于追溯和比较不同 checkpoint 的采集效果，是本项目的工程元数据；它不是 RECAP 论文规定的训练输入，也不参与当前的成功/失败判定。

模型身份由 RECAP 服务端在加载前生成，通过 gRPC trailing metadata 返回：

- `model_version=sha256:...`：checkpoint 目录中 `.safetensors/.json/.model/.txt/.tiktoken` 文件相对路径和内容的合成指纹，包含权重及 processor/config；相同路径替换文件会改变指纹。
- `label`：可选的 `recap_policy_version`，供人阅读，不能代替指纹。
- `requested_path/resolved_path/hub_commit`：请求位置、服务端实际加载目录、Hub snapshot commit。
- `server_load_id/loaded_at_utc`：区分每次实际加载。每次连接重新加载，不接受 `reuse_last_policy=true`。

Hub 模型先解析成 snapshot，再从同一目录加载；可用服务端 `--recap_model_revision=<commit>` 固定版本。首次连接计算哈希需要读取权重，启动会变慢。加载前后检查文件 stat，发现期间发生变化会拒绝启动；使用完成写入的独立 checkpoint 目录，不要在训练中原地覆盖。指纹不是代码版本或外部基础模型依赖的完整环境快照。

RECAP 客户端要求服务端返回身份；连接旧 DAgger 服务端会报错，不会静默用路径冒充模型版本。一个服务端实例按现有 RTC 使用方式服务一套机器人客户端。

## 实现与验证

- [robot_client_recap.py](../../src/lerobot/async_inference/client/robot_client_recap.py)：RECAP 状态边界、操作入口、模型握手、退出顺序。
- [piper_hil_recap.py](../../src/lerobot/async_inference/client/piper_hil_recap.py)：完整 rollout 记录、schema、结果和异常日志。
- [policy_server_recap.py](../../src/lerobot/async_inference/server/policy_server_recap.py)：实际加载目录和模型指纹。
- [metadata_recap.py](../../src/lerobot/async_inference/shared/metadata_recap.py)：共享身份校验和 JSON 写入。

针对新代码的测试命令：

```bash
uv run --locked --extra test --extra piper-client --extra pi05-server pytest -q \
  tests/async_inference/test_piper_hil_recap.py \
  tests/async_inference/test_policy_server_recap.py \
  tests/async_inference/test_robot_client_recap.py
```

测试覆盖真实无相机 LeRobot 数据读写、跨阶段 episode、结果/模型身份、恢复检查、保存故障、旧 RTC 请求隔离以及慢保存时的硬件退出顺序。真机 CAN、相机吞吐和 GPU 模型推理仍需现场试采验证。
