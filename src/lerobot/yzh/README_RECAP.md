# RECAP 第一阶段：完整任务采集、结果标注与模型追溯

> 日常操作统一从 [docs/README.md](../../../docs/README.md) 进入。本文保留 RECAP 整轮任务边界、结果标注、数据 schema、模型身份和保存恢复约定；当前仅实现采集基础，尚未实现 RECAP 训练。

当前入口分别位于 `async_inference/server/` 和 `async_inference/client/`，双端文件完整保留在 `shared/`。目录调整后的启动路径与迁移说明见 [异步推理目录说明](../async_inference/README.md)。

本实现补齐后续 RECAP 离线训练需要的数据基础：自主执行、人工接管、任务结果和模型版本。当前尚未实现 value/advantage 标注、RECAP 策略训练或 RLinf 接入。技术路线见 [RECAP / RLinf 调研](../../../docs/research/recap_rlinf_integration_2026-10-13.md)。

## 与 DAgger 分开运行

| 项目 | 原 DAgger 入口 | 新 RECAP 入口 |
| --- | --- | --- |
| 客户端 | `robot_client_dagger` | `robot_client_recap` |
| 服务端 | `policy_server_dagger` | `policy_server_recap` |
| 数据 | 只保存人工纠正窗口 | 同一任务的自主、暂停和人工接管段 |
| episode 边界 | 一次保存的 correction | 一次任务尝试，按 S/F 确认结果后保存 |
| 结束人工接管 | 保存纠正窗口并复位 | 暂停，等待标注结果；也可再次接管 |
| 任务结果 | 不要求 | `success / failure` |
| 复位/退出 | 沿用原来的丢弃/部分保存规则 | 丢弃尚未确认保存的本轮；已提交的保存继续完成 |

新代码单向引用原 DAgger 的双臂控制、接管校验、RTC generation 隔离和复位逻辑；原有 DAgger 文件和运行命令不需要修改。两套入口使用不同的数据目录。RECAP 的接管参数和输入设备参数仍沿用 `dagger_*` 名称，数据参数使用 `recap_*`。

## 启动

服务端与客户端使用同一份包含新文件的代码。以下是 guided RTC / Pi05 示例，模型路径位于**服务端**；将地址、路径、任务及相机参数换成已有部署的实际值。机器人和相机配置必须与训练时一致；示例沿用本项目 Piper 默认硬件配置。

服务端：

```bash
uv run --locked --extra pi05-server python -m lerobot.async_inference.server.policy_server_recap \
  --host=0.0.0.0 --port=8080 --fps=30 \
  --rtc.mode=guided --rtc.execution_horizon=20 \
  --recap_policy_version=pi05-iteration-03
```

客户端：

```bash
uv run --locked --extra piper-client python -m lerobot.async_inference.client.robot_client_recap \
  --robot.type=piper_follower \
  --server_address=10.0.0.2:8080 \
  --policy_type=pi05 \
  --pretrained_name_or_path=/data/checkpoints/iteration-03/pretrained_model \
  --policy_device=cuda --client_device=cpu \
  --rtc_mode=guided --fps=30 --actions_per_chunk=50 \
  --interpolation_multiplier=2 \
  --task="pick the cube and place it in the bowl" \
  --dagger_master_can_name=can_master \
  --dagger_follower_can_name=can_follower \
  --dagger_input_device=keyboard \
  --recap_dataset_repo_id=your_name/piper_recap_iteration03 \
  --recap_dataset_root=/data/piper_recap_iteration03 \
  --recap_num_episodes=20
```

环境按主入口使用项目锁定的 uv Python 和对应 extra。`recap_num_episodes=0` 表示不限数量；目标按**本进程成功写入的数据 episode** 计数，包括任务成功和任务失败，不按接管次数计数。丢弃或零帧的尝试不占 episode，也不保留结果文件。

沿用已验证的 trained RTC 部署时，同时保留服务端的 `--rtc.mode=trained --trained.delay_steps=...` 和客户端的 `--rtc_mode=trained --trained.queue_threshold=...`，加载对应 trained checkpoint。这里没有修改 guided/trained 算法。

追加采集使用同一 `recap_dataset_repo_id/root` 并加 `--recap_dataset_resume=true`；相机、动作特征、记录 fps 和 RECAP schema 必须匹配。更换 checkpoint 后可追加到同一数据集，每个 episode 自带自己的模型身份。

当前采集元数据 schema 为 2，取消了旧版的额外结束观测和未知/中止结果。已有旧 schema 数据不会自动改写或混合追加，请使用新的采集目录。模型身份自身的 schema 不受此改动影响。

## 操作流程

默认右键依次执行：

```text
READY → 自主执行 → 暂停 → 人工接管 → 暂停
                                      ↑      │
                                      └──────┘  可再次接管
```

| 按键 | Prompt 等价命令 | 行为 |
| --- | --- | --- |
| 右方向键 | `/start [task]`、`/pause`、`/correct` | 按当前阶段开始、暂停或切换接管 |
| S | `/success` | 任务成功，保存整轮并复位 |
| F | `/failure` | 任务失败，保存整轮并复位 |
| 左方向键 | `/reset` | 暂停/接管/故障时丢弃尚未确认保存的本轮，再复位 |
| Esc | `/stop` | 退出，丢弃尚未确认保存的本轮；已提交的保存继续完成 |

纯自主成功/失败也要按 S/F，无需先接管。人工接管后完成任务应标 `success`；它表示**整轮任务**成功，自主完成与人工辅助完成可通过 `intervention` 区分。暂停和接管本身不会结束任务，也不会自动产生失败标签。已判定失败的一轮可以按 F 保存，即使它比成功的一轮更短；仅因设备故障、取消采集或想重来而中断时，使用复位/退出丢弃。

按现有操作习惯，可以手动将机械臂回 home 后再按 S/F。停止时不再读取额外观测、不追加无动作末帧；最后一帧就是停止前的正常采样。录制期间手动回 home 的动作仍会被记录。若希望策略也完成回 home，可将它作为任务结束条件；若回 home 仅用于人工复位，后续训练应在实际任务结束处裁剪，否则回报会把回 home 和等待按键的时间也算入任务耗时。当前不会自动猜测或裁剪这个边界。

这些是本项目的采集约定。RECAP 论文 §IV-A 使用到 episode 结束的经验回报，§V-C 根据成功/失败标签构造末步奖励；论文没有定义 `unknown/aborted`，也没有明确要求保留无任务结局的中断片段。因此当前实现不把这类片段保存进训练数据。[RECAP 论文](https://arxiv.org/html/2511.14759v2)

到达采集数量目标时保存后直接退出，使用原硬件退出流程；否则复位后等待操作者恢复环境，再开始下一轮。当前没有在被人工改变的环境中自动恢复策略的 `/resume` 功能。

用 `--dagger_input_device=prompt` 可改为命令输入，额外支持 `/status` 和 `/task <instruction>`。任务开始后禁止改任务描述。此版本暂不接受 pedal，因为现有两键踏板没有独立的任务结果输入。

**键盘模式**下，Esc 在后台编码期间仍可先关闭硬件，数据收尾随后完成。Prompt 模式的命令串行执行，保存期间不能再从同一输入行读取 `/stop`；主线程 Ctrl+C 和后台控制故障仍可触发退出。保存/复位期间重复按键和旧 generation 的排队事件会被拒绝。

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

## 异常与恢复

退出会先停止 CAN 和机械臂，再清理未确认的缓冲区及临时图像；采集故障会停止策略，并在退出时丢弃本轮。已经按 S/F 提交结果的一轮继续完成保存，Esc 不会把它改为丢弃。明确标注的任务失败仍是有效数据。

sidecar 经过 `recording → saving → committed`，原子替换 JSON。这三个字段值是内部写盘阶段，不是任务结果或训练标签。正常丢弃会删除该轮临时 sidecar。保存失败不自动重试、不覆盖下一 episode，保留日志供检查，避免误删已写入的数据。追加时先校验 sidecar 状态、episode 对应关系和 schema；有未完成写盘记录就拒绝追加。不要只删元数据或手工把状态改成 committed，应先核对数据/视频/episode 元数据并修复或隔离损坏的数据集。

这不是断电恢复系统。尚未保存的数值帧仍在内存，`kill -9`、掉电或底层硬件阻塞不能保证保全；日志用于发现不完整记录。正常 `committed` 数据也应在程序完成 finalize 后再用于训练。`recap_dataset_push_to_hub` 默认 false，开启时按原 dataset 上传逻辑携带 `meta/recap`。

原有 merge/edit 工具尚未适配这些 sidecar。不要直接用它们合并、删除或重排 RECAP episode 后继续训练；下一阶段导出器需同步重写 episode 映射、过滤无效动作，并按确认的任务边界和成功/失败结果构造回报。

## 实现与验证

- [robot_client_recap.py](../async_inference/client/robot_client_recap.py)：RECAP 状态边界、操作入口、模型握手、退出顺序。
- [piper_hil_recap.py](../async_inference/client/piper_hil_recap.py)：完整 rollout 记录、schema、结果和异常日志。
- [policy_server_recap.py](../async_inference/server/policy_server_recap.py)：实际加载目录和模型指纹。
- [metadata_recap.py](../async_inference/shared/metadata_recap.py)：共享身份校验和 JSON 写入。

针对新代码的测试命令：

```bash
uv run pytest -q \
  tests/async_inference/test_piper_hil_recap.py \
  tests/async_inference/test_policy_server_recap.py \
  tests/async_inference/test_robot_client_recap.py
```

测试覆盖真实无相机 LeRobot 数据读写、跨阶段 episode、结果/模型身份、恢复检查、保存故障、旧 RTC 请求隔离以及慢保存时的硬件退出顺序。真机 CAN、相机吞吐和 GPU 模型推理仍需现场试采验证。
