# RECAP 完整任务采集指令

[返回文档首页](../../README.md) · [环境准备](../setup/README.md) · [数据与实现约定](../../architecture/recap.md) · [异常与恢复](../../troubleshooting/recap.md)

首次准备依赖：服务端执行 `uv sync --locked --inexact --extra pi05-server`；机器人电脑执行 `uv sync --locked --inexact --extra piper-client`；同机运行时同时选择这两个 extra。下方 `uv run --no-sync` 只运行已准备好的 `.venv`。

本实现补齐后续 RECAP 离线训练需要的数据基础：自主执行、人工接管、任务结果和模型版本。当前尚未实现 RECAP 的 value/advantage 标注、策略训练及其与 RLinf 训练器的集成。仓库中的 [RLinf / RLT 流程](../rlt/README.md) 是另一条独立实现，不能直接读取这里的 RECAP 数据作为 replay。技术路线见 [RECAP / RLinf 调研](../../research/recap_rlinf_integration_2026-10-13.md)。

## 与 DAgger 分开运行

| 项目 | DAgger | RECAP |
| --- | --- | --- |
| 客户端 | `robot_client_dagger` | `robot_client_recap` |
| 服务端 | `policy_server_dagger` | `policy_server_recap` |
| 数据 | 只保存人工纠正窗口 | 同一任务的自主、暂停和人工接管段 |
| episode 边界 | 一次保存的 correction | 一次任务尝试，按 S/F 确认结果后保存 |
| 结束人工接管 | 保存纠正窗口并复位 | 暂停，等待标注结果；也可再次接管 |
| 任务结果 | 不要求 | `success / failure` |
| 复位/退出 | 沿用原来的丢弃/部分保存规则 | 丢弃尚未确认保存的本轮；已提交的保存继续完成 |

两套入口使用不同的数据目录。RECAP 的接管参数和输入设备参数沿用 `dagger_*` 名称，数据参数使用 `recap_*`；不要将它们替换成 DAgger 的数据参数。

## 启动

服务端与客户端使用同一份当前代码，所有命令从对应机器的仓库根目录执行。以下是 guided RTC / Pi05 示例，模型路径位于**服务端**；将地址、路径、任务及相机参数换成已有部署的实际值。机器人和相机配置必须与训练时一致；示例沿用本项目 Piper 默认硬件配置：`can_follower` 与 `/dev/video_top`、`/dev/video_wrist`、`/dev/video_down`。需要不同相机时显式传 `--robot.cameras`。这里示例直连 `10.0.0.2:8080`；使用 [SSH 隧道](../deployment/connection.md)时，将服务端 host 和客户端 server_address 改成 `127.0.0.1` / `127.0.0.1:8080`。

服务端：

```bash
uv run --no-sync python -m lerobot.async_inference.server.policy_server_recap \
  --host=0.0.0.0 --port=8080 --fps=30 \
  --rtc.mode=guided --rtc.execution_horizon=20 \
  --recap_policy_version=pi05-iteration-03
```

客户端：

```bash
uv run --no-sync python -m lerobot.async_inference.client.robot_client_recap \
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

依赖见[环境准备](../setup/README.md)，使用项目锁定的 uv Python 和对应 extra。`recap_num_episodes=0` 表示不限数量；目标按**本进程成功写入的数据 episode** 计数，包括任务成功和任务失败，不按接管次数计数。丢弃或零帧的尝试不占 episode，也不保留结果文件。

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

到达采集数量目标时保存后直接退出，使用原硬件退出流程；否则复位后等待操作者恢复环境，再开始下一轮。当前没有在被人工改变的环境中自动恢复策略的 `/resume` 功能。

用 `--dagger_input_device=prompt` 可改为命令输入，额外支持 `/status` 和 `/task <instruction>`。任务开始后禁止改任务描述。此版本暂不接受 pedal，因为现有两键踏板没有独立的任务结果输入。

**键盘模式**下，Esc 在后台编码期间仍可先关闭硬件，数据收尾随后完成。Prompt 模式的命令串行执行，保存期间不能再从同一输入行读取 `/stop`；主线程 Ctrl+C 和后台控制故障仍可触发退出。保存/复位期间重复按键和旧 generation 的排队事件会被拒绝。

## 指示灯

RECAP 复用远端 DAgger 的灯态；S/F 结果需要通过输入确认，灯光不表示任务成功率或数据校验结果。各阶段颜色与参数见[指示灯说明](../tools/status-indicator.md)。

## 采集后检查

数据为 LeRobot v3，加上 `meta/recap/<rollout-uuid>.json` 的任务结果与模型身份。采样、动作有效性、真实时间和模型指纹的定义统一见[RECAP 数据与实现约定](../../architecture/recap.md)。

普通训练器不会自动过滤 `recap.action_valid=false` 的帧，当前 merge/edit 工具也没有维护 sidecar 的 episode 映射。不要直接将其当作普通 DAgger 数据合并或训练；当前 RECAP 训练与导出适配尚未实现。保存失败或续录被拒绝时先看[异常与恢复](../../troubleshooting/recap.md)。
