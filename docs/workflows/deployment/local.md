# 单机本地真机部署

[部署目录](README.md) · [本地 DAgger 采集](../dagger/local.md) · [本地推理架构](../../architecture/local-inference.md)

适用于 GPU、机械臂和相机在同一机器的 PI05 部署。此入口不需要启动远端服务或 SSH 隧道；当前没有本地 RECAP 入口。所有命令从仓库根目录运行。

## 启动

在目标设备使用主仓库的完整当前代码、`pyproject.toml`、`uv.lock` 和 checkpoint。
如果当前实现包含未提交文件，仅复制 Git 提交不能完整迁移。
本地部署依赖安装见[环境准备](../setup/README.md)；先执行 `uv sync --locked --inexact --extra pi05-server --extra piper-client`，同时准备模型、数据集和机器人依赖；下方入口运行时不再同步。

```bash
uv run --no-sync python -m lerobot.async_inference_local \
  --robot.type=piper_follower \
  --pretrained_name_or_path=/absolute/path/to/pretrained_model \
  --task="pick up the object" \
  --inference.device=cuda:0 \
  --inference.precision_profile=openpi \
  --inference.compile_model=true \
  --inference.compile_mode=max-autotune \
  --inference.image_mode=openpi_uint8 \
  --rtc.mode=guided \
  --rtc.execution_horizon=20 \
  --actions_per_chunk=50 \
  --fps=30 \
  --interpolation_multiplier=2
```

替换 checkpoint 路径和任务描述，并按 checkpoint 的相机名称、图像设置及动作时基配置。
`empty_cameras` 声明的占位图像交由 PI05 补零，其余图像特征必须匹配真实相机。
相机参数仍通过 `--robot.cameras` 配置；`--rename_map` 可以映射观测图像特征名。
现有 Piper 驱动使用 `can_follower`，默认相机设备是 `/dev/video_top`、
`/dev/video_wrist`、`/dev/video_down`，默认旋转分别为 90°、0°、0°。这与远端双相机配方不同；不要直接沿用错误的相机集合或旋转，应通过 `--robot.cameras` 显式匹配训练数据。

启动时加载模型并在实际推理线程预热 ordinary/RTC 两种调用形态，预热结果不会进入动作计划。
首次编译可能较慢，`--startup_timeout_s` 默认 1800 秒；编译开启时要求至少两次预热。
仅保留每种形态首轮之后的预热时延用于初始延迟估计。
默认等待 `/start`；可使用 `--autostart=true --duration_s=60` 自动运行指定时长。

连接及正常退出沿用现有 Piper 驱动：连接可能执行校准，正常退出会回安全位并失能。
`autostart=false` 只阻止策略自主动作，不改变驱动连接行为。
故障退出不会调用会触发回位的 Piper `disconnect()`，而是发送 SDK 电子停止并关闭相机，
不自动解除电子停止；之后需要操作者处理机械臂状态。
硬件线程退出后即执行硬件清理，不等待已取消的 GPU 调用；并发退出只执行一次断开操作。

离线运行需提前准备权重、处理器 JSON/统计文件和 tokenizer。
保存的 tokenizer 路径不可用时，可通过 `--inference.tokenizer_path=/absolute/path/to/tokenizer` 覆盖。
当前入口要求普通或已合并的 PI05 权重，不直接加载 PEFT adapter。

## 交互与生命周期

| 命令 | 行为 |
| --- | --- |
| `/start`、`/resume` | 从当前实测姿态和全新观测开始自主运行 |
| `/pause` | 撤销队列和未完成的推理结果，停止下发新的策略动作 |
| `/reset` | 撤销旧结果，持握回位、实测确认后再打开夹爪，验证完成后进入 ready |
| `/task 新任务` | 更换任务并撤销旧任务的计划，等待全新观测 |
| `/status` | 查看状态、剩余动作、故障及 P50/P95/P99 时延 |
| `/quit`、Ctrl-C | 撤销动作、停止线程并执行退出清理 |

暂停/重置会使旧的推理结果失效；线程与动作代际机制见[本地架构](../../architecture/local-inference.md)。
复位在独立的临时线程中执行，期间仍可用 `/status` 查看进度、`/quit` 取消并退出；
`/start`、`/pause`、`/reset` 和 `/task` 在复位结束前会被拒绝，不会排队后自动执行。
`/reset` 与远端 Deployment 使用相同的持握回位顺序：先以实测开度保持夹爪，
关节稳定到位后再张开夹爪。Piper 默认张开到 0.07 m；可用
`--reset_gripper_open_position` 调整，或用 `--reset_open_gripper=false` 恢复目标姿态中的夹爪位置。
关节容差默认 0.02 rad，夹爪张开后的容差默认 0.001 m；每阶段需稳定
`--reset_settle_s=0.1` 秒，超时由 `--reset_settle_timeout_s=5` 控制。
持握阶段只以关节到位决定何时释放，但所有反馈仍须有限、完整。
`--reset_to_initial_position=false` 且没有显式 `reset_pose` 时，保持当前关节姿态并完成相同的
稳定确认和夹爪释放流程；显式 `reset_pose` 优先，保存的目标不会被夹爪释放参数改写。
故障状态不允许继续 `/start`，需要退出后创建新的运行实例。

## 切换 trained RTC 和检查输入

上述命令是 guided。使用实际训练过前缀条件的 checkpoint 时，将 `--rtc.mode=guided` 换成 `--rtc.mode=trained`，并显式添加 `--trained_delay_steps=25`；该本地参数与远端 `trained.delay_steps` 名称不同。训练上限 `D`、固定前缀 `B`、动作块 `C` 必须满足 `0 < B <= D`、`2*B <= C <= chunk_size`。本页 `C=50`，因此 `B=25` 要求 `D>=25`。没有显式设置本地 delay 时使用 checkpoint 训练上限。

`inference.image_mode=openpi_uint8` 要求图像归一化为 identity；非 identity checkpoint 应用 `--inference.image_mode=lerobot_float32`。相机图像名称、相对动作配置、策略 fps 应与训练一致。模式细节见[本地推理架构](../../architecture/local-inference.md)。

使用 `/status` 检查 P50/P95/P99 推理与控制时延。动作预测时域必须覆盖两轮推理消耗及余量；仅增大超时不会增加动作块覆盖时长。目标 GPU、摄像头和 CAN 的性能需在实际设备测量；[统计与验证范围](../../architecture/local-inference.md)。
