# 单机本地 DAgger 纠正采集

[DAgger 目录](README.md) · [环境准备](../setup/README.md) · [本地推理架构](../../architecture/local-inference.md)

所有命令从仓库根目录运行。下方 extra 与本地部署一致；按训练相机配置添加 `--robot.cameras`，默认硬件使用 `can_follower` 与三相机 `/dev/video_top`、`/dev/video_wrist`、`/dev/video_down`。本页没有远端服务和 SSH 步骤。

DAgger 使用同一个进程内推理核心：模型始终由一个推理线程持有，接管和复位会撤销旧动作代际，
接管期间返回的推理结果不会下发。硬件接管、采集和状态机均在本包内实现，不调用远端部署代码。
自主阶段主从两臂接收相同的平滑后控制命令；人工阶段关闭计算机动作流，由物理主臂的 CAN
控制帧直接驱动从臂，不经过 GPU 或 Python 动作插值。SocketCAN 的本机回环帧不会进入接管桥。

先执行 `uv sync --locked --inexact --extra pi05-server --extra piper-client`，为本地推理和 DAgger 数据写入补齐依赖。

```bash
uv run --no-sync python -m lerobot.async_inference_local.scripts.dagger \
  --robot.type=piper_follower \
  --pretrained_name_or_path=/absolute/path/to/pretrained_model \
  --task="pick up the object" \
  --inference.device=cuda:0 \
  --inference.precision_profile=openpi \
  --inference.compile_model=true \
  --rtc.mode=guided \
  --rtc.execution_horizon=20 \
  --actions_per_chunk=50 \
  --fps=30 \
  --hardware.master_can_name=can_master \
  --hardware.follower_can_name=can_follower \
  --dataset.repo_id=your_name/local_dagger_corrections \
  --dataset.root=/absolute/path/to/new_corrections_dataset \
  --dataset.use_videos=true \
  --num_episodes=50 \
  --input_device=keyboard
```

checkpoint、相机名称/分辨率和 `fps` 的要求与[本地部署](../deployment/local.md)相同。
`hardware.follower_can_name` 必须为现有 Piper 驱动固定使用的 `can_follower`；
主臂另接 `can_master`。默认要求主从固件版本相同且至少为 `1.8.9`，并确认主臂已使能。
连接时沿用驱动校准；主臂在自主阶段也会随从臂运动。

默认键盘流程如下，每次右键只执行一步：

| 当前状态 | 右键 | 左键 |
| --- | --- | --- |
| READY | 开始新的自主 rollout | 复位 |
| 自主运行 | 暂停，撤销旧推理及动作并对齐双臂 | 拒绝；先按右键暂停 |
| 暂停 | 校验主臂物理控制量与从臂姿态，再开始人工纠正 | 丢弃未保存片段并复位 |
| 人工纠正 | 停采、关闭接管桥、保存当前片段，再复位 | 丢弃当前片段并复位 |
| 采集/复位故障 | 拒绝继续采集，等待复位 | 尝试恢复并复位 |

复位完成后回到 READY，布置好下一次任务场景，再按右键开始。暂停后不能直接恢复旧 rollout。
主臂接管对齐未通过时保持暂停，可重新对齐后重试；记录或复位失败后必须先成功复位。
物理 CAN 接管桥故障时停止转发、停采并丢弃当前片段，主臂保持人工 leader 模式；
不会自动切回从臂模式并拉回从臂位置。按左键或 `/reset` 后才执行恢复和复位。
复位时先保留当前夹爪开度返回机械臂姿态，确认主从两臂的**新鲜物理反馈**稳定到位，
再打开夹爪；不会用 SDK 的旧缓存作为到位依据。

保存和复位期间重复按键会被丢弃，不会在完成后补执行。Esc 独立于普通操作队列请求退出；
保存视频时也能关闭接管桥、停止硬件，随后等待文件保存完成。Esc 是正常软件退出，
沿用受控回位流程；终止性推理/控制故障会对两臂保持电子停止，不启动新回位动作。
模型编译/启动阶段可以用 Ctrl+C 退出。

输入方式：

- `--input_device=prompt`：支持 `/start [task]`、`/pause`、`/correct`（开始/结束纠正）、
  `/reset`、`/task TEXT`、`/status`、`/stop`。换任务仅在 READY 允许；保存/复位期间
  `/status` 和 `/stop` 仍可使用。输入 EOF 也会退出。
- `--input_device=pedal`：使用通用踏板监听器，默认路径
  `/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd`，`KEY_A` 前进、`KEY_B` 丢弃并复位；
  用 `--pedal_device_path`、`--pedal_advance`、`--pedal_reset` 覆盖。需要 evdev 和设备读取权限。
  同时启用键盘 Esc 退出；没有可用键盘时用 Ctrl+C。
- 键盘映射由 `--advance_key`、`--reset_key`、`--stop_key` 配置。
  `--status_indicator.backend=none` 可关闭可选状态灯；本地阶段对应的颜色与闪烁见[指示灯说明](../tools/status-indicator.md)。

数据语义与远端 DAgger 一致：仅人工纠正区间写入数据，一次纠正对应一个 episode；
观测取从臂，`action` 为主臂原始绝对控制命令，`intervention=[true]`。
自主动作、暂停、对齐、复位过程不写入。图像保留原始硬件特征名和像素；策略侧的
`rename_map`、resize、归一化和相对动作处理不会作用到采集标签上。
任务文本在纠正开始时固定，不能在窗口中途改写。

数据集参数：

- `dataset.repo_id` 不设置时不写数据，可验证接管流程；实际采集必须设置它。
  `dataset.root` 是新建数据集目录；追加已有数据需同时指定原 `repo_id`、`root` 和
  `--dataset.resume=true`。续录检查帧率、特征、形状、名称顺序和图片/视频类型。
- `dataset.fps=0` 默认继承策略 `fps`，也可显式指定正整数；采集 fps 要符合后续训练时基。
  `num_episodes` 只计算**本次进程成功保存的纠正片段数**，0 表示不限。
  达到目标后正常退出，不再额外执行一次双臂复位。
- 默认退出时丢弃尚未结束的纠正片段；`--save_partial_on_stop=true` 可在正常退出时保存。
  采集或保存失败的片段不会在退出时重试提交。
- 默认 `dataset.push_to_hub=false`，数据保存在本地；需要上传时显式开启该参数。
- `dataset.image_writer_processes` 必须为 0，因为本进程持有 CUDA 上下文和 CAN 读取线程。
  图像写入使用线程（`dataset.image_writer_threads_per_camera=4`），视频按相机依次编码，
  避免对当前 GPU/硬件进程执行 `fork`。编码期间不持有电机或采集锁。

`autostart=true` 会在模型预热和采集器就绪后自动开始第一次 rollout。
DAgger 中可选的 `duration_s` 限制从就绪开始的整个采集会话时长；默认不限时。
`/status` 包含当前阶段、保存状态、当前片段帧数、本次已保存数量及故障信息。
GPU 实际速度、USB/CAN 时序与任务成功率仍需在目标 4090 和双 Piper 上验证。

切换 trained RTC 时，使用实际训练过前缀条件的 checkpoint，将 `--rtc.mode=guided` 换成 `--rtc.mode=trained`，并按模型上限添加 `--trained_delay_steps=25`。本页动作块为 50，要求训练上限至少 25；完整条件见[本地部署](../deployment/local.md#切换-trained-rtc-和检查输入)。
