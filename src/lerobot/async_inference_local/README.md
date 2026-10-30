# 单机本地 Deployment / DAgger

独立的 PI05 本地部署运行时，面向机械臂、相机和推理 GPU 位于同一台机器的场景。
包含自主运行 Deployment 和只采集人工纠正片段的 DAgger 入口；尚不包含 RECAP 完整 rollout 录制。

本包不导入、调用或继承 `lerobot.async_inference` 的任何模块，不依赖 gRPC、
LeRobot transport、网络端口或 pickle。两种运行时可以分别使用；通用的
`policies`、`processor`、相机和机器人驱动仍由 LeRobot 提供。

## 目录

```text
async_inference_local/
├── __main__.py              # python -m lerobot.async_inference_local
├── scripts/                # deploy.py 自主运行；dagger.py 人工纠正采集
├── shared/                 # 配置、内存对象、有限长度的性能统计
├── inference/              # 模型加载、图像准备、独立的轻量 RTC guidance
├── control/                # 动作计划、Piper I/O、双臂控制与物理 CAN 接管
├── recording/              # 纠正片段写入 LeRobotDataset
└── runtime/                # deployment.py 本地核心；dagger.py 接管/采集状态机
```

## 执行路径

同一进程内运行三个工作线程：

1. 采集线程独立读取传感器，只发布最新的完整观测。每幅图像读出后立即取得一次独立所有权，
   防止后续相机采集覆盖正在推理的内容；不累积待处理观测队列。
2. 唯一推理线程负责模型加载、预热、处理器状态、RTC 前缀、模型调用和后处理。
   没有同时在运行的第二个模型调用。GPU 模型空间/执行空间前缀直接保留；
   完整动作块后处理后才复制到 CPU。CPU 状态分词和 GPU 前缀归一化使用独立统计对象。
3. 控制线程只访问 CPU 动作，按 `fps * interpolation_multiplier` 下发。
   当前插值终点保持不变，只替换尚未开始执行的动作；成功发送后才推进执行进度。

Piper 适配将相机读取放在电机锁之外，避免等相机时占住 CAN 控制锁。
底层驱动没有独立电机/相机接口时，程序化接入其他机器人使用保守的串行 I/O；
当前 CLI 注册的是 `piper_follower`。

推理线程由新观测事件唤醒，始终使用最新有效观测连续重规划。
采集频率、策略动作频率和插值后的电机命令频率分别配置。
不要把提高模型推理速度等同于可以修改训练数据对应的 `fps`。

## 安装与启动

在目标设备使用主仓库的完整当前代码、`pyproject.toml`、`uv.lock` 和 checkpoint。
如果当前实现包含未提交文件，仅复制 Git 提交不能完整迁移。
本地部署需要 PI05 和 Piper 的依赖，不需要选用远端 server/client extras。

```bash
uv sync --locked --extra pi --extra piper --extra intelrealsense

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
`/dev/video_wrist`、`/dev/video_down`。本包没有改动驱动中的设备名称或默认旋转。

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

暂停/重置通过 generation 隔离旧结果。已经发起的 GPU 调用不会强行中断，
其结果会被丢弃，完成之后才允许开始后续模型调用。
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

## RTC 和输入语义

- `guided` 使用本包独立的 identity-VJP 轻量实现，并把延迟、前缀长度转为编译器可用的张量输入。
  本地 delay 根据完整推理流水线的分位数、动作频率和余量估计；不会因为去掉网络而置零。
  保留完整的真实剩余前缀，`execution_horizon` 只控制 guidance 掩码。
  与远端默认策略一致，实际消耗（含插值中的终点）超出条件延迟 3 步的结果会被拒收。
- `trained` 要求 checkpoint 的 `rtc_training_max_delay > 0`。
  `--trained_delay_steps` 默认取训练上限，要求 `0 < B <= D` 且 `2*B <= actions_per_chunk <= chunk_size`。
  逐次验证硬前缀保持不变；推理耗时或实际已提交动作超出条件前缀时丢弃该结果。
- `openpi_uint8` 在 CPU 保留 OpenPI 的 PIL bilinear、向下取整尺寸和居中黑边语义，
  通过复用的 pinned buffer 将 uint8 送入 GPU 后转换为 float32；没有字节打包/解包。
- `lerobot_float32` 保留先按 checkpoint 特征尺寸缩放、再按模型尺寸 resize/pad 的路径。
  保存的处理器在两次缩放之间执行；第二次由 PI05 完成，保持图像归一化/变换的先后顺序。
  `openpi_uint8` 要求图像归一化为 identity；非 identity 的 checkpoint 应选择 `lerobot_float32`。
  两种模式是输入语义选择，不是传输协议；应与原部署选择保持一致。
- 相对动作以前缀的绝对坐标和新观测状态重新锚定。重置/暂停后不复用旧的状态、动作或相机历史。
  处理器使用 checkpoint 声明的动作名称；缺失时使用机器人实际动作顺序（或显式 `action_keys`），
  与远端保持一致，确保 `relative_exclude_joints` 对夹爪等维度正确生效。

运行时独立检查输入观测新鲜度（默认 0.1 秒）、执行时来源年龄（默认 2 秒）、
推理超时（默认 10 秒）和启动超时。相机支持时使用原始帧时间戳；不支持时只能保守地
使用采集起始时间，不能保证曝光年龄。一台旧相机不影响其他相机的时间戳检查。
暂停/复位/换任务后，带时间戳的相机必须提供边界之后的新帧；异常时间戳不会被改记为新帧。
电机锁等待结束、即将实际发送时再检查一次动作期限。动作队列耗尽后进入故障，不无限重复过期动作。
推理或控制异常先撤销动作、设置故障状态并通知退出，再在锁外记录错误，避免日志拖延动作撤销。
连续重规划还要求动作预测时域能覆盖两轮推理所消耗的步数及余量：仅能覆盖一轮推理的
chunk 并不足以保证持续运行。应根据目标设备的稳态时延检查这一容量，而不是仅增大超时。

## 性能验证

`/status` 输出的样本窗口有固定上限，不逐帧写日志、不在控制线程同步 CUDA。
统计包括整条推理耗时、观测到动作块可用、观测到该来源首次下发、动作来源年龄、
电机下发耗时（含锁等待）和控制周期迟到。
图像 resize 数值、原始模型精度和已有去噪步数不会为了展示吞吐而被静默替换。

默认复用 OpenPI 精度配置和已有 PI05 编译接口。本实现没有修改公共模型中的 KV cache 或
attention 算法，也没有开启量化或减少去噪步数。它们需要在目标 4090 上独立评测；
GPU 编译、显存峰值、真实摄像头/CAN 时序和机械臂任务成功率不由 CPU 模拟测试证明。

回归测试位于仓库的 `tests/async_inference_local`，包括数值对照、并发取消、
动作衔接、分阶段夹爪复位、图像所有权与时效、超时、配置、保存后重载的真实 PI05 处理器，
复位期间的命令响应、错误日志阻塞，以及在阻断远端模块时导入 CLI/PI05 的隔离检查。
模型权重计算和硬件 I/O 在这些测试中使用替身；DAgger 另外使用真实数据集写入器，
验证双相机图片／视频保存、关闭和续录。

```bash
uv run --extra test --extra pi --extra piper --extra intelrealsense \
  pytest tests/async_inference_local --confcutdir=tests/async_inference_local -q
```

## DAgger 纠正采集

DAgger 使用同一个进程内推理核心：模型始终由一个推理线程持有，接管和复位会撤销旧动作代际，
接管期间返回的推理结果不会下发。硬件接管、采集和状态机均在本包内实现，不调用远端部署代码。
自主阶段主从两臂接收相同的平滑后控制命令；人工阶段关闭计算机动作流，由物理主臂的 CAN
控制帧直接驱动从臂，不经过 GPU 或 Python 动作插值。SocketCAN 的本机回环帧不会进入接管桥。

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

checkpoint、相机名称/分辨率和 `fps` 的要求与 Deployment 相同。
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
  `--status_indicator.backend=none` 可关闭可选状态灯。

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
