# 单机本地推理架构

[架构索引](README.md) · [本地部署指令](../workflows/deployment/local.md) · [本地 DAgger 指令](../workflows/dagger/local.md)

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

正常策略动作通过 streaming 路径下发；复位经 `RobotIO.send_planned()` 路由。普通 Piper 使用 planned `move_j`，本地 DAgger 进入双臂控制器的 planned 路径；两者复位均绕过 streaming EMA。

Piper 适配将相机读取放在电机锁之外，避免等相机时占住 CAN 控制锁。
底层驱动没有独立电机/相机接口时，程序化接入其他机器人使用保守的串行 I/O；
当前 CLI 注册的是 `piper_follower`。

推理线程由新观测事件唤醒，始终使用最新有效观测连续重规划。
采集频率、策略动作频率和插值后的电机命令频率分别配置。
不要把提高模型推理速度等同于可以修改训练数据对应的 `fps`。

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
uv run --locked --extra test --extra pi --extra piper --extra intelrealsense --extra dataset \
  pytest tests/async_inference_local --confcutdir=tests/async_inference_local -q
```

DAgger 使用同一个进程内推理核心：模型始终由一个推理线程持有，接管和复位会撤销旧动作代际，
接管期间返回的推理结果不会下发。硬件接管、采集和状态机均在本包内实现，不调用远端部署代码。
自主阶段主从两臂接收相同的平滑后控制命令；人工阶段关闭计算机动作流，由物理主臂的 CAN
控制帧直接驱动从臂，不经过 GPU 或 Python 动作插值。SocketCAN 的本机回环帧不会进入接管桥。

更多本地 DAgger 数据语义与生命周期见 [本地 DAgger](../workflows/dagger/local.md)，启动和交互见 [本地部署](../workflows/deployment/local.md)。
