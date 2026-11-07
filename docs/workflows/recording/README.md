# Piper 演示数采

[返回文档导航](../../README.md) · 前置：[环境安装](../setup/README.md) / [设备绑定](../tools/device-bindings.md) · 后续：[数据处理](../datasets/README.md) / [训练](../training/README.md)

数采用人工操纵主臂带动一台从臂，通过 `lerobot-record` 保存相机、从臂状态、示教动作和任务文字，不需要训练好的模型。这里的“单臂”是一台执行任务的从臂，仍需要一台主臂。

## 选择对应配方

| 情况 | 使用文档 | 相机与保存方式 |
| --- | --- | --- |
| 第一次使用本机，按当前硬件采集 | **[单臂现场指南](single-arm.md)** | 默认 `top` / `wrist` / `down` 三相机，自动生成时间戳目录，本地保存 |
| 沿用旧 `piper.txt` 的两相机数据 | [两相机分批配方](two-camera.md) | 显式 `top` / `wrist`，自定义批次名称，支持续采 |
| 模型自主执行 | [真机部署](../deployment/README.md) | 使用模型推理流程 |
| 模型执行时人工接管纠正 | [DAgger 数采](../dagger/README.md) | 使用 DAgger 接管与数据标记流程 |

三相机和两相机配方的 `top` 分辨率、旋转以及 `wrist` 旋转不同，不能当作同一训练 / 部署配置混用。按实际硬件和已有数据选择一份配方，并保持同一数据集的相机特征一致。

## 常用操作直接跳转

| 操作 | 指令位置 |
| --- | --- |
| 配置相机别名、激活主从 CAN | [设备绑定](../tools/device-bindings.md) |
| 本机三相机新建采集 | [完整启动命令](single-arm.md#32-复制启动命令) |
| 采集前验证两相机遥操作 | [遥操作命令](two-camera.md#1-先验证遥操作) |
| 按批次新建两相机数据 | [新建数据集](two-camera.md#2-新建数据集) |
| 追加已有数据 | [续采配方](two-camera.md#3-续采已有数据集)；三相机按下方说明 |
| 成功保存、重录、提前收工 | [按键表](single-arm.md#5-按键速查) / [四种现场情景](single-arm.md#6-四种常见操作情景) |
| 看懂录制、复位、退出时的灯色和闪烁 | [指示灯作用与配置](../tools/status-indicator.md) |
| 检查条数、逐条审查、合并或上传 | [数据处理](../datasets/README.md) |
| 排查环境、相机、按键和保存问题 | [问题记录](../../troubleshooting/recording.md) |

三相机续采时，复用[现场指南启动命令](single-arm.md#32-复制启动命令)，将 `DATASET_NAME` / `DATASET_DIR` 指向已有数据集、`resume` 改为 `true`，不要再生成新时间戳。`dataset.repo_id` 必须使用原标识，保持相机、机器人和 FPS 兼容；`dataset.num_episodes` 表示**本次追加**条数。

## 环境与使用范围

维护者首次准备环境：

```bash
uv sync --locked --inexact --extra piper-client --extra core_scripts
```

同一机器承担其他用途时，同时保留相应 extra，见[环境安装](../setup/README.md)。环境已经同步后可用 `uv run --no-sync python -m lerobot.scripts.lerobot_record`。命令使用项目 `.venv`；终端环境检查和缺少依赖时的处理见[环境安装](../setup/README.md)。

当前 Piper 固定使用 `can_master` 和 `can_follower`，没有 `robot.port` / `teleop.port` 参数。每次重启或重新插拔 CAN 后重新激活。连接、复位和退出都会触发机械臂动作；现场指南中的按键和退出步骤属于软件流程，不能替代硬件急停。
