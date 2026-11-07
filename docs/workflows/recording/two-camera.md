# 两相机遥操作与分批演示配方

[返回数采导航](README.md) · 前置：[环境](../setup/README.md) / [设备绑定](../tools/device-bindings.md) · 后续：[数据处理](../datasets/README.md) / [训练](../training/README.md)

本页保留旧 `piper.txt` 的 **top + wrist 两相机**配方，适用于仍使用该硬件和图像约定的历史数据。第一次使用本机默认三相机配置，请先读[单臂现场指南](single-arm.md)。本页用于主臂带从臂采集演示。模型接管纠正数据见 [DAgger](../dagger/README.md)，模型自主执行见[真机部署](../deployment/README.md)。命令在机器人电脑的仓库根目录执行。

## 启动前与公共配置

先在[设备绑定面板](../tools/device-bindings.md)确认主臂为 `can_master`、从臂为 `can_follower`，波特率 `1000000`。当前 Piper 实现固定使用这两个 CAN 名称，**没有** `--robot.port` / `--teleop.port` 配置项。不要根据 USB 设备号大小猜测主从臂。

连接、复位和退出流程会使机械臂回零；执行前腾出主从臂运动空间，结束后等程序完成回零再断电。单独预览相机的程序应先退出，避免与采集占用同一视频设备。

先修改并执行这段公共配置；每个新终端都需执行一次：

```bash
DATASET_NAMESPACE='block'
DATASET_NAME='shape-matching_0-10'
DATASET_BASE_DIR="$HOME/.cache/huggingface/lerobot"
DATASET_REPO_ID="${DATASET_NAMESPACE}/${DATASET_NAME}"
DATASET_ROOT="${DATASET_BASE_DIR}/${DATASET_REPO_ID}"
TASK='Place the block on the shape-matching board.'
NUM_EPISODES=10
EPISODE_SECONDS=90
RESET_SECONDS=30

CAMERAS='{
  "top": {
    "type": "opencv",
    "index_or_path": "/dev/video_top",
    "width": 640,
    "height": 480,
    "fps": 30,
    "rotation": 0
  },
  "wrist": {
    "type": "opencv",
    "index_or_path": "/dev/video_wrist",
    "width": 640,
    "height": 480,
    "fps": 30,
    "rotation": 180
  }
}'
```

此处保留旧 `piper.txt` 的两相机、30 FPS 配方。`rotation` 与分辨率需要按实际画面调整；当前代码默认还有 `down` 相机，本页显式传入完整 `CAMERAS` 后只使用 `top`、`wrist`。后续训练和部署必须保持相机名称及预处理约定一致。

`DATASET_ROOT` 是最终数据集目录，将包含 `meta/`、`data/`、`videos/`；不是这些目录的父级。新建采集时使用尚不存在的目录。`DATASET_NAMESPACE` 在本地可自定义，上传时再使用有权限的 HF 用户名 / 组织名。

## 1. 先验证遥操作

执行公共配置后，运行：

```bash
uv run --no-sync lerobot-teleoperate \
  --robot.type=piper_follower \
  --robot.id=my_follower_arm \
  --robot.cameras="$CAMERAS" \
  --teleop.type=piper_leader \
  --teleop.id=my_leader_arm \
  --display_data=true \
  --fps=30
```

确认主从臂方向、夹爪以及两路画面正确后，按 `Ctrl+C` 结束遥操作，等待断开流程完成，再启动采集。

## 2. 新建数据集

执行公共配置后，运行：

```bash
uv run --no-sync lerobot-record \
  --robot.type=piper_follower \
  --robot.id=my_follower_arm \
  --robot.cameras="$CAMERAS" \
  --teleop.type=piper_leader \
  --teleop.id=my_leader_arm \
  --display_data=true \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.no_stamp=true \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.episode_time_s="$EPISODE_SECONDS" \
  --dataset.reset_time_s="$RESET_SECONDS" \
  --dataset.fps=30 \
  --dataset.push_to_hub=false
```

`no_stamp=true` 保留此处自管的名称，便于后续分批合并。若移除该参数，当前代码会为新建数据集的 `repo_id` 追加日期时间；续采时需使用实际生成的名称。另录一批时修改 `DATASET_NAME` 并重新执行公共配置，避免复用已有目录。

旧“长尾夹收纳”配方只需要把公共配置改为以下值，再运行相同的录制命令：

```bash
DATASET_NAMESPACE='take_storage'
DATASET_NAME='storage_binder'
DATASET_REPO_ID="${DATASET_NAMESPACE}/${DATASET_NAME}"
DATASET_ROOT="${DATASET_BASE_DIR}/${DATASET_REPO_ID}"
TASK='Put the long tail binder clips on the desk back into their box.'
NUM_EPISODES=50
EPISODE_SECONDS=60
RESET_SECONDS=30
```

### 采集过程与按键

三相机和两相机使用相同的录制流程。开始前阅读[采集到保存的完整过程](single-arm.md#4-一条演示从开始到保存的完整过程)和[按键速查](single-arm.md#5-按键速查)：`→/n` 结束当前循环、`←/r` 重录、`Esc/q` 停止；停止不会自动丢弃当前片段。

## 3. 续采已有数据集

公共配置中的 `DATASET_REPO_ID`、`DATASET_ROOT` 改为已有数据集，保持相机、FPS 和机器人设置兼容，再运行：

```bash
uv run --no-sync lerobot-record \
  --robot.type=piper_follower \
  --robot.id=my_follower_arm \
  --robot.cameras="$CAMERAS" \
  --teleop.type=piper_leader \
  --teleop.id=my_leader_arm \
  --display_data=true \
  --resume=true \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.no_stamp=true \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.episode_time_s="$EPISODE_SECONDS" \
  --dataset.reset_time_s="$RESET_SECONDS" \
  --dataset.fps=30 \
  --dataset.push_to_hub=false
```

这里 `num_episodes` 表示**本次追加**的轮数，不是期望最终总数。续采分支不会再添加时间戳。

## 4. 采后检查

退出后等待视频编码完成，按[数据处理](../datasets/README.md)查看 episode、多相机同步、动作和任务文字，再决定过滤、合并和上传。采集故障统一查[问题记录与排错](../../troubleshooting/recording.md)。

参数依据：[Piper 从臂配置](../../../src/lerobot/robots/piper_follower/config_piper_follower.py)、[主臂配置](../../../src/lerobot/teleoperators/piper_leader/config_piper_leader.py)、[录制入口](../../../src/lerobot/scripts/lerobot_record.py)、[数据集录制配置](../../../src/lerobot/configs/dataset.py)、[键盘控制](../../../src/lerobot/utils/keyboard_input.py)。

指示灯的绿常亮同时用于遥操作和录制，是否写盘取决于当前入口；颜色、闪烁及配置见[指示灯说明](../tools/status-indicator.md)。
