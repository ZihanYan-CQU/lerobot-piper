# 远端真机部署：trained RTC

[流程目录](README.md) · [环境准备](../setup/README.md) · [SSH 连接](../deployment/connection.md)

所有命令从对应机器的仓库根目录执行。顺序为 GPU 服务器 → 机器人电脑建立 SSH 隧道 → 机器人客户端。服务端与客户端使用同一代码版本和 RTC 模式，同一端口只启动一个服务。

## 1. GPU 服务器（S2）

本终端先定义地址与日志目录，再运行下方服务。常规推理取消会改变 CUDA 时序的调试环境变量。模型由客户端指定，服务端读取。

```bash
SERVER_HOST='127.0.0.1'
SERVER_PORT=8080
FAULT_LOG_DIR='/home/yzh/具身智能/DEBUG日志/'

mkdir -p "$FAULT_LOG_DIR"
export PYTHONFAULTHANDLER=1
unset CUDA_LAUNCH_BLOCKING TORCH_SHOW_CPP_STACKTRACES
```

```bash
CUDA_VISIBLE_DEVICES=7 uv run --no-sync python -m lerobot.async_inference.server.policy_server_deployment \
  --host="$SERVER_HOST" \
  --port="$SERVER_PORT" \
  --fps=30 \
  --inference_latency=0 \
  --rtc.mode=trained \
  --trained.delay_steps=25 \
  --reuse_last_policy=true \
  --pi05_inference.warmup_timeout_s=1800 \
  --debug_rtc_diagnostics=false \
  --debug_cuda_sync=false \
  --rtc.debug=false \
  --fault_log_path="$FAULT_LOG_DIR/deployment_trained_crash.log"
```

本配方未显式覆盖编译与精度参数，沿用所加载 checkpoint 的设置。

## 2. 机器人电脑（C2）

先按 [SSH 连接](../deployment/connection.md)建立隧道（同机可跳过）。在客户端的新终端定义本页变量，然后启动。`MODEL_ROOT` 是服务器路径，`CAMERAS` 中的设备属于机器人电脑；相机名称、任务文本、模型子目录与 checkpoint 都要按实际训练结果核对。

```bash
SERVER_ADDRESS='127.0.0.1:8080'
TASK='Place the block on the shape-matching board.'

# 服务器上的模型基础目录；具体模型和 checkpoint 在执行命令中填写。
MODEL_ROOT='/home/yzh/具身智能/微调模型/Lerobot'

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

```bash
env \
  -u http_proxy \
  -u https_proxy \
  -u HTTP_PROXY \
  -u HTTPS_PROXY \
  -u all_proxy \
  -u ALL_PROXY \
  -u grpc_proxy \
  -u GRPC_PROXY \
uv run --no-sync python -m lerobot.async_inference.client.robot_client_deployment \
  --server_address="$SERVER_ADDRESS" \
  --robot.type=piper_follower \
  --robot.cameras="$CAMERAS" \
  --policy_type=pi05 \
  --pretrained_name_or_path="${MODEL_ROOT}/pi0.5_shapeMatching_b32_s20000_trainedRTC_delay25/checkpoints/020000/pretrained_model" \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --fps=30 \
  --interpolation_multiplier=2 \
  --robot.teleop_gripper_alpha=1.0 \
  --robot.teleop_joint_alpha=1.0 \
  --reset_to_initial_position=true \
  --reset_duration_s=3.0 \
  --reset_fps=50 \
  --rtc_mode=trained \
  --trained.queue_threshold=25 \
  --trained.replan_on_result=true \
  --resize_images_on_client=true \
  --continuous_observation=true \
  --observation_max_age_s=0.1 \
  --startup_calibration_runs=3 \
  --startup_calibration_timeout_s=1800 \
  --latency_history_ttl_s=300 \
  --observation_capture_fps=30 \
  --debug_rtc_diagnostics=false \
  --log_rtc_timing=false \
  --debug_visualize_queue_size=false \
  --suppress_can_debug_logs=true \
  --task="$TASK"
```

命令只对当前客户端进程清除代理。客户端保持前台运行，策略频率 30 Hz，插值命令目标频率 60 Hz。

本配方使用默认 `lerobot_float32` 图像上传；如需 OpenPI PIL 缩放/补边，显式添加 `--image_transport_mode=openpi_uint8` 并保留 `--resize_images_on_client=true`，两端同步代码后重启。详见[部署配置](../deployment/configuration.md)。

启动后使用 `/start` 开始，`/task 新任务` 改任务，`/reset` 停止当前轮次并复位，`/status` 查看状态，`/stop` 退出。

## trained 参数配套

| 参数 | 本手册取值 | 要求 |
| --- | --- | --- |
| 训练 `policy.rtc_training_max_delay` | `25` | 模型必须实际在此设置下训练并保存 |
| 服务端 `trained.delay_steps` | `25` | 大于 0，且不超过训练最大前缀长度 |
| 客户端 `actions_per_chunk` | `50` | 至少是服务端 delay 的 2 倍，且不超过模型 chunk size |
| 客户端 `trained.queue_threshold` | `25` | 不小于服务端 delay，且小于动作块长度 |
| 客户端 `trained.replan_on_result` | `true` | 接纳结果后允许尽快发起下一次请求 |

trained 调度不使用 guided 的 `chunk_size_threshold`、`adaptive_rtt_trigger`、`rtt_trigger_margin_steps`。请加载真正训练过动作前缀条件的 checkpoint；[训练命令](../training/trained.md)。
