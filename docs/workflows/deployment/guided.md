# 远端真机部署：guided RTC

[流程目录](README.md) · [环境准备](../setup/README.md) · [SSH 连接](../deployment/connection.md)

所有命令从对应机器的仓库根目录执行。顺序为 GPU 服务器 → 机器人电脑建立 SSH 隧道 → 机器人客户端。服务端与客户端使用同一代码版本和 RTC 模式，同一端口只启动一个服务。

## 1. GPU 服务器（S1）

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
  --rtc.mode=guided \
  --rtc.execution_horizon=20 \
  --rtc.max_guidance_weight=10 \
  --rtc.prefix_attention_schedule=LINEAR \
  --reuse_last_policy=true \
  --pi05_inference.compile_mode=max-autotune \
  --pi05_inference.profile=openpi \
  --pi05_inference.warmup_timeout_s=1800 \
  --pi05_inference.compile_model=true \
  --debug_rtc_diagnostics=false \
  --debug_cuda_sync=false \
  --rtc.debug=false \
  --fault_log_path="$FAULT_LOG_DIR/deployment_guided_crash.log"
```

本完整 CLI 配方显式使用 `pi05_inference.profile=openpi` 和编译；[JSON 快速启动](README.md)当前为 `lerobot` profile、GPU 6，两者不是同一份配置。当前 JSON 还使用 `chunk_size_threshold=0.5`、两项 alpha 为 `1`、`lerobot_float32`；下方 CLI 保留独立的实验参数，不是 JSON 的等价展开。

## 2. 机器人电脑（C1）

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
  --pretrained_name_or_path="${MODEL_ROOT}/pi0.5_shapeMatching_b32_s20000/checkpoints/020000/pretrained_model" \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --chunk_size_threshold=1.0 \
  --fps=30 \
  --interpolation_multiplier=2 \
  --adaptive_rtt_trigger=true \
  --rtt_trigger_margin_steps=5 \
  --robot.teleop_gripper_alpha=0.25 \
  --robot.teleop_joint_alpha=0.25 \
  --reset_to_initial_position=true \
  --reset_duration_s=3.0 \
  --reset_fps=50 \
  --rtc_mode=guided \
  --resize_images_on_client=true \
  --image_transport_mode=openpi_uint8 \
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

本配方使用 `openpi_uint8` 图像上传；其他配方默认 `lerobot_float32`。两种模式像素值不完全相同，选择规则见[配置说明](configuration.md)。

启动后使用 `/start` 开始，`/task 新任务` 改任务，`/reset` 停止当前轮次并复位，`/status` 查看状态，`/stop` 退出。
