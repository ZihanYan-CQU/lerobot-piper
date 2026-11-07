# Deployment 配置说明

[部署目录](README.md) · [连接与迁移](connection.md)

以下命令均从仓库根目录执行。`--config_path` 加载 JSON，额外 CLI 参数覆盖同名字段。

## 客户端

在仓库根目录运行：

```bash
uv run --no-sync python -m lerobot.async_inference.client.robot_client_deployment \
  --config_path=src/lerobot/async_inference/config/robot_client_deployment.json
```

配置文件：[config/robot_client_deployment.json](../../../src/lerobot/async_inference/config/robot_client_deployment.json)。其中保存了 Piper / Pi05 guided-RTC 的完整启动参数，使用前按实际 checkpoint 修改 `pretrained_name_or_path`。当前客户端 JSON 使用 `image_transport_mode=lerobot_float32`、`chunk_size_threshold=0.5`，两项 `teleop_*_alpha` 都为 `1`；完整 CLI 配方是独立示例，参数可能不同。

`server_address`、模型路径、`robot.cameras` 和 `task` 按当前部署机器填写。使用前按实际环境修改这四处；模型路径由服务端读取，相机设备路径属于客户端。JSON 中直接填写实际值，无需再设置 `SERVER_ADDRESS`、`MODEL_ROOT`、`CAMERAS`、`TASK`，也不会展开 `$变量` 或 `${变量}`。

`disable_proxy: true` 会在创建机器人连接和 gRPC 通道前，清除当前客户端进程中的 `http_proxy`、`https_proxy`、`all_proxy`、`grpc_proxy` 及其大写形式，因此无需保留原命令开头的八个 `env -u`。此配置不影响父 shell 和独立执行的 `uv sync` 依赖安装；如需让 uv 下载依赖也绕过代理，仍需在 uv 前使用 `env -u`。该选项在未指定时默认为 `false`。

`--config_path` 后必须给出文件路径，支持 `--config_path=路径` 和 `--config_path 路径`，也支持绝对路径。命令行上额外指定的参数会覆盖 JSON 中对应的值，例如临时修改任务和关节平滑系数：

```bash
uv run --no-sync python -m lerobot.async_inference.client.robot_client_deployment \
  --config_path=src/lerobot/async_inference/config/robot_client_deployment.json \
  --task="Pick up the red block." \
  --robot.teleop_joint_alpha=0.5
```

## 服务端

在 GPU 服务器的仓库根目录运行：

```bash
uv run --no-sync python -m lerobot.async_inference.server.policy_server_deployment \
  --config_path=src/lerobot/async_inference/config/policy_server_deployment.json
```

配置文件：[config/policy_server_deployment.json](../../../src/lerobot/async_inference/config/policy_server_deployment.json)。其中保存了 guided-RTC 窗口 20、最大引导权重 10、LINEAR 调度、模型复用，以及 `pi05_inference.profile=lerobot`、`compile_model=true`、`compile_mode=max-autotune` 和 1800 秒预热超时等参数。显式的编译设置会覆盖 profile 的默认编译设置。

`host=127.0.0.1`、`port=8080` 和故障日志路径 `/home/yzh/具身智能/DEBUG日志/deployment_guided_crash.log` 使用仓库当前示例值，适用于通过 SSH 隧道连接的用法。按实际部署修改 JSON 中的 `host`、`port` 和 `fault_log_path`，无需再定义 `SERVER_HOST`、`SERVER_PORT`、`FAULT_LOG_DIR`；JSON 不展开 shell 变量。模型路径仍由客户端指定。

JSON 中的 `"cuda_visible_devices": "6"` 选择 GPU 6，无需再在命令前设置 `CUDA_VISIBLE_DEVICES`。服务端会在创建推理服务、加载模型和初始化 CUDA 前设置该环境变量。该配置会覆盖当前进程已有的同名环境变量；省略此字段或设为 `null` 则保留原环境。使用其他 GPU 时直接修改字符串，也可写成 `"4,5"`。客户端的 `policy_device=cuda`（或 `cuda:0`）对应所选可见 GPU 中的第一张。若 CUDA 已初始化且配置要求更换 GPU，服务端会报错并要求重新启动进程。

服务端同样支持 `--config_path 路径` 和额外命令行参数覆盖，例如临时修改端口或 RTC 窗口：

```bash
uv run --no-sync python -m lerobot.async_inference.server.policy_server_deployment \
  --config_path=src/lerobot/async_inference/config/policy_server_deployment.json \
  --port=8081 \
  --rtc.execution_horizon=25
```

临时换卡可追加 `--cuda_visible_devices=5`，覆盖 JSON 中的设置。

三种客户端均可通过 `--image_transport_mode=lerobot_float32`（默认）或 `--image_transport_mode=openpi_uint8` 选择 π0.5 图像上传方式，需启用 `--resize_images_on_client=true`（默认）。前者保留原有 float32 缩放与无损压缩，后者匹配 OpenPI 的 PIL 等比例缩放/补边并上传 RGB uint8；服务端仍转换为浮点模型输入。新模式需两端同步更新 `shared/` 及各自入口代码，无需修改服务端启动参数；细节见 [Pi0.5 图像与精度说明](../../architecture/pi05-inference.md)。

依赖安装与 extra 选择统一见[环境准备](../setup/README.md)，训练与下载见[训练目录](../training/README.md)，完整运行命令见[部署目录](README.md)。
