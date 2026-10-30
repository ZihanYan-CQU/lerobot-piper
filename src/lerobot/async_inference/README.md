# 异步推理目录

代码按部署位置分为三类，主要入口的文件名和算法实现保持不变：

| 目录 | 职责 | 迁移到哪端 |
| --- | --- | --- |
| `shared/` | 双端共用的配置、通信对象、图像传输、诊断、RTC 与 RECAP 辅助逻辑 | 两端 |
| `server/` | 策略加载、推理、PI05 预热和 guided RTC 处理器 | 服务端 |
| `client/` | 机器人控制、插值、人工接管、DAgger / RECAP 数据记录 | 客户端 |

`shared/configs.py` 和 `shared/rtc_trained.py` 仍完整保留两端代码。`shared/` 不导入本目录下的服务端或客户端入口；服务端与客户端也不互相导入。子包的 `__init__.py` 不主动加载入口。

## 当前使用的入口

| 用途 | 服务端模块 | 客户端模块 |
| --- | --- | --- |
| Deployment | `lerobot.async_inference.server.policy_server_deployment` | `lerobot.async_inference.client.robot_client_deployment` |
| DAgger | `lerobot.async_inference.server.policy_server_dagger` | `lerobot.async_inference.client.robot_client_dagger` |
| RECAP | `lerobot.async_inference.server.policy_server_recap` | `lerobot.async_inference.client.robot_client_recap` |

原 deployment / dagger / recap 启动命令只需在 `async_inference.` 后增加 `server.` 或 `client.`，其余参数保持原样。例如：

```bash
uv run --locked --extra pi05-server python -m lerobot.async_inference.server.policy_server_deployment \
  --host=0.0.0.0 --port=8080 --fps=30 \
  --rtc.mode=guided --rtc.execution_horizon=20

uv run --locked --extra piper-client python -m lerobot.async_inference.client.robot_client_deployment --help
```

完整运行参数见 [Deployment](../yzh/README_DEPLOYMENT.md)、[DAgger](../yzh/README_DAGGER.md)、[RECAP](../yzh/README_RECAP.md)。

## Deployment JSON 配置

### 客户端

在仓库根目录运行：

```bash
uv run --locked --extra piper-client python -m lerobot.async_inference.client.robot_client_deployment \
  --config_path=src/lerobot/async_inference/config/robot_client_deployment.json
```

配置文件：[config/robot_client_deployment.json](config/robot_client_deployment.json)。其中保存了 Piper / Pi05 guided-RTC 的完整启动参数，使用前按实际 checkpoint 修改 `pretrained_name_or_path`。

`server_address`、模型根目录、`robot.cameras` 和 `task` 按仓库现有 [Piper Pi0.5 命令手册](../../../docs/README.md) 的公共配置填写。使用前按实际环境修改这四处；模型路径由服务端读取，相机设备路径属于客户端。JSON 中直接填写实际值，无需再设置 `SERVER_ADDRESS`、`MODEL_ROOT`、`CAMERAS`、`TASK`，也不会展开 `$变量` 或 `${变量}`。

`disable_proxy: true` 会在创建机器人连接和 gRPC 通道前，清除当前客户端进程中的 `http_proxy`、`https_proxy`、`all_proxy`、`grpc_proxy` 及其大写形式，因此无需保留原命令开头的八个 `env -u`。此配置不影响父 shell 和启动 Python 前的 uv 依赖同步；如需让 uv 下载依赖也绕过代理，仍需在 uv 前使用 `env -u`。该选项在未指定时默认为 `false`。

`--config_path` 后必须给出文件路径，支持 `--config_path=路径` 和 `--config_path 路径`，也支持绝对路径。命令行上额外指定的参数会覆盖 JSON 中对应的值，例如临时修改任务和关节平滑系数：

```bash
uv run --locked --extra piper-client python -m lerobot.async_inference.client.robot_client_deployment \
  --config_path=src/lerobot/async_inference/config/robot_client_deployment.json \
  --task="Pick up the red block." \
  --robot.teleop_joint_alpha=0.5
```

### 服务端

在 GPU 服务器的仓库根目录运行：

```bash
uv run --locked --extra pi05-server python -m lerobot.async_inference.server.policy_server_deployment \
  --config_path=src/lerobot/async_inference/config/policy_server_deployment.json
```

配置文件：[config/policy_server_deployment.json](config/policy_server_deployment.json)。其中保存了 guided-RTC 窗口 20、最大引导权重 10、LINEAR 调度、模型复用，以及 `pi05_inference.profile=lerobot`、`compile_model=true`、`compile_mode=max-autotune` 和 1800 秒预热超时等参数。显式的编译设置会覆盖 profile 的默认编译设置。

`host=127.0.0.1`、`port=8080` 和故障日志路径 `/home/yzh/具身智能/DEBUG日志/deployment_guided_crash.log` 按仓库现有命令手册填写，适用于通过 SSH 隧道连接的用法。按实际部署修改 JSON 中的 `host`、`port` 和 `fault_log_path`，无需再定义 `SERVER_HOST`、`SERVER_PORT`、`FAULT_LOG_DIR`；JSON 不展开 shell 变量。模型路径仍由客户端指定。

JSON 中的 `"cuda_visible_devices": "6"` 选择 GPU 6，无需再在命令前设置 `CUDA_VISIBLE_DEVICES`。服务端会在创建推理服务、加载模型和初始化 CUDA 前设置该环境变量。该配置会覆盖当前进程已有的同名环境变量；省略此字段或设为 `null` 则保留原环境。使用其他 GPU 时直接修改字符串，也可写成 `"4,5"`。客户端的 `policy_device=cuda`（或 `cuda:0`）对应所选可见 GPU 中的第一张。若 CUDA 已初始化且配置要求更换 GPU，服务端会报错并要求重新启动进程。

服务端同样支持 `--config_path 路径` 和额外命令行参数覆盖，例如临时修改端口或 RTC 窗口：

```bash
uv run --locked --extra pi05-server python -m lerobot.async_inference.server.policy_server_deployment \
  --config_path=src/lerobot/async_inference/config/policy_server_deployment.json \
  --port=8081 \
  --rtc.execution_horizon=25
```

临时换卡可追加 `--cuda_visible_devices=5`，覆盖 JSON 中的设置。

三种客户端均可通过 `--image_transport_mode=lerobot_float32`（默认）或 `--image_transport_mode=openpi_uint8` 选择 π0.5 图像上传方式，需启用 `--resize_images_on_client=true`（默认）。前者保留原有 float32 缩放与无损压缩，后者匹配 OpenPI 的 PIL 等比例缩放/补边并上传 RGB uint8；服务端仍转换为浮点模型输入。新模式需两端同步更新 `shared/` 及各自入口代码，无需修改服务端启动参数；细节见 Deployment 第 15 节。

Pi0.5 服务端使用 `uv sync --locked --extra pi05-server` 安装依赖，Piper 客户端使用 `uv sync --locked --extra piper-client`。客户端依赖包含固定提交的 `pyAgxArm`、gRPC、CAN、RealSense（`intelrealsense`，Linux 上为 `pyrealsense2`）、键盘控制和数据集支持，无需额外添加 `--extra intelrealsense`。其他策略还需选择对应的策略 extra。训练、下载和完整运行命令见 [Piper Pi0.5 命令手册](../../../docs/README.md)。

## 迁移服务端

两台机器需先具备相同版本的 LeRobot 基础代码和各自所需依赖。这里划分的是 `async_inference` 子包，并非独立安装包；`policies/`、`processor/`、`transport/`、`utils/` 等目录仍来自 LeRobot。

首次迁移应携带完整仓库，包括 `pyproject.toml`、`uv.lock` 和 `.python-version`，然后在目标机器运行对应的 `uv sync --locked --extra ...`。不要搬运旧机器的 `.venv`。下面的增量复制仅适用于目标机器已有配套基础代码和环境的情况；依赖配置有变化时也需同步上述文件。

在仓库根目录运行下列命令，将主机名和目标仓库路径替换为实际值：

```bash
rsync -av --exclude='__pycache__/' \
  src/lerobot/async_inference/__init__.py \
  src/lerobot/async_inference/shared \
  src/lerobot/async_inference/server \
  user@server:/path/to/lerobot/src/lerobot/async_inference/
```

迁移客户端时，同样携带 `__init__.py` 和 `shared/`，将 `server/` 换为 `client/`。

使用上述 JSON 启动方式时，也需将 `config/` 中对应的配置文件复制到目标机器的 `src/lerobot/async_inference/config/`。

**此次目录调整需要两端同步更新并重启。** 通信字段和协议版本没有变化，但 pickle 会记录对象的完整 Python 模块路径；通信类型现在位于 `lerobot.async_inference.shared.*`。旧平铺目录进程与新目录进程不能混用。外部启动脚本、服务配置中的模块路径也应更新为上表路径。

## 文件维护

- 会话代际编号常量及三个编解码函数已整体合入 `shared/rtc_remote.py`，原 `shared/rtc_session.py` 已删除。通信对象和会话编号规则现在统一维护，编号格式与函数行为不变。
- `server/policy_server_dagger.py` 虽然较短，但定义了独立入口和 DAgger 模型缓存默认值，RECAP 也复用它；保留可以直接比较三种运行方式。
- `client/piper_hil_dagger.py` / `piper_hil_recap.py` 分别承载硬件与记录逻辑，不并入已经较大的客户端入口。
- `shared/metadata_recap.py` 同时被服务端、客户端和记录器使用，保留它可以避免三者相互导入。

## 旧 RTC 命令迁移

旧 RTC 迭代链的八个实现文件及六个专用测试文件已删除。当前 deployment / dagger / recap 不依赖它们；公共协议和仍适用的传输测试继续保留，并通过当前 deployment 入口验证。

| 旧入口 | 当前入口 |
| --- | --- |
| `policy_server_rtc`、`policy_server_rtc_resettable` | `server.policy_server_deployment` |
| `robot_client_rtc` 及 interpolated / filtered / resettable 变体 | `client.robot_client_deployment` |
| `policy_server_rtc_resettable_dagger` | `server.policy_server_dagger` |
| `robot_client_rtc_interpolated_filtered_resettable_dagger` | `client.robot_client_dagger` |

表中的模块路径均位于 `lerobot.async_inference.` 下。两端应成对切换；旧配置中的 `observation_similarity_atol`、`cooldown_margin_steps`、`aggregate_fn_name`（客户端）和 `obs_queue_timeout`（服务端）不适用于当前 deployment 入口，应移除。[操作手册](../../../docs/README.md) 中的命令已同步迁移，其他外部脚本请按当前入口的 `--help` 检查参数。

基础异步入口 `server/policy_server.py`、`client/robot_client.py` 仍由 `examples/tutorial/async-inf/`、异步推理和 Unitree 文档使用，也有独立端到端测试，因此本轮保留。
