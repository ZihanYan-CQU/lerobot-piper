# 真机部署指令

[返回文档首页](../../README.md) · [环境准备](../setup/README.md) · [部署架构](../../architecture/deployment.md) · [部署排错](../../troubleshooting/deployment.md)

首次准备依赖：服务端执行 `uv sync --locked --inexact --extra pi05-server`；机器人电脑执行 `uv sync --locked --inexact --extra piper-client`；同机运行时同时选择这两个 extra。下方 `uv run --no-sync` 只运行已准备好的 `.venv`。

| 部署方式 | 入口 |
| --- | --- |
| GPU 在服务器上，使用现成 JSON | 本页快速启动 |
| GPU 在服务器上，完整 guided 参数 | [远端 guided 双端命令](guided.md) |
| GPU 在服务器上，完整 trained 参数 | [远端 trained 双端命令](trained.md) |
| GPU、机器人、相机在同一机器 | [本地部署](local.md) |
| SSH 隧道、代码迁移、旧入口替换 | [连接与迁移](connection.md) |
| JSON 参数覆盖、GPU 和图像模式 | [配置说明](configuration.md) |
| 执行中人工纠正并保存数据 | [DAgger 采集](../dagger/README.md) |
| 保存完整任务尝试与成功/失败 | [RECAP 采集](../recap/README.md) |
| 部署导出的 RLT 特征模型或 Stage 2 actor | [RLT 流程与条件](../rlt/README.md) |

普通远端 / 本地 deployment 当前没有驱动 USB 状态灯。数采和 DAgger / RECAP 的灯态范围见[指示灯说明](../tools/status-indicator.md)。

## 快速启动：现有 guided JSON

先完成[环境与设备准备](../setup/README.md)，两端均在仓库根目录运行。使用前编辑以下 JSON，直接填实际值；JSON 不会展开 shell 变量。

| 配置 | 需要核对 |
| --- | --- |
| [服务端 JSON](../../../src/lerobot/async_inference/config/policy_server_deployment.json) | `host`、`port`、`cuda_visible_devices`、`fault_log_path` |
| [客户端 JSON](../../../src/lerobot/async_inference/config/robot_client_deployment.json) | `server_address`、`pretrained_name_or_path`、`robot.cameras`、`task` |

模型路径由服务器读取，相机设备路径属于机器人电脑。服务器 `rtc.mode` 与客户端 `rtc_mode` 必须一致。

**1. 在 GPU 服务器启动：**

```bash
uv run --no-sync python -m lerobot.async_inference.server.policy_server_deployment \
  --config_path=src/lerobot/async_inference/config/policy_server_deployment.json
```

**2. 在机器人电脑建立 [SSH 隧道](connection.md#ssh-隧道)，然后启动客户端：**

```bash
uv run --no-sync python -m lerobot.async_inference.client.robot_client_deployment \
  --config_path=src/lerobot/async_inference/config/robot_client_deployment.json
```

当前 JSON 使用 GPU 6、guided RTC、客户端 `lerobot_float32` 图像上传、服务端 `lerobot` 精度档位及显式编译；客户端 `chunk_size_threshold=0.5`、关节和夹爪平滑系数均为 `1`。[完整 guided CLI 配方](guided.md)使用 GPU 7 和 `openpi` 精度档位。两种入口分别保存自己的参数，不要把它们视为完全相同的一套配置。CLI 额外参数可以覆盖 JSON，详见[配置说明](configuration.md)。

## 启动后的操作

| 输入 | 作用 |
| --- | --- |
| `/start` | 使用配置任务开始运行 |
| `/task 新任务描述` | 修改任务描述 |
| `/reset` | 停止当前轮次、清空动作并按配置复位 |
| `/status` | 查看状态 |
| `/stop` | 断开并退出远端客户端 |

本地入口的退出命令为 `/quit`，按[本地部署页](local.md)操作。使用 trained RTC 前先执行[对应训练](../training/trained.md)，并核对训练上限、固定前缀预算和动作块长度。
