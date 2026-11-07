# 环境安装与机器迁移

[返回文档导航](../../README.md) · 下一步：[设备绑定](../tools/device-bindings.md) / [Piper 数采](../recording/README.md) / [训练](../training/README.md) / [真机部署](../deployment/README.md)

本页只负责环境、依赖和目录约定。命令使用 Bash，在**当前机器的仓库根目录**执行；本机原工作目录是 `/home/cpy/桌面/yzh/piper/lerobot`，其他机器按实际位置进入仓库。

本仓库的 Piper 工作流统一由 uv 管理和启动。主项目使用根 `.venv`，MuJoCo 仿真使用 `examples/piper_sim/.venv`；两者都是 uv 管理的 Python，按用途隔离依赖。设备绑定面板使用系统 Python / GTK。旧 Conda 环境不作为日常入口或自动回退；保留的旧环境只用于[显式诊断与对照](../../troubleshooting/uv-conda.md#旧入口退役与旧环境对照)。项目环境缺失时按本页重新同步。

## 0. 直接运行流程命令

Piper 数采、训练、部署和常用桌面工具统一使用仓库根目录 `.venv`。在仓库根目录执行：

```bash
# 检查当前项目使用的 Python；不会启动机械臂。
uv run --no-sync python -c 'import sys; print(sys.executable); print(sys.base_prefix)'
```

先按下一节安装所需依赖，日常直接执行 `uv run --no-sync lerobot-record ...`、`uv run --no-sync lerobot-train ...` 或 `uv run --no-sync python -m 模块 ...`。`--no-sync` 使用已准备好的项目环境，启动时不增删依赖。

uv 本身应独立安装并加入 PATH，见[官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)。本机使用 `~/.local/bin/uv`，可用 `command -v uv` 核对。设备绑定的 GTK 面板使用系统 Python；MuJoCo 使用[独立仿真环境](../simulation/README.md)。

核查证据、旧入口退役情况和外部环境混用排查见 [uv 环境整理记录](../../troubleshooting/uv-conda.md)。

## 1. 选择所需环境

先安装 uv 和 Git。如果当前终端激活了 Conda，先运行 `conda deactivate`，嵌套激活时重复退出。直接 `uv run` 会继承终端环境，不负责清理外部变量；安装或运行前若曾设置这些覆盖项，在当前终端清除：

```bash
unset UV_PROJECT_ENVIRONMENT UV_PROJECT UV_WORKING_DIR UV_PYTHON UV_CONFIG_FILE UV_NO_CONFIG
unset UV_PYTHON_PREFERENCE UV_NO_MANAGED_PYTHON UV_ENV_FILE UV_ISOLATED UV_NO_PROJECT
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
unset VIRTUAL_ENV
```

项目由 [.python-version](../../../.python-version) 固定 Python `3.12.14`，[pyproject.toml](../../../pyproject.toml) 要求 Python `>=3.12` 并使用 uv 管理的解释器；依赖版本由 [uv.lock](../../../uv.lock) 锁定。

| 机器用途 | 同步命令 | 覆盖范围 |
| --- | --- | --- |
| Pi0.5 训练 | `uv sync --locked --extra pi05-training` | 单卡 / 多卡、guided / trained RTC、数据集 |
| GPU 推理服务器 | `uv sync --locked --extra pi05-server` | deployment / DAgger、gRPC、Pi0.5、诊断 |
| Piper 真机部署 / DAgger | `uv sync --locked --extra piper-client` | gRPC、pyAgxArm、CAN、数据集、RealSense、键盘控制 |
| 单机 Pi0.5 + Piper 推理 | `uv sync --locked --extra pi05-server --extra piper-client` | 同一机器运行模型和机器人；见[本地部署](../deployment/local.md) |
| Piper 遥操作 / 普通数采，含实时画面 | `uv sync --locked --extra piper-client --extra core_scripts` | 上述机器人依赖及录制 / 遥操作 CLI 的可视化依赖 |
| 数据集查看 / 过滤 / 合并 | `uv sync --locked --extra dataset_viz --extra desktop` | 数据读写、视频解码、可视化和审查 GUI |
| WN-A100 / GitHub 上传 GUI | `uv sync --locked --extra desktop` | 使用 PyQt5；数据审核 GUI 的 CustomTkinter 也在此 extra，Pillow 为核心依赖 |
| 可选 USB 指示灯 | `uv sync --locked --inexact --extra gamepad` | 复用现有 extra 的 hidapi，同时会安装 pygame；配置见[指示灯](../tools/status-indicator.md) |

`piper-client` 已包含 `intelrealsense`，Linux 上会安装 `pyrealsense2`。实时显示所需的 `viz` 由 `core_scripts` 提供，不要只装 `piper-client` 后直接使用 `--display_data=true`。

同一机器承担多种用途时，一次列出所有需要的 extra，例如：

```bash
uv sync --locked \
  --extra pi05-training \
  --extra pi05-server \
  --extra piper-client \
  --extra core_scripts \
  --extra desktop
```

`uv sync` 默认会移除未选中的依赖，完整同步时要一次列出这台机器需要的全部 extra。向已经可用的机器补充某项能力时使用 `--inexact`，例如 `uv sync --locked --inexact --extra desktop`，保留已有的其他依赖。流程页采用这个增量准备方式；需要从头复现环境时按上表选择用途并集做完整同步。

日常运行统一使用 `uv run --no-sync 命令 参数...`。安装和运行分开，避免数采、训练或 GUI 启动时临时改变环境。

## 2. 系统与桌面依赖

uv 管理 Python 包；新机器仍需安装对应的 NVIDIA 驱动、FFmpeg 及共享库，并配置 CAN、相机设备和访问权限。设备绑定工具使用系统 Python/GTK，安装命令见[设备绑定](../tools/device-bindings.md)。桌面数据审查、GPU 监控和代码上传工具各自的额外依赖见[工具导航](../tools/README.md)。

普通 `lerobot-record` 的离散按键在 Wayland / SSH 下可回退到交互终端，见[数采控制键](../recording/single-arm.md#5-按键速查)；DAgger 的键盘接管要求按其[流程说明](../dagger/README.md)配置，不应套用普通数采的按键表。

默认 Python 包源是项目配置的清华源；Linux 上的 `torch`、`torchvision` 使用 PyTorch CUDA 12.8 专用源。`pyAgxArm` 从 `agilexrobotics/pyAgxArm` 的 GitHub 仓库安装，锁定提交 `4b0f06585db3324222616999afeda9037df2e8bd`。清华源不代理这两个外部来源。

## 3. 迁移与依赖维护

迁移时带上完整代码、`pyproject.toml`、`uv.lock` 和 `.python-version`，在新机器重新同步 `.venv`。不要直接复制旧 `.venv` 或 Conda 环境。

无需激活 Conda，也不要默认把 `$CONDA_PREFIX/lib` 加到 `LD_LIBRARY_PATH`。未指定解释器的 `uv pip install` 会考虑已激活的环境，可能选中 Conda；临时补包必须指定 `--python .venv/bin/python`，但这不会更新锁文件。正式流程在仓库根目录使用 `uv run --no-sync` 选择项目环境。[uv 环境选择规则](https://docs.astral.sh/uv/pip/environments/)

正式新增依赖应写入项目，例如 `uv add --optional piper-client 包名`，并同步提交更新后的 `pyproject.toml` 和 `uv.lock`。

`--locked` 报锁文件不一致时，先按[环境排错](../../troubleshooting/README.md)检查配套文件与源配置，再由维护者确认是否需要重新 `uv lock`。

OpenPI 使用自己的仓库与运行环境；本页的 Python 和 extra 只适用于 LeRobot。服务端与客户端应使用同一版本代码，并选择一致的用途和 RTC 模式。

## 4. 路径约定

服务器以 `/home/yzh/具身智能` 为基础目录，按“用途 / 框架或来源”分层：

| 内容 | LeRobot | OpenPI |
| --- | --- | --- |
| 基座模型 | `基座模型/Lerobot` | `基座模型/OpenPI` |
| 微调模型 | `微调模型/Lerobot` | `微调模型/OpenPI` |
| 训练日志 | `训练日志/Lerobot` | `训练日志/OpenPI` |
| 训练数据 | `训练数据` | 按实际数据来源选择 |
| 调试日志 | `DEBUG日志/CodeX` | 按实际用途选择 |

这些路径是现有机器的约定，可整体改成新机器上的基础目录。具体数据集名称、模型目录、checkpoint 和 GPU 在对应流程命令中指定。

客户端传入的模型路径由**服务器**读取；相机 `/dev/video_*` 路径属于**机器人电脑**。JSON 配置不展开 `$变量`，需要填写实际路径；Bash 命令中的变量会由当前终端展开。

## 5. 按需登录

下载受限模型、上传数据或启用 W&B 时，在运行该操作的机器上交互登录：

```bash
uv run --no-sync hf auth login
uv run --no-sync wandb login
```

W&B 还需要在训练命令设置 `--wandb.enable=true`。`--wandb.disable_artifact=true` 关闭模型 Artifact 上传，`WANDB_MODE=offline` 用于离线记录。认证信息不写入共享命令或文档。
