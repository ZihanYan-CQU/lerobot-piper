# uv 环境统一与混用排查记录

[问题索引](README.md) · [日常环境准备](../workflows/setup/README.md)

## 原因与范围

2026-11-07 检查实际仓库 `/home/cpy/桌面/yzh/piper/lerobot`：主 `.venv` 原本已经是 uv-managed Python 3.12.14，`include-system-site-packages=false`；混用来自启动器回退、终端变量和新增仿真环境。

| 原来哪里混用 | 证据 | 整理后的入口 |
| --- | --- | --- |
| 终端默认 Conda base | `CONDA_PREFIX=~/anaconda3`；裸 Python 是 3.13.5 | 终端直接 `uv run --no-sync` 使用根 `.venv`；运行前按环境页检查终端变量 |
| uv 只存在于旧 Conda 环境 | `~/anaconda3/envs/lerobot/bin/uv` 为独立二进制，PATH 找不到 uv | 独立 `~/.local/bin/uv` 加入 PATH，用 `command -v uv` 核对 |
| GitHub GUI 自动查找 Conda | 项目缺 CustomTkinter 时会尝试固定 Anaconda / Miniconda 路径 | [启动器](../../utils/github_upload_gui.sh) 统一通过 uv 固定项目环境，缺包提示同步 |
| GUI 依赖临时安装 | `uv --with` 提供 PyQt5 / CustomTkinter，自启动可能记录缓存解释器 | 根项目 `desktop` extra 与锁文件；WN-A100 自启动直接调用固定项目环境的 uv 命令 |
| 仿真借 Conda 包与 C++ 库 | 旧 `outputs/piper_sim/venv` 使用 `--system-site-packages`；`run.sh` 预加载 Conda `libstdc++` | [独立仿真项目](../workflows/simulation/README.md) 使用 `examples/piper_sim/.venv` 和自己的锁文件 |
| 安装提示指错环境 | 审查 GUI 固定建议 `conda install`；监控提示无目标 `uv pip install` | 提示使用项目 extra；手工安装必须指定解释器 |

uv 可执行文件放在 Conda 目录，不代表它会使用 Conda Python。整改前已实际用该 uv 执行 `run --no-sync python`，得到根 `.venv/bin/python` 与 uv-managed `sys.base_prefix`。真正要检查的是解释器、包路径和动态库来源。[uv 环境选择说明](https://docs.astral.sh/uv/pip/environments/)

## 现在各环境负责什么

| 环境 | 用途 |
| --- | --- |
| 根 `.venv` | Piper 数采、训练、部署、数据工具、上传 GUI、WN-A100；由根 `pyproject.toml` / `uv.lock` / `.python-version` 维护 |
| `examples/piper_sim/.venv` | 独立 MuJoCo 仿真；使用当前仓库 editable 源码和仿真锁文件，不继承 Conda 包 |
| `/usr/bin/python3` | 系统 GTK 设备绑定，入口为 [device_bindings/start.sh](../../src/lerobot/yzh/device_bindings/start.sh) |

两套 `.venv` 都由 uv 管理，主项目和仿真分别维护自己的锁文件；依赖按用途隔离。

终端流程直接使用 `uv run --no-sync`。它继承当前 shell 的变量；先按[环境准备](../workflows/setup/README.md#1-选择所需环境)退出 Conda 并清理外部 Python / 动态库 / uv 覆盖项。`python-preference=only-managed` 本身只约束 Python 来源。WN-A100 桌面图标和自启动直接调用 uv，在桌面命令中固定解释器并清理变量；GitHub 图标由自己的 [启动器](../../utils/github_upload_gui.sh) 调用 uv。

此前终端还带有 `/usr/local/cuda-11.8/lib64`，而主环境 Torch 为 `2.11.0+cu128`。这只是潜在库搜索冲突线索，不是已经确认的 CUDA 故障；常规流程按环境页清除该外部覆盖，专门依赖自定义 CUDA 工具链的实验需独立配置并验证。

## 复查命令

在实际仓库根目录执行；以下检查不连接硬件：

```bash
cd /home/cpy/桌面/yzh/piper/lerobot
command -v uv
uv run --no-sync python -c 'import sys; print(sys.executable); print(sys.base_prefix)'
uv run --no-sync python -c 'import tkinter, customtkinter; from PyQt5 import QtCore; print(QtCore.QT_VERSION_STR)'
cat .venv/pyvenv.cfg

# 仿真先按其流程准备独立环境和模型资源，再做无硬件诊断。
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl uv run --project examples/piper_sim --no-sync piper-sim doctor
```

首个 Python 应位于本仓库 `.venv/bin/python`，基础解释器应来自 `~/.local/share/uv/python`。GUI 包导入通过只说明依赖可加载，不代表已经验证实际开窗或硬件流程。

## 旧入口退役与旧环境对照

旧仿真的 `setup.sh`、`run.sh` 和 `run-conda-legacy.sh` 已移除。仿真统一从独立 uv 项目启动，环境缺失时按[仿真流程](../workflows/simulation/README.md)同步标准 `.venv`。

本机仍保留 `outputs/piper_sim/uv-venv`、`outputs/piper_sim/venv` 以及 `base-prefix` / `uv-path`，当前入口已不再读取。新机器只重建上表中的 uv 环境；原有数据集、模型、视频和日志继续按各自目录使用。

旧 Conda `lerobot` 环境仅保留作显式人工诊断或对照。以下路径已确认仍存在；需要时可检查其解释器，不影响当前 shell：

```bash
"$HOME/anaconda3/envs/lerobot/bin/python" -c 'import sys; print(sys.executable); print(sys.base_prefix)'
```

这不代表当前数采、部署、GUI 或仿真已在旧环境验证通过。日常 uv 命令不会自动回退；不要把旧环境的包或 C++ 库路径注入新的 `.venv`。

上传 GUI 已移除任意 Python 解释器回退，旧 `GITHUB_UPLOAD_PYTHON` 变量会被清除。所有依赖都准备在根 `.venv`，缺包按[环境准备](../workflows/setup/README.md)补齐。

历史研究文档中的 Conda 测试保留当时事实，日常启动以流程文档中的 uv 命令为准。

## 中文路径下的 Qt 插件

本机实际界面验证发现，PyQt 的 `QLibraryInfo.PluginsPath` 会把仓库路径中的“桌面”变成 `??`，导致 Qt 窗口找不到平台插件。WN-A100 在创建 Qt 窗口前，从当前加载的 PyQt5 模块定位真实插件目录并设置 `QT_PLUGIN_PATH` 与 `QT_QPA_PLATFORM_PLUGIN_PATH`；系统安装没有 bundled 插件目录时保留原有查找机制。无需搬动仓库或切回 Conda。

Qt 路径修复位于应用内，直接 `uv run --no-sync python -m lerobot.yzh.wn_a100` 也会生效，无需脚本前缀。验证方式为中文路径回归测试及 Qt 离屏控件检查，不启动远程监控轮询。
