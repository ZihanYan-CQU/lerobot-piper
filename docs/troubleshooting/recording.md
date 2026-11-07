# Piper 普通数采问题

[返回排错导航](README.md) · [三相机现场指南](../workflows/recording/single-arm.md) · [数采配方导航](../workflows/recording/README.md)

适用于人工主臂示教的 `lerobot-record`。模型执行中的人工接管问题见 [DAgger 排错](dagger.md)。

| 现象 | 处理方式 |
| --- | --- |
| `./.venv/bin/python` 不存在 | 先确认位于正确仓库根目录；环境缺失时按[环境安装](../workflows/setup/README.md)重建项目环境，不随意切换系统 Python |
| `uv: 未找到命令` | 按[环境安装](../workflows/setup/README.md)提供独立 uv；用 `command -v uv` 核对 PATH，本机独立可执行文件为 `~/.local/bin/uv` |
| 找不到 `can_master` 或 `can_follower` | 在设备面板重新激活；电脑重启或 CAN 插拔后需再次执行 |
| 相机路径不存在，尤其是 `video_down` | 默认配置需要三路；在 RGB 页签补齐别名，保存时保留整组三路。历史两相机设备使用[显式两相机配方](../workflows/recording/two-camera.md) |
| 相机打开失败或超时 | 关闭预览和其他控制程序，检查 USB 连接、别名和实际画面；绑定冲突见[工具排错](tools.md#相机和-can) |
| 方向键没有反应 | 将焦点放在采集终端，用英文 `n/r/q`，并确认不在倒计时、回零或编码阶段；监听环境说明见下方 |
| 按右键后没有立即显示保存成功 | 正常流程先回零、恢复环境，再保存和编码视频；等待 `episode N 采集完毕`，不要提前关闭终端 |
| 物体还没摆好就开始下一条 | 复位计时会自动结束；下次启动前增大 `dataset.reset_time_s`，本入口没有暂停键 |
| 按 Esc 后保存了未完成的任务 | Esc 不等于丢弃，会保留当前已有帧；在本轮尚未保存时用重录流程丢弃，或采后[审核过滤](../workflows/datasets/README.md#2-人工审查与过滤) |
| 目录已经存在，无法新建 | 重新生成时间戳名称或改批次名，不删除已有数据绕过错误；确实追加时使用数采页的续采步骤 |
| 实际帧率偏低或循环超时 | 关闭额外预览，检查相机和磁盘负载，保存日志供排查；目标 `30` FPS 不代表实际已达到 |
| 温度异常或超限 | 停止安排后续采集，按现场流程处理设备；温度监控和软件退出键不能替代硬件急停 |
| 审核窗口缺少 `customtkinter` | 使用[数据审查命令](../workflows/datasets/README.md#2-人工审查与过滤)提供依赖，或由维护者按下方方法补齐现有环境 |
| 可选指示灯连接失败或看不懂灯色 | 默认允许无灯继续，以终端提示为准；`required=true` 会要求初始化成功。依赖、USB 权限和运行中降级见[指示灯排错](status-indicator.md) |

## 终端与按键

X11 使用全局键盘监听；Wayland / 无桌面 SSH 回退到交互 TTY。保持运行采集的终端焦点，使用英文输入模式，优先 `n`（结束当前循环）、`r`（重录）、`q`（停止），避免远程方向键转义序列被拆分。没有可用桌面且 stdin 不是 TTY 时，快捷键不可用；不要把交互采集放到后台。

倒计时、自动回零和保存过程不保证立即处理按键，不要连续提前按键。空格、Enter、上下方向键没有暂停 / 开始功能。完整边界和“丢弃并收工”操作见[现场按键与情景](../workflows/recording/single-arm.md#5-按键速查)。

## 为既有环境补齐审核工具

先在仓库根目录补齐锁定的 GUI 和数据集依赖，再从固定项目环境启动：

```bash
uv sync --locked --inexact --extra desktop --extra dataset
uv run --no-sync python -m lerobot.yzh.lerobot_review_dataset
```

`--inexact` 保留现有数采 / 训练包。不要把缺包问题转交给另一套 Conda 环境；Tk 字体或开窗问题应针对实际解释器检查。普通数采可用并不代表桌面依赖已经安装，完整操作见[数据处理](../workflows/datasets/README.md)。
