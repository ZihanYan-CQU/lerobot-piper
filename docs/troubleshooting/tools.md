# 设备、数据与监控工具排错

[排错索引](README.md) · [工具流程](../workflows/tools/README.md) · [数采指令](../workflows/recording/README.md)

普通数采的环境、按键、复位和保存顺序问题统一见 [Piper 数采排错](recording.md)。

## 相机和 CAN

- 旧 `/etc/udev/rules.d/99-cameras.rules` 与工具的 `99-lerobot-cameras.rules` 不可同时定义同一个 `video_top` / `video_wrist`。先处理重复旧映射，再使用绑定工具。
- udev 别名不改变设备权限；无唯一序列号而按 ID_PATH 绑定时，保持 USB 插口和拓扑一致。
- CAN 配置不持久化，重启或重新插拔后再次激活；先停止占用对应 CAN 的程序。主从端口不能相同，不能按 USB 数值大小猜角色。
- 程序默认相机包含 down；只有 top/wrist 的设备使用数采流程的显式 cameras 配方。预览程序退出后再启动录制，避免争用。
- 绑定面板要系统 Python / GTK / 图形会话；不要用不带 GTK 的 Conda 解释器替代 `/usr/bin/python3`。

## 指示灯

灯色与阶段查[指示灯作用](../workflows/tools/status-indicator.md)；HID 依赖、设备连接、权限及 `required` 的实际作用查[指示灯排错](status-indicator.md)。

## 数据工具

审查 / 过滤、Qt 合并、旧界面兼容、输出位置与停止后的残留目录，统一见[数据工具排错](datasets.md)。正常操作见[数据处理流程](../workflows/datasets/README.md)，模块职责见[数据工具架构](../architecture/dataset-tools.md)。

## WN-A100 训练状态

状态解释：

- 初次读取 / 新空日志：暂不生成任务轮播页或圆环，详细训练信息中显示等待状态，不编造 0% 或完成率。
- 日志超过 180 秒未写入：已有弧段变灰，注明“日志未更新”。这不等于训练已停止。
- SSH 超时、断连或单份日志读取失败：保留上次步数，变灰并标明失败；另一份正常日志仍独立更新。
- 最新进度之后出现 Python traceback：保留灰色进度，提示检查训练日志。
- 新任务、日志重建或截断：使用新日志的最新计数，允许进度下降或从零开始，不沿用上次任务进度。
- 达到最后一步：显示“步数已达 100%”，不保证后续最终检查点保存也已完成。
- ETA 是日志的最新估计，不是自行倒数的计时器；灰色状态明确标注为上次估计。


监控只复用已登录 SSH ControlMaster，缺失时不会自行密码登录。按 [WN-A100 工具流程](../workflows/tools/wn-a100.md)的本机终端命令建立共享连接。日志移动后同步修改 [WN-A100 配置](../../src/lerobot/yzh/wn_a100/config.json)；网页 PID 不能作为 SSH 终端 kill 的依据，监控展示的有效 PID 必须来自当前 SSH 命名空间。

自启动项记录 uv、项目和根 `.venv` 解释器的绝对路径，直接调用 uv；移动仓库后重新安装自启动项。日常终端启动使用流程页中的 `uv run --no-sync` 命令。

## GitHub 上传工具

当前上传窗口使用 PyQt5；缺包时运行 `uv sync --locked --inexact --extra desktop`。启动器统一使用 uv 项目环境，解释器与动态库排查见[环境整理记录](uv-conda.md)。默认代码模式遇到目标分支被其他人更新时，重新预览并上传；完整历史模式遇到分歧时不强推，已经生成的本地提交保留。预览后文件或分支变化应重新确认。
