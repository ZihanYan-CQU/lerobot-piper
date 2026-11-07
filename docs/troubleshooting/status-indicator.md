# 状态指示灯排错

[问题目录](README.md) · [使用与状态速查](../workflows/tools/status-indicator.md) · [实现边界](../architecture/status-indicator.md)

先确认正在使用的是[已接入入口](../workflows/tools/status-indicator.md#哪些流程会控制灯)。普通部署没有配置这套灯时，不亮属于当前功能范围；不要反复插拔 USB 或添加不存在的参数来尝试启用。

## 常见现象

| 现象 / 日志 | 含义与处理 |
| --- | --- |
| 启动出现 `Orvyn status indicator initialization failed`，流程仍运行 | 默认允许缺灯继续。根据后面的具体异常检查依赖、设备、权限和应用响应；无灯期间按终端阶段提示操作 |
| `No module named 'hid'` 或 `USB access requires a Python hidapi binding` | 当前项目环境缺少提供 `hid` 模块的 `hidapi`；按下方补齐，不要装到别的 Python / Conda 环境 |
| 枚举列表为空 | 未找到 VID/PID `36b7:2030` 的设备；检查供电、数据线、USB 接口和设备枚举 |
| 枚举得到路径，但 `failed to open Orvyn USB HID device` | 检查对应 `/dev/hidraw*` 的用户访问权限、USB 状态和其他使用此灯的进程；不要用 `sudo` 启动整个机器人流程掩盖环境问题 |
| 能打开 USB，但报 `Command ... timeout` 或 Runtime 状态错误 | 固件 / 运行中的设备应用可能不支持当前 LED 应用协议，或设备响应失败；查询 Runtime / 应用，再核对设备侧版本，不能仅凭枚举成功判定正常 |
| 运行中出现 `cue ... failed` 后不再变灯 | 适配器已禁用，当前会话不会自动重连；灯可能停留在旧颜色。结束当前流程、修复后重新启动 |
| 设了 `backend=orvyn`，设备失败却没有停止启动 | `orvyn` 只是选择后端；还需 `required=true` 才要求初始化成功 |
| 设了 `required=true`，运行中拔灯仍继续 | 这是现有行为：required 约束初始化，不是在线监护；运行中更新失败仍降级 |
| `backend=none` 与 `required=true` 同时设置仍无灯 | none 返回空实现，required 不会重新启用硬件；需要灯时改为 auto / orvyn |
| 灯色看似与数据记录不一致 | 绿常亮也可能是遥操作或本地 DAgger 保存阶段保留的旧提示；红常亮是完成 / 退出，RECAP S/F 不靠灯色区分。查看对应流程灯序和终端保存结果 |
| 程序结束后灯灭 | 正常 close 会休眠；不代表机器人已经断电 |
| 异常退出后灯仍亮 / 停在旧色 | 强制结束、清理被异常打断或 USB 通信失败时可能未完成休眠；先处理程序与硬件状态，再单独检查灯，不把旧灯效当作当前状态 |

## 检查依赖和 USB

所有命令从仓库根目录运行。先结束使用此灯的采集 / 控制进程；下面的查询不启动机械臂。

```bash
uv run --no-sync python -c 'import sys, hid; print(sys.executable); print(hid.__file__)'
lsusb -d 36b7:2030
uv run --no-sync python -m lerobot.light.orvyn_device devices
```

缺少 `hidapi` 时，当前锁文件通过 `gamepad` extra 提供；该 extra 也会安装 pygame。补充到已有项目环境：

```bash
uv sync --locked --inexact --extra gamepad
```

`--inexact` 保留已有 Piper / 训练依赖。若 Python 路径或动态库来自旧 Conda 环境，先按[uv / Conda 环境排错](uv-conda.md)处理；日常入口仍用 `uv run --no-sync`。

权限问题需定位枚举结果对应的 HID 节点，检查用户及组访问权限、设备是否被其他进程使用，再由机器维护者配置针对该设备的 udev 规则。设备枚举成功本身不保证有打开权限。

## USB 正常但灯命令超时

先查询运行时和设备应用，不执行安装、删除或重启：

```bash
uv run --no-sync python -m lerobot.light.orvyn_device info
uv run --no-sync python -m lerobot.light.orvyn_device list
```

需要指定某一只设备时，把 `devices` 输出中的实际路径完整复制进变量；这是低层诊断选项：

```bash
HID_PATH='替换为 devices 输出中的完整 path'
uv run --no-sync python -m lerobot.light.orvyn_device --path="$HID_PATH" --timeout=2000 info
```

业务配置没有 `status_indicator.path`，多灯环境不能用这个参数强行选择；现场采集先确保只连接目标设备。CLI `status` 读取的是安装进度，灯的颜色 / 模式 / 休眠状态请用[单灯检查](../workflows/tools/status-indicator.md#单独检查灯)中的 `get_status()`。

确认协议和 USB 通信正常后，确有慢响应时才考虑提高启动参数 `status_indicator.timeout_ms`。灯命令是同步调用，增大超时会增加阶段切换等待时间，不能用它修复权限或协议错误。

## 亮度和休眠

模式亮度 `brightness` 配置范围为 `0–100`，实际显示还受设备持久化 `max_brightness` 限制；`base_brightness` 范围为 `0–30`，且不能超过该上限。出现 `Base brightness cannot exceed max brightness` 时，先查询设备配置，降低底色亮度，不盲目调高设备上限。

默认底色为 `0`，未选择的灯应不显示底色；若自行设置了非零底色，其他颜色可能仍微亮。设 `brightness=0` 也会让阶段灯难以辨认。

设备 API 支持自动休眠，但业务适配器没有调用 `set_sleep_config()` 去关闭或重设它。若空闲一段时间后灯灭，先查 `auto_sleep_enabled`、`sleep_timeout_s`、`sleeping` 等当前设备字段，再核对设备应用的休眠 / 唤醒行为。业务退出则会显式调用 `sleep_now()`。

## 记录问题时保留什么

保留实际启动入口、`status_indicator.*` 参数、完整异常、`sys.executable` / `hid.__file__`、USB 枚举与 Runtime / 应用查询结果，以及出问题时的操作阶段。说明是初始化失败、运行中更新失败还是退出休眠失败。软件测试通过只能证明映射和降级逻辑，不证明 USB 权限、设备固件或现场灯效已经验证。
