# 状态指示灯：使用与状态速查

[工具目录](README.md) · [实现结构](../../architecture/status-indicator.md) · [连接与异常排查](../../troubleshooting/status-indicator.md)

Orvyn USB 灯把当前流程显示为绿、橙、红三种颜色及常亮 / 闪烁。它由机器人电脑上的采集或控制进程驱动，不需要单独启动灯服务。默认尝试连接，缺少设备或依赖时允许无灯运行；阶段、保存结果和故障原因仍以终端信息为准。

## 哪些流程会控制灯

| 流程 | 已接入入口 | 完整操作命令 |
| --- | --- | --- |
| 遥操作 | `lerobot.scripts.lerobot_teleoperate` | [遥操作配方](../recording/two-camera.md#1-先验证遥操作) |
| 人工演示数采 | `lerobot.scripts.lerobot_record` | [三相机现场指南](../recording/single-arm.md) |
| 远端 DAgger | `lerobot.async_inference.client.robot_client_dagger` | [DAgger 流程](../dagger/README.md) |
| 单机 DAgger | `lerobot.async_inference_local.scripts.dagger` | [本地 DAgger](../dagger/local.md) |
| RECAP 整轮采集 | `lerobot.async_inference.client.robot_client_recap` | [RECAP 流程](../recap/README.md) |

普通 `robot_client_deployment`、单机 `scripts.deploy` 和 `lerobot-rollout` 当前没有接入这套状态灯配置。DAgger / RECAP 的自主执行会亮灯，不代表所有真机部署入口都支持；不要把下面的 `status_indicator.*` 参数加到未接入的入口上。

## 按颜色快速判断

默认亮度为 `10`，闪烁周期为 `800 ms`，底色亮度为 `0`。颜色是当前流程的提示，同一种灯效可能对应多个阶段。

| 灯效 | 对应语义 | 操作者如何理解 |
| --- | --- | --- |
| 绿灯常亮 | `TELEOPERATING` / `RECORDING` / `CORRECTING` | 遥操作、正在录演示、或人工接管纠正；是否在写数据要看当前入口及终端 |
| 绿灯闪烁 | `RECORDING_PENDING` / `AUTONOMOUS` | 数采开始前倒计时，或 DAgger / RECAP 的模型自主执行 |
| 橙灯闪烁 | `PAUSED` / `RESETTING` / `WARNING` | 暂停、正在复位，或温度 / 接管对齐警告；查看终端再操作 |
| 橙灯常亮 | `RESET_REQUIRED` | 数采等待恢复场景；远端 DAgger / RECAP 的准备、复位及等待下一轮；遥操作退出阶段也会短暂使用 |
| 红灯闪烁 | `FAULT` / `DISCONNECTING` | DAgger / RECAP 控制或记录故障，或普通数采正在断连收尾；红闪不总是故障 |
| 红灯常亮 | `COMPLETE` / `EXIT` | 会话完成或准备退出，不表示这一条任务结果是 `failure` |
| 熄灯 | 正常 `close()` 会让设备休眠；禁用、连接失败也可能无灯 | 不能据此判断程序是否运行、机器人是否已经停机 |

这套状态适配器没有调用蜂鸣器；提示音或语音需按各流程自己的配置理解。

## 各流程的实际灯序

### 遥操作和普通数采

遥操作连接成功后为**绿常亮**；启用温度监控、机器人支持该功能且触发过热回调时变为**橙闪**。退出阶段会显示**橙常亮**，之后关闭灯。

普通数采每条演示的顺序是：

| 阶段 | 灯效 |
| --- | --- |
| 开始前 3 秒倒计时 | 绿闪 |
| 正在记录 episode | 绿常亮 |
| 下一条 / 重录前的 3 秒回零倒计时及自动回零 | 橙闪 |
| 等待恢复物体和场景，尚未开始下一条 | 橙常亮 |
| 下一条开始前倒计时 | 回到绿闪 |
| 正常结束采集循环 | 红常亮 |
| 数据收尾后，断开机械臂及后续退出清理 | 红闪 |
| 指示灯连接正常且完成关闭 | 休眠熄灯 |

最后一条和停止采集会跳过中间的复位阶段。温度回调可将当时灯效改成橙闪。灯不是保存确认：当前实现可能先复位 / 等待恢复场景，再保存 episode；保存成功看终端及数据集文件，详见[现场流程](../recording/single-arm.md#4-一条演示从开始到保存的完整过程)。

### 远端 DAgger 与 RECAP

| 阶段 | 灯效 |
| --- | --- |
| 启动就绪、复位前后、等待恢复场景 | 橙常亮 |
| 模型自主执行 | 绿闪 |
| 暂停；远端接管对齐被拒绝但已恢复暂停 | 橙闪 |
| 人工纠正 / 接管 | 绿常亮 |
| DAgger 结束纠正并保存；RECAP 结束接管或 S/F 提交保存 | 橙闪 |
| 控制、采集、保存或复位进入 FAULT | 红闪 |
| 达到本次采集目标或退出 | 红常亮，正常关闭后熄灯 |

远端复位前后都使用橙常亮，没有单独的“正在复位”闪烁提示。RECAP 的 S/F 是数据结果标签，**不会分别点亮绿灯 / 红灯**；两者提交保存时都显示橙闪，随后复位等待，达到数量目标则退出。

### 单机 DAgger

| 阶段 | 灯效 |
| --- | --- |
| 启动就绪 / 暂停 / 复位成功后 | 橙闪 |
| 模型自主执行 | 绿闪 |
| 开始人工纠正 | 绿常亮 |
| 接管对齐检查未通过 | 橙闪，保持暂停 |
| 纠正结束、保存文件期间 | 当前实现没有独立保存灯效，可能继续显示之前的绿常亮 |
| 复位中 | 橙闪 |
| 可恢复的纠正 / 采集 / 复位故障进入 FAULT | 红闪 |
| 成功保存数量达到目标 | 红常亮，随后退出 |
| 普通停止 | 没有单独发送 `EXIT` 灯效；正常关闭后熄灯 |

本地终止性推理 / 控制故障没有统一保证发送红闪，仍须查看 `/status`、异常和退出日志；不能用“没亮红灯”证明控制正常。

## 配置开关

在上方已接入流程的**完整启动命令末尾**追加参数。以下是参数片段，不是独立启动命令；仍使用各流程原来的 `uv run --no-sync ...` 入口。

```text
--status_indicator.backend=none
```

此项完全禁用灯，不导入 HID 驱动、不尝试连接。正常使用默认 `auto` 即可；需要明确指定 Orvyn 并要求启动时必须有灯，可以追加：

```text
--status_indicator.backend=orvyn --status_indicator.required=true
```

`auto` 和 `orvyn` 当前选择同一个适配器；`orvyn` 本身不强制连接成功。`required=true` 只对**初始化失败**生效，运行中灯通信失败仍会记录警告并禁用灯；它不是持续在线检测。`backend=none` 会直接禁用，不能再靠 `required=true` 要求有灯。

| 参数 | 默认值 | 含义与范围 |
| --- | --- | --- |
| `status_indicator.backend` | `auto` | `auto` / `orvyn` / `none` |
| `status_indicator.required` | `false` | 初始化失败是否阻止正常启动 |
| `status_indicator.timeout_ms` | `2000` | HID / 应用响应超时，正整数毫秒 |
| `status_indicator.brightness` | `10` | 模式亮度百分比，配置允许 `0–100`，实际还受设备上限限制 |
| `status_indicator.blink_period_ms` | `800` | 闪烁周期毫秒；保持正整数且不超过设备的 16 位周期范围 `65535` |
| `status_indicator.base_brightness` | `0` | 底色亮度百分比，`0–30`，不可超过设备当前 `max_brightness` |

## 单独检查灯

从仓库根目录执行，先结束其他正在使用此灯的采集 / 控制程序。下方只操作 USB 灯，不启动机器人。

当前 `hidapi` 由 `gamepad` extra 提供，`piper-client` 不包含它。已有项目环境缺少此依赖时，按锁文件补齐：

```bash
uv sync --locked --inexact --extra gamepad
uv run --no-sync python -c 'import hid; print(hid.__file__)'
```

这个 extra 同时带入 pygame；`--inexact` 保留既有采集 / 训练依赖。完整环境组合继续按[环境安装](../setup/README.md)维护，不需要激活旧 Conda 环境。

只枚举目标 USB 设备，不改变灯色：

```bash
uv run --no-sync python -m lerobot.light.orvyn_device devices
```

查看设备当前灯色、亮度、周期及休眠状态：

```bash
uv run --no-sync python - <<'PY'
from pprint import pprint
from lerobot.light.light_api import OrvynDevice

with OrvynDevice(timeout_ms=2000) as light:
    pprint(light.get_status())
PY
```

需要确认六种灯效时运行以下自检。它会将底色亮度设为默认 `0`（设备基础配置）、模式亮度设为 `10`，结束时让灯休眠；不会自动恢复之前的灯配置。

```bash
uv run --no-sync python - <<'PY'
import time
from lerobot.light import StatusCue, StatusIndicatorConfig, make_status_indicator

light = make_status_indicator(StatusIndicatorConfig(backend="orvyn", required=True))
try:
    light.start()
    for cue in (
        StatusCue.RECORDING,          # 绿常亮
        StatusCue.AUTONOMOUS,         # 绿闪
        StatusCue.PAUSED,             # 橙闪
        StatusCue.RESET_REQUIRED,     # 橙常亮
        StatusCue.FAULT,              # 红闪
        StatusCue.COMPLETE,           # 红常亮
    ):
        print(cue.value, flush=True)
        light.show(cue)
        time.sleep(2)
finally:
    light.close()
PY
```

如果无灯、无响应、缺少 `hid` 或出现权限错误，按[状态灯排错](../../troubleshooting/status-indicator.md)处理。上面的命令供现场验证使用，文档整理期间没有连接硬件验证灯效。
