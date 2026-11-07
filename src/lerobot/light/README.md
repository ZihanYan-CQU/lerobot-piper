# 状态指示灯模块

`lerobot.light` 为遥操作、数采、DAgger 和 RECAP 提供可选的阶段提示。应用通过 `StatusCue` 和 `start()` / `show()` / `close()` 接口使用指示灯；默认 Orvyn 后端缺少设备时允许无灯运行，`backend=none` 显式禁用。

| 要查什么 | 文档 |
| --- | --- |
| 颜色与闪烁含义、各流程灯序、开关与单灯检查命令 | [使用与状态速查](../../../docs/workflows/tools/status-indicator.md) |
| 模块分工、调用入口、协议及失败行为 | [实现结构](../../../docs/architecture/status-indicator.md) |
| 缺依赖、USB / 权限、超时、亮度和旧灯色 | [问题排查](../../../docs/troubleshooting/status-indicator.md) |

正式实现为本目录的 `status_indicator.py`、`light_api.py`、`orvyn_device.py`。具体支持范围见使用文档，普通部署入口当前未统一接入。
