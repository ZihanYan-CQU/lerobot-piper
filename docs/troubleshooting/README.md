# 问题记录与排查

[返回文档首页](../README.md) · [操作流程](../workflows/README.md) · [代码架构](../architecture/README.md) · [研究记录](../research/README.md)

按现象进入对应文档。诊断参数、日志解释和未解决问题集中放在这里；正常启动命令在操作流程中维护。

| 现象或任务 | 入口 |
| --- | --- |
| 登录、SSH 动态库、HF 镜像、磁盘、依赖环境 | [环境与日志](environment.md) |
| uv 环境、外部变量混用、旧入口退役 | [环境整理记录](uv-conda.md) |
| 普通数采环境、按键、复位、保存与相机问题 | [Piper 数采排错](recording.md) |
| 审查播放、过滤失败、Qt 合并与输出目录 | [数据工具排错](datasets.md) |
| 设备绑定、监控、GitHub 工具问题 | [工具排错](tools.md) |
| 指示灯不亮、灯态误读、USB / HID 连接失败 | [指示灯排错](status-indicator.md) |
| 仿真资源、显示、回放误差与 uv 环境 | [仿真排错](simulation.md) |
| RLT checkpoint、结构参数、replay 和训练边界 | [RLinf / RLT 排错](rlt.md) |
| 不发请求、结果被拒收、队列空、复位失败 | [远程部署排错](deployment.md) |
| 重连段错误、CUDA 非法访问、如何关闭诊断开销 | [远程部署排错](deployment.md) |
| 接管按键、CAN 桥、纠正数据未保存 | [DAgger 排错](dagger.md) |
| RECAP 未完成写盘、续录被拒绝、sidecar 完整性 | [RECAP 异常与恢复](recap.md) |
| 干预采样、统计量、padding、resume 行为 | [训练排错](training.md) |
| 延迟抖动、编译成本、guided 开销、离线 profile | [PI05 性能排查](pi05-performance.md) |

## 按日期保留的诊断记录

| 日期 | 记录 | 阅读边界 |
| --- | --- | --- |
| 2026-10-15/16 | [PI05 / A100 延迟日志排查](pi05_inference_latency_2026-10-15.md) | 用户日志与本地检查，不是 A100 优化后测速 |
| 2026-10-15～30 | [PI05 精度、编译与官方对齐](pi05_official_alignment_2026-10-15.md) | 各轮固定版本、修订和缩小模型验证，按更新日期阅读 |
| 2026-10-15/16 | [传输基准与 guided 开销复查](pi05-performance.md) | 保留测量设备、输入范围及尚未完成的重连验证 |

记录新问题时，写清日期、入口/代码版本、完整非敏感参数、症状、日志证据、已验证结论和仍待验证的事项。历史测试数字不因搬迁而代表当前重新运行通过。
