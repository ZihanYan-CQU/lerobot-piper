# 远端异步推理代码

本目录实现服务器推理与机器人客户端控制。`shared/` 保存两端共用配置、通信对象及 RTC 辅助逻辑；`server/` 负责策略加载与推理；`client/` 负责机器人控制、人工接管和数据记录。

| 要查什么 | 文档入口 |
| --- | --- |
| 真机部署、现有 JSON 快速启动 | [部署指令](../../../docs/workflows/deployment/README.md) |
| DAgger 人工纠正采集 | [DAgger 指令](../../../docs/workflows/dagger/README.md) |
| RECAP 整轮采集与结果标注 | [RECAP 指令](../../../docs/workflows/recap/README.md) |
| 配置覆盖、GPU、图像模式 | [部署配置](../../../docs/workflows/deployment/configuration.md) |
| SSH、代码迁移、旧模块替换 | [连接与迁移](../../../docs/workflows/deployment/connection.md) |
| 模块职责与依赖边界 | [远端推理架构](../../../docs/architecture/async-inference.md) |
| RTC 调度与动作衔接 | [部署架构](../../../docs/architecture/deployment.md) |
| 日志、异常与恢复 | [集中排错](../../../docs/troubleshooting/README.md) |

命令以 `lerobot.async_inference.server.*` / `lerobot.async_inference.client.*` 为入口，均从仓库根目录运行。单机不经过远端服务的运行时见 [async_inference_local](../async_inference_local/README.md)。
