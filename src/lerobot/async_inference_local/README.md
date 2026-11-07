# 本地推理代码

本目录提供 GPU、机械臂和相机位于同一机器时的独立 PI05 运行时，包括自主 Deployment 与 DAgger 人工纠正采集；当前不含本地 RECAP。模型推理、观测与控制在同一进程中协作，不调用远端 `async_inference` 运行时。

| 要查什么 | 文档入口 |
| --- | --- |
| 启动自主运行、暂停、复位、退出 | [本地部署指令](../../../docs/workflows/deployment/local.md) |
| 双臂接管与纠正数据保存 | [本地 DAgger 指令](../../../docs/workflows/dagger/local.md) |
| 环境与设备准备 | [环境准备](../../../docs/workflows/setup/README.md) |
| 目录、线程、RTC、输入语义与验证边界 | [本地推理架构](../../../docs/architecture/local-inference.md) |
| 远端服务器与客户端方案 | [真机部署目录](../../../docs/workflows/deployment/README.md) |
| 故障与日志 | [集中排错](../../../docs/troubleshooting/README.md) |

入口为 `python -m lerobot.async_inference_local` 和 `python -m lerobot.async_inference_local.scripts.dagger`，完整 `uv run` 参数统一维护在上述流程页。
