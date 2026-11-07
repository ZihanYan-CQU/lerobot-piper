# Piper MuJoCo 仿真插件

本目录实现 `piper_mujoco` Robot/Env 插件与 `piper-sim` CLI，使用独立 uv 项目。
日常从仓库根目录执行直接 uv 命令，完整配方统一在流程文档维护。

| 要查的内容 | 文档 |
| --- | --- |
| 环境、模型准备、主臂遥操作、数采、回放、训练与评估 | [仿真操作指令](../../docs/workflows/simulation/README.md) |
| 数采 → 回放 → 五步训练 → 推理 | [仿真短流程](../../docs/workflows/simulation/smoke.md) |
| 代码职责、动作与数据语义、模型来源和 MIT 许可证 | [仿真实现](../../docs/architecture/simulation.md) |
| 资源与显示、回放限制、真机边界和环境检查 | [仿真排错](../../docs/troubleshooting/simulation.md) |
