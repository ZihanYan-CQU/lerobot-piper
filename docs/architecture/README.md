# 代码架构与数据语义

[返回文档首页](../README.md) · [操作流程](../workflows/README.md) · [问题排查](../troubleshooting/README.md) · [研究记录](../research/README.md)

这里回答代码如何组织、数据怎样流动、状态和参数有什么含义。复制启动命令请进入对应流程；历史日志、故障诊断和待验证实验放在各自分类。

| 想理解的内容 | 文档 | 对应操作 |
| --- | --- | --- |
| LeRobot 全仓库模块与数采 → 训练 → 部署的数据流 | [总体架构](overview.md) | [全部操作流程](../workflows/README.md) |
| 主臂示教、录制 / 复位 / 保存、相机与数据命名 | [人工示教数采](recording.md) | [Piper 数采](../workflows/recording/README.md) |
| 采后审查、Qt 合并预览、工作线程与编辑后端 | [数据工具结构](dataset-tools.md) | [数据处理](../workflows/datasets/README.md) |
| 服务端、客户端与 shared 的依赖边界 | [异步推理目录](async-inference.md) | [远端连接与迁移](../workflows/deployment/connection.md) |
| 请求槽、动作计划、guided/trained RTC、插值与复位 | [远程部署架构](deployment.md) | [真机部署](../workflows/deployment/README.md) |
| 同机 GPU 推理、传感器线程与动作时效 | [单机本地推理](local-inference.md) | [本地部署](../workflows/deployment/local.md) |
| PI05 图像处理、精度档位、编译与连接预热 | [PI05 推理架构](pi05-inference.md) | [部署配置](../workflows/deployment/configuration.md) |
| 人工接管状态机、双臂 CAN、纠正 episode | [DAgger 架构](dagger.md) | [DAgger 采集](../workflows/dagger/README.md) |
| 完整 rollout、结果标签与模型身份 | [RECAP 架构](recap.md) | [RECAP 采集](../workflows/recap/README.md) |
| checkpoint 统计量、干预比例、epoch、padding loss | [训练架构](training.md) | [训练与续训](../workflows/training/README.md) |
| RLinf 后端、RLT 特征与 actor-critic 数据约定 | [RLinf / RLT 架构](rlt.md) | [RLT 流程](../workflows/rlt/README.md) |
| MuJoCo 插件、动作映射与仿真数据 | [Piper 仿真架构](simulation.md) | [仿真流程](../workflows/simulation/README.md) |
| 语义状态到 USB 灯态、故障降级 | [指示灯架构](status-indicator.md) | [灯态速查](../workflows/tools/status-indicator.md) |
| 设备识别、绑定校验与图形授权 | [设备绑定工具结构](device-bindings.md) | [设备绑定](../workflows/tools/device-bindings.md) |
| GitHub 上传后端、Qt 窗口与参数保存 | [上传工具结构](github-upload.md) | [上传工具](../workflows/tools/github-upload.md) |
| WN-A100 GUI 与脚本的维护入口 | [WN-A100 工具结构](wn-a100.md) | [工具操作](../workflows/tools/wn-a100.md) |

新功能的实现边界更新在相应专题，完整运行配方更新在流程文档。实验参数和历史验证结果保留原日期；不会因文档整理而变成新的实测结论。
