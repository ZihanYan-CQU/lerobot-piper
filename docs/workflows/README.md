# 流程指令

[返回文档首页](../README.md)

所有命令默认在对应机器的仓库根目录执行。先打开要做的流程，再按页内顺序执行；示例中的用户名、设备、GPU、数据集和 checkpoint 路径需要按实际情况修改。

| 流程 | 从这里开始 | 完成后 |
| --- | --- | --- |
| 环境准备与迁移 | [环境、依赖和目录](setup/README.md) | 绑定设备或准备训练数据 |
| Piper 遥操作与数采 | [数采指令](recording/README.md) | 审核数据 |
| 数据检查与整理 | [回放、筛选、合并、上传](datasets/README.md) | 训练 |
| 模型训练 | [下载资源、单卡 / 多卡、guided / trained](training/README.md) | 真机部署 |
| RLT 两阶段训练 | [RLinf / RLT 操作入口](rlt/README.md) | 导出模型、核对 replay 与部署条件 |
| Piper MuJoCo 仿真 | [遥操作、数采、训练、评估](simulation/README.md) · [短流程](simulation/smoke.md) | 检查虚拟环境中的数据与模型 |
| 真机部署 | [远程双端 / 单机本地](deployment/README.md) | 评估或采集纠正数据 |
| DAgger 纠正采集 | [自主执行、人工接管、再训练](dagger/README.md) | 合并纠正数据并训练 |
| RECAP 整轮采集 | [完整任务、成功失败标注](recap/README.md) | 保存结果和模型追溯信息 |
| 评估 | [Piper 真机评估](evaluation/README.md) | 比较成功率 |
| 指示灯 | [颜色、闪烁与配置](tools/status-indicator.md) | 在数采或接管流程中判断阶段 |
| 辅助工具 | [设备绑定、服务器监控、代码上传](tools/README.md) | 返回对应流程 |

本地操作文档以 Piper 主从臂为对象，端口、相机与采集配置均按 Piper 流程查阅。

遇到异常查 [问题排查](../troubleshooting/README.md)；理解实现查 [代码架构](../architecture/README.md)；选择策略和训练预算查 [参数参考](../reference/README.md)。
