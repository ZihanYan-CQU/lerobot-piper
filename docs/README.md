<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# LeRobot / Piper 文档导航

日常找命令从这里开始。每个流程单独维护操作顺序和完整命令；代码原理、故障记录和研究材料分别归档。所有命令默认在对应机器的仓库根目录执行。

## 常用流程

| 现在要做什么 | 指令文档 |
| --- | --- |
| 安装依赖、迁移机器、确认目录 | [环境准备](workflows/setup/README.md) |
| 确认实际 Python、排查 uv / Conda 混用、查旧入口退役记录 | [环境整理记录](troubleshooting/uv-conda.md) |
| 绑定相机和主从臂 CAN | [设备绑定](workflows/tools/device-bindings.md) |
| 看懂绿 / 橙 / 红灯及闪烁、启用或关闭指示灯 | [指示灯作用与配置](workflows/tools/status-indicator.md) |
| 遥操作、采集原始示范 | [Piper 数采入口](workflows/recording/README.md) · [单臂现场指南](workflows/recording/single-arm.md) |
| 看数据、删坏 episode、合并、上传 / 下载 | [数据处理](workflows/datasets/README.md) |
| 下载模型、单卡 / 多卡微调 | [模型训练](workflows/training/README.md) |
| RLinf 后端、RLT 两阶段训练与导出 | [RLT 操作入口](workflows/rlt/README.md) |
| MuJoCo 仿真遥操作、数采、训练和评估 | [Piper 仿真](workflows/simulation/README.md) · [短流程](workflows/simulation/smoke.md) |
| 启动机器人执行模型 | [真机部署：远程 / 单机](workflows/deployment/README.md) |
| 自主执行中接管并记录纠正数据 | [DAgger 纠正采集](workflows/dagger/README.md) |
| 保存整轮任务、成功失败结果与模型版本 | [RECAP 完整任务采集](workflows/recap/README.md) |
| 检查服务器 GPU、训练日志与磁盘 | [WN-A100 监控](workflows/tools/wn-a100.md) |
| 用界面上传代码 | [GitHub 上传工具](workflows/tools/github-upload.md) |
| 对比 Piper 模型的实际任务表现 | [Piper 真机评估](workflows/evaluation/README.md) |

常见顺序：**环境 → 数采 → 数据处理 → 训练 → 真机部署 → DAgger → 合并后再训练**。RECAP 另行记录完整任务，其 value/advantage 与策略训练尚未接通；已实现的 RLT 训练使用独立流程，不能直接将 RECAP 数据视为 RLT replay。

这些指令验证到哪一步、当前机器还缺什么，见[指令核查记录（2026-11-07）](maintenance/command-audit-2026-11-07.md)。

## 原理、问题和其他资料

| 分类 | 内容 |
| --- | --- |
| [代码架构](architecture/README.md) | 仓库模块、远程 / 本地推理、RTC、RLT、仿真、指示灯和工具实现 |
| [问题排查与记录](troubleshooting/README.md) | 环境、设备、连接、CUDA、精度 / 编译、延迟及历史修复记录 |
| [研究与实验](research/README.md) | 论文对照、实验预算、方案评估和尚未实现的建议 |
| [参数与经验参考](reference/README.md) | 数据质量、策略选择、训练步数与 batch |
| [辅助工具](workflows/tools/README.md) | 设备绑定、指示灯、监控、数据工具和代码上传入口 |
| [文档维护](maintenance/README.md) | 内容归属、移动规则、网站构建与预览 |

通用教程仍在 [docs/source](source/)，网站导航由 [_toctree.yml](source/_toctree.yml) 管理。常用入口：[安装](source/installation.mdx)、[模仿学习](source/il_robots.mdx)、[Pi0.5](source/pi05.mdx)、[多 GPU](source/multi_gpu_training.mdx)、[RTC](source/rtc.mdx)。开发约定见 [AGENTS.md](../AGENTS.md)。

## 旧章节入口

以下锚点供旧书签继续定位；正文已迁移到对应流程或主题页。

<a id="deployment-quickstart"></a>

[Deployment 快速启动](workflows/deployment/README.md)

<a id="环境与路径"></a>

[环境与路径](workflows/setup/README.md)

<a id="训练"></a>

[训练](workflows/training/README.md)

<a id="服务端推理"></a>
<a id="客户端推理"></a>

[远程推理双端命令](workflows/deployment/README.md)

<a id="local-tools"></a>
<a id="工具与常用检查"></a>

[工具与常用检查](workflows/tools/README.md)

<a id="troubleshooting"></a>
<a id="登录日志与排错"></a>

[登录、日志与排错](troubleshooting/README.md)

<a id="references"></a>
<a id="专题参考"></a>

[专题原理](architecture/README.md)

<a id="documentation"></a>
<a id="doc-maintenance"></a>
<a id="generating-the-documentation"></a>
<a id="文档放在哪里"></a>
<a id="building-the-documentation"></a>
<a id="previewing-the-documentation"></a>
<a id="本地构建与预览"></a>
<a id="adding-a-new-element-to-the-navigation-bar"></a>
<a id="adding-a-new-tutorial"></a>
<a id="新增或调整教程"></a>
<a id="renaming-section-headers-and-moving-sections"></a>
<a id="writing-source-documentation"></a>
<a id="writing-a-multi-line-code-block"></a>
<a id="adding-an-image"></a>

[文档维护与网站构建](maintenance/README.md)
